"""
Persistent Session Agent with Agno and Google Gemini
A personal assistant that remembers user preferences across conversations.
"""
from typing import Dict, Any
import os

from agno.agent import Agent
from agno.models.google import Gemini
from agno.db.sqlite import SqliteDb
from agno.memory import MemoryManager

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
AGENT_ID = "agent_12345"
USER_ID = "user_12345"
DB_FILE = "memory_data.db"

# Verify API key is loaded
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

# Setup SQLite database
db = SqliteDb(db_file=DB_FILE)

# Setup your Memory Manager, to adjust how memories are created
memory_manager = MemoryManager(
    db=db,
    # Select the model used for memory creation and updates. If unset, the default model of the Agent is used.
    model=Gemini(id="gemini-3-flash-preview"),
    additional_instructions="Summarize any big/long/accumulated memory before storing it.",
)

# Initialize Agent with Gemini and persistent storage
agent = Agent(
    # Use Google Gemini model
    model=Gemini(id="gemini-3-flash-preview"),

    # Instructions for the agent
    instructions=[
        "You are a helpful personal assistant with access to user memories and history.",
        "",
        "IMPORTANT GUIDELINES:",
        "- Provide personalized responses based on stored memories.",
        "- Be conversational and friendly.",
        "",
    ],
    
    # Add database for session persistence
    db=SqliteDb(db_file=DB_FILE),

    # Add session history to context
    add_history_to_context=True,
    num_history_runs=5,  # Include last 5 conversation turns
        
    # Add memory system
    memory_manager=memory_manager,
    enable_user_memories=True,  # Automatically extract and store user insights    
    add_memories_to_context=True,

    # Enable markdown formatting
    markdown=True,
)


def display_user_memories(agent: Agent, user_id: str):
    """Display user memories."""
    memories = agent.get_user_memories(user_id=user_id)
    if memories:
        print("\n📝 Learned Memories:")
        for mem in memories:
            print(f"  • {mem.memory}")


def main():
    """Main loop using agent with automatic memory."""
    
    print("\n" + "=" * 80)
    print("AGNO + GEMINI: Assistant with Automatic Memory")
    print("=" * 80)
    print("This agent automatically learns from conversation!")
    print("Type 'exit' or 'quit' to end")
    print("=" * 80 + "\n")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit"]:
            # Display learned memories
            try:
                print("\n" + "=" * 80)
                print("Session saved. Have a great day!")
                
                # Display final preferences
                display_user_memories(agent, USER_ID)
                print("=" * 80 + "\n")

            except:
                pass
            print("\nGoodbye!\n")
            break
        
        if user_input.lower() == "info":
            display_user_memories(agent, USER_ID)
            continue

        if not user_input.strip():
            continue
        
        print()
        agent.print_response(
            user_input,
            stream=True,
            agent_id=AGENT_ID,
            user_id=USER_ID,
        )
        print()


if __name__ == "__main__":
    main()
