"""
Persistent Session Agent with Agno and Google Gemini
A personal assistant that remembers user preferences across conversations.
"""
from typing import Dict, Any
import os

from agno.agent import Agent
from agno.models.google import Gemini
from agno.db.sqlite import SqliteDb

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
USER_ID = "user_12345"
DB_FILE = "session_data.db"

# Verify API key is loaded
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables")


# Custom functions to manage preferences in session state
def get_user_preferences(session_state: Dict[str, Any]) -> str:
    """Retrieve user preferences from the session state.
    
    Args:
        session_state: Automatically injected by Agno
        
    Returns:
        str: Formatted preference information
    """
    preferences = session_state.get("preferences", {})
    
    pref_list = [f"{k}: {v}" for k, v in preferences.items()]
    return f"Current preferences:\n" + "\n".join(f"- {p}" for p in pref_list)


def update_preference(
    session_state: Dict[str, Any],
    preference_key: str, 
    preference_value: str
) -> str:
    """Update or add a user preference to the session state.
    
    Args:
        session_state: Automatically injected by Agno
        preference_key: The preference name (e.g., 'favorite_color', 'favorite_food')
        preference_value: The new preference value
        
    Returns:
        str: Operation status message
    """
    # Get current preferences or initialize empty dict
    if "preferences" not in session_state:
        session_state["preferences"] = {}
    
    # Update the specific preference
    old_value = session_state["preferences"].get(preference_key, "not set")
    session_state["preferences"][preference_key] = preference_value
    
    return (
        f"✓ Updated {preference_key} from '{old_value}' to '{preference_value}'.\n\n"
        f"All preferences: {session_state['preferences']}"
    )


def list_all_preferences(session_state: Dict[str, Any]) -> str:
    """List all stored preferences.
    
    Args:
        session_state: Automatically injected by Agno
        
    Returns:
        str: Formatted list of all preferences
    """
    preferences = session_state.get("preferences", {})
    
    if not preferences:
        return "No preferences stored yet."
    
    pref_list = "\n".join([f"  • {k}: {v}" for k, v in preferences.items()])
    return f"Your stored preferences:\n{pref_list}"


# Initialize Agent with Gemini and persistent storage
agent = Agent(
    # Use Google Gemini model
    model=Gemini(id="gemini-3-flash-preview"),

    # Instructions for the agent
    instructions=[
        "You are a helpful personal assistant with access to user preferences and history.",
        "",
        "IMPORTANT GUIDELINES:",
        "- When a user asks about their preferences, use the get_user_preferences or list_all_preferences tool.",
        "- When a user tells you about a new preference or wants to update one, use the update_preference tool.",
        "- Provide personalized responses based on stored preferences.",
        "- Be conversational and friendly.",
        "- If asked about something not in preferences, politely say you don't have that information stored.",
        "",
        "TOOL USAGE:",
        "- get_user_preferences: Shows all current preferences",
        "- update_preference: Updates a single preference (takes preference_key and preference_value)",
        "- list_all_preferences: Lists all preferences in a formatted way",
    ],
    
    # Add tools for preference management
    tools=[get_user_preferences, update_preference, list_all_preferences],

    # Add database for session persistence
    db=SqliteDb(db_file=DB_FILE),
        
    # Initialize session state with default preferences
    session_state={
        "preferences": {
            "favorite_color": "blue",
            "favorite_food": "pizza",
            "preferred_language": "English"
        }
    },
    
    # Add session history to context
    add_history_to_context=True,
    num_history_runs=5,  # Include last 5 conversation turns
    # Add session state to context (available in instructions)
    add_session_state_to_context=True,
        
    # Enable markdown formatting
    markdown=True,
    # Enable debug_mode
    debug_mode=True,
)

def display_session_info(agent: Agent, user_id: str, session_id: str):
    """Display current session information."""
    print("\n" + "=" * 80)
    print("SESSION INFORMATION")
    print("=" * 80)
    print(f"User ID: {user_id}")
    print(f"Session ID: {session_id}")
    print(f"Database: {DB_FILE}")

    # Get current session state (may not exist yet)
    try:
        session_state = agent.get_session_state(session_id)
        if session_state and "preferences" in session_state:
            print(f"Current Preferences: {session_state['preferences']}")
    except Exception:
        print("Session not yet created (will be initialized on first message)")
    print("=" * 80 + "\n")


def main():
    """Main interaction loop."""
    print("\n" + "=" * 80)
    print("AGNO + GEMINI: Personal Assistant with Persistent Preferences")
    print("=" * 80)
    print("Type 'exit' or 'quit' to end the conversation")
    print("Type 'info' to see session information")
    print("=" * 80 + "\n")
    
    session_id = "main_session"
    
    # Display initial session info
    display_session_info(agent, USER_ID, session_id)
    
    # Example prompts to try
    
    # Main interaction loop
    while True:
        try:
            user_input = input("You: ")
            
            if user_input.lower() in ["exit", "quit"]:
                print("\n" + "=" * 80)
                print("Session saved. Have a great day!")
                
                # Display final preferences
                display_session_info(agent, USER_ID, session_id)
                print("=" * 80 + "\n")
                break
            
            if user_input.lower() == "info":
                display_session_info(agent, USER_ID, session_id)
                continue
            
            if not user_input.strip():
                continue
            
            print()  # Add spacing before assistant response
            
            # Send message with user_id and session_id
            agent.print_response(
                user_input,
                stream=True,
                user_id=USER_ID,
                session_id=session_id
            )
            
            print()  # Add spacing after assistant response
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Saving session...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Continuing...\n")


if __name__ == "__main__":
    main()
