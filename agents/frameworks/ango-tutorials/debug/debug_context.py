from typing import Dict, Any, Optional

from rich.pretty import pprint

from pydantic import BaseModel, Field

from agno.agent import Agent
from agno.models.google import Gemini
from agno.db.sqlite import SqliteDb

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
DB_FILE = "context_data.db"

# Functions
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

# Data structures
class ResponseModel(BaseModel):
    response: str = Field(description="LLM response")
    score: int = Field(description="Confidence of the response provided by LLM based on the reason. 0-1. 0 is lowest confidence on the response. 1 is highest confidence on the response")
    reason: str = Field(description="Reason why LLM provide the response")

# Functions
def create_agent():
    agent = Agent(
        model=Gemini(id="gemini-3-flash-preview"),
        # system prompt
        # role / instructions / output
        description="You are a helpful AI assistant",
        instructions=["Provide the best answer to the user questions given your knowledge",
                      "Evaluate your response providing score and reason",
                      "IMPORTANT GUIDELINES:",
                      "- When a user asks about their preferences, use the get_user_preferences or list_all_preferences tool.",
                      "- When a user tells you about a new preference or wants to update one, use the update_preference tool.",
                      "- Provide personalized responses based on stored preferences.",
                      "- Be conversational and friendly.",
                      "- If asked about something not in preferences, politely say you don't have that information stored.",
                      "TOOL USAGE:",
                      "- get_user_preferences: Shows all current preferences",
                      "- update_preference: Updates a single preference (takes preference_key and preference_value)",
                      "- list_all_preferences: Lists all preferences in a formatted way"],
        # output
        # disable output_schema for tool calls
        #output_schema=ResponseModel,
        # system prompt extra
        add_datetime_to_context=True,
        # tools
        tools=[get_user_preferences, update_preference, list_all_preferences],
        tool_call_limit=5,

        # session storage
        db=SqliteDb(db_file=DB_FILE),
        # session history
        add_history_to_context=True,
        num_history_runs=2,
        # session state
        add_session_state_to_context=True,
        # Initialize session state with default preferences
        session_state={
            "preferences": {
                "favorite_color": "blue",
                "favorite_food": "pizza",
                "preferred_language": "English"
            }
        },
       
        # debug
        debug_mode=True,
    )
    return agent


def execute_turn(agent, turn: str, user_id: Optional[str]=None, session_id: Optional[str]=None):
    response = agent.run(turn, user_id=user_id, session_id=session_id)
    pprint(f"Run ID: {response.run_id}\nAgent ID: {response.agent_id}\nSession ID: {response.session_id}\nContent: {response.content}")
    return response

agent = create_agent()

session_id = 'session_01'
user_id = 'user_01'

while True:
    user_input = input("\nYou: ")
    if user_input == "/exit":
        break
    execute_turn(agent, user_input, user_id, session_id)
