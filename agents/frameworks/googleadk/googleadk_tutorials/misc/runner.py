import asyncio
from datetime import datetime
from typing import Dict, Any

from google.adk.sessions import InMemorySessionService
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from dotenv import load_dotenv

load_dotenv()

# Custom functions to manage session state
def get_user_state(tool_context: ToolContext) -> Dict[str, Any]:
    """Retrieve all user information from the session state.
    
    Args:
        tool_context: Automatically injected by ADK
        
    Returns:
        dict: User profile including name and preferences
    """
    preferences = tool_context.state.get("preferences", {})
    
    return {
        "status": "success",
        "preferences": preferences,
        "retrieved_at": datetime.now().isoformat()
    }

def update_preference(
    preference_key: str, 
    preference_value: str, 
    tool_context: ToolContext
) -> Dict[str, Any]:
    """Update or add a user preference to the session state.
    
    Args:
        preference_key: The preference name (e.g., 'favorite_color', 'favorite_food')
        preference_value: The new preference value
        tool_context: Automatically injected by ADK
        
    Returns:
        dict: Operation status and updated preferences
    """
    # Get current preferences or initialize empty dict
    preferences = tool_context.state.get("preferences", {})
    
    # Update the specific preference
    preferences[preference_key] = preference_value
    
    # Save back to state
    tool_context.state["preferences"] = preferences
    
    return {
        "status": "success",
        "message": f"Updated {preference_key} to {preference_value}",
        "updated_preferences": preferences,
        "updated_at": datetime.now().isoformat()
    }

def init_state():
    return {
        "preferences": {
            "favorite_color": "blue",
            "favorite_food": "pizza"
        }
    }

async def create_session(session_service: InMemorySessionService, app_name: str, user_id: str):
    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        state=init_state()
    )
    return session

async def get_session(session_service: InMemorySessionService, app_name: str, user_id: str, session_id: str):
    session = await session_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )
    return session

def create_agent(app_name: str):
    agent = Agent(
        name=app_name,
        model="gemini-3-flash-preview",
        instruction="""
        You are a helpful personal assistant with access to user preferences and state information.
        
        IMPORTANT GUIDELINES:
        - When a user asks about their preferences, use the get_user_state tool to retrieve their information from the session state.
        - When a user tells you about a new preference or wants to update an existing one, use the update_preference tool to save it to the session state.
        - Provide personalized responses based on the user's stored preferences.
        
        TOOL USAGE:
        - get_user_state: Retrieves all user preferences
        - update_preference: Updates or adds a new preference (takes preference_key and preference_value)
        
        Be conversational and helpful. If you don't know something that isn't in the state, say so politely.        
        """,
        tools=[get_user_state, update_preference]
    )
    return agent

def create_runner(app_name: str, session_service: InMemorySessionService, agent: Agent):
    runner = Runner(
        app_name=app_name,
        session_service=session_service,
        agent=agent
    )
    return runner

def invoke_message(runner: Runner, user_id: str, session_id: str, message_text: str):
    """Send a message to the agent and print the response.
    
    Args:
        runner: The Runner instance
        user_id: The user ID
        session_id: The session ID
        message_text: The text message to send
    """
    user_message = types.Content(
        role="user",
        parts=[types.Part(text=message_text)]
    )
    
    for event in runner.run(
        user_id=user_id,
        session_id=session_id,
        new_message=user_message
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                print(f"Agent: {event.content.parts[0].text}")

async def main():
    # Variables
    app_name = "personal_assistant"
    user_id = "user_12345"
    # Initialize session service
    session_service = InMemorySessionService()
    session = await create_session(session_service, app_name, user_id)
    session_id = session.id

    # Create agent
    agent = create_agent(app_name)

    # Initialize runner with agent and session service
    runner = create_runner(app_name, session_service, agent)

    # Define test interactions
    test_messages = [
        "What is my favorite color?",
        "What are all my preferences?",
        "My favorite movie is The Matrix",
        "What's my favorite movie?",
    ]
    
    print(f"Session created with ID: {session_id}")
    print(f"Initial state: preferences={session.state.get('preferences')}\n")
    print("=" * 80)
    
    # Execute multiple conversations to demonstrate state management
    for i, message_text in enumerate(test_messages, 1):
        print(f"\n[Interaction {i}]")
        print(f"User: {message_text}")
        
        invoke_message(runner, user_id, session_id, message_text)
        
        print("-" * 80)
    
    # Show final state by retrieving the latest session
    final_session = await session_service.get_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )
    print(f"\nFinal session state:")
    if final_session:
        print(f"  preferences: {final_session.state.get('preferences')}")
    else:
        print("  Session not found!")

if __name__ == "__main__":
    asyncio.run(main())