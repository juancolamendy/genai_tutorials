"""
Persistent Session Agent with Google ADK
A personal assistant that remembers user preferences across conversations.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

from google.adk.sessions import DatabaseSessionService
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants
APP_NAME = "personal_assistant"
USER_ID = "user_12345"
DB_URL = "sqlite:///./session_data.db"


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


def initial_state():
    """Initialize the session state with default preferences."""
    return {
        "preferences": {
            "favorite_color": "blue",
            "favorite_food": "pizza"
        }
    }


def create_agent(app_name: str) -> Agent:
    """Create and configure the agent with tools and instructions."""
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


def create_runner(app_name: str, session_service: DatabaseSessionService, agent: Agent) -> Runner:
    """Create a runner to orchestrate the agent and session service."""
    runner = Runner(
        app_name=app_name,
        session_service=session_service,
        agent=agent
    )
    return runner


def process_agent_event(event) -> Optional[str]:
    """Process agent events and return text to display.
    
    Args:
        event: Event from the agent runner
        
    Returns:
        str: Text to display, or None if no display needed
    """
    if event.is_final_response():
        if event.content and event.content.parts:
            return event.content.parts[0].text
    return None


async def ainvoke_message(runner: Runner, user_id: str, session_id: str, message_text: str):
    """Send a message to the agent and stream the response.
    
    Args:
        runner: The Runner instance
        user_id: The user ID
        session_id: The session ID
        message_text: The text message to send
    """
    # Create message content
    message = types.Content(
        role="user",
        parts=[types.Part(text=message_text)]
    )
    
    print("\nAssistant: ", end="", flush=True)
    
    # Stream agent response
    try:
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=message
        ):
            response = process_agent_event(event)
            if response:
                print(response, end="", flush=True)
    except Exception as e:
        print(f"\nError: {e}")
    
    print("\n")


async def main():
    """Main function to run the persistent session assistant."""
    print("\n" + "=" * 80)
    print("Initializing Persistent Session Agent...")
    print("=" * 80)
    
    # Initialize session service with database
    db_url = DB_URL
    session_service = DatabaseSessionService(db_url=db_url)
    
    # Check for existing sessions
    existing_sessions = await session_service.list_sessions(
        app_name=APP_NAME,
        user_id=USER_ID
    )
    
    # Use existing or create new session
    if existing_sessions and len(existing_sessions.sessions) > 0:
        session_id = existing_sessions.sessions[0].id
        print(f"✓ Continuing session: {session_id}")
        
        # Get and display current state
        current_session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            session_id=session_id
        )
        if current_session:
            print(f"✓ Current preferences: {current_session.state.get('preferences', {})}")
    else:
        new_session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=USER_ID,
            state=initial_state()
        )
        session_id = new_session.id
        print(f"✓ Created new session: {session_id}")
        print(f"✓ Initial preferences: {initial_state()['preferences']}")
    
    print("\n" + "=" * 80)
    print("Personal Assistant - Type 'exit' or 'quit' to end the conversation")
    print("Try: 'What are my preferences?' or 'I like chocolate'")
    print("=" * 80 + "\n")
    
    # Create agent and runner
    agent = create_agent(APP_NAME)
    runner = create_runner(APP_NAME, session_service, agent)
    
    # Main interaction loop
    while True:
        try:
            user_input = input("You: ")
            
            if user_input.lower() in ["exit", "quit"]:
                print("\n" + "=" * 80)
                print("Session saved. Have a great day!")
                
                # Display final state
                final_session = await session_service.get_session(
                    app_name=APP_NAME,
                    user_id=USER_ID,
                    session_id=session_id
                )
                if final_session:
                    print(f"✓ Final preferences: {final_session.state.get('preferences', {})}")
                print("=" * 80 + "\n")
                break
            
            # Skip empty inputs
            if not user_input.strip():
                continue
            
            # Send message and get response
            await ainvoke_message(runner, USER_ID, session_id, user_input)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Saving session...")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            print("Continuing...\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"\nFatal error: {e}")
