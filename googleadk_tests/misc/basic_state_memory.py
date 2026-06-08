import asyncio
from google.adk.sessions import InMemorySessionService

# Initialize session service
session_service = InMemorySessionService()

# Application configuration
APP_NAME = "personal_assistant"
USER_ID = "user_12345"
SESSION_ID = "session_67890"

# Basic state operations
async def demonstrate_basic_state():
    # Create or retrieve session
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID
    )
    
    # Reading with default value
    user_name = session.state.get("user:name", "Guest")
    
    # Writing to different scopes
    session.state["user:name"] = "Alice"
    session.state["last_interaction"] = "greeting"
    session.state["app:total_users"] = session.state.get("app:total_users", 0) + 1
    
    user_name = session.state.get("user:name", "Guest")
    last_interaction = session.state.get("last_interaction", "none")
    total_users = session.state.get("app:total_users", 0)
    print(f"Hello, {user_name}!")
    print(f"Total app users: {total_users}")
    print(f"Last interaction: {last_interaction}")

# Run the async function
asyncio.run(demonstrate_basic_state())