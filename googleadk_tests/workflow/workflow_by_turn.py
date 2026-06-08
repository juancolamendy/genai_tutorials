# agents.py
import asyncio
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import DatabaseSessionService
from google.genai import types as genai_types

agent1 = LlmAgent(
    name="agent1",
    model="gemini-3-flash-preview",
    instruction="Step 1: Greet the user and store a short note in state under 'note_1'.",
    output_key="note_1",
)

agent2 = LlmAgent(
    name="agent2",
    model="gemini-3-flash-preview",
    instruction="""
Step 2: Read {note_1} and write a follow-up note into state under 'note_2'.
Only use state, do not ask the user again.
""",
    output_key="note_2",
)

agent3 = LlmAgent(
    name="agent3",
    model="gemini-3-flash-preview",
    instruction="""
Step 3: Read {note_1} and {note_2} and compose the final answer for the user.
""",
    output_key="final_answer",
)

# Don't use SequentialAgent - we'll run one agent at a time
all_agents = [agent1, agent2, agent3]

db_name = 'workflow_turn'
DB_URL = f"sqlite+aiosqlite:///./{db_name}.db"
APP_NAME = "sequential-demo"
USER_ID = "user-123"

async def get_or_create_session(session_service, user_id: str):
    """Get existing session or create a new one."""
    # Check for existing sessions
    existing_sessions = await session_service.list_sessions(
        app_name=APP_NAME,
        user_id=user_id
    )

    # Use existing or create new session
    if existing_sessions and len(existing_sessions.sessions) > 0:
        session_id = existing_sessions.sessions[0].id
        print(f"Continuing existing session: {session_id}")

        # Get and display current state
        current_session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id
        )
        if current_session and current_session.state:
            print(f"Current state keys: {list(current_session.state.keys())}")
    else:
        new_session = await session_service.create_session(
            app_name=APP_NAME,
            user_id=user_id,
            state={}
        )
        session_id = new_session.id
        print(f"Created new session: {session_id}")

    return session_id

async def determine_next_agent(session_service, session_id: str, user_id: str):
    """Determine which agent should run next based on session state."""
    current_session = await session_service.get_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id
    )

    state = current_session.state if current_session else {}

    # Determine next agent based on what's in state
    if 'final_answer' in state:
        return None, "complete"
    elif 'note_2' in state:
        return agent3, "agent3"
    elif 'note_1' in state:
        return agent2, "agent2"
    else:
        return agent1, "agent1"

async def run_turn(session_service, message: str, user_id: str, session_id: str):
    """Run a single turn of the sequential agent workflow."""

    # Determine which agent should run
    agent_to_run, agent_status = await determine_next_agent(session_service, session_id, user_id)

    if agent_status == "complete":
        print(f"\n{'='*60}")
        print("WORKFLOW ALREADY COMPLETE! All agents have finished.")
        print(f"{'='*60}\n")
        return True

    print(f"\n{'='*60}")
    print(f"Running {agent_status} for session: {session_id}")
    print(f"{'='*60}\n")

    # Create a runner for just this agent
    runner = Runner(
        agent=agent_to_run,
        session_service=session_service,
        app_name=APP_NAME,
    )

    user_message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=message)],
    )

    response_text = None

    # Run the single agent
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_message,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                response_text = event.content.parts[0].text
                print(f"[{agent_status.upper()} RESPONSE] {response_text}")

    # Check if we've reached the final agent
    if agent_status == "agent3":
        print(f"\n{'='*60}")
        print("WORKFLOW COMPLETE! All agents have finished.")
        print(f"{'='*60}\n")
        return True
    else:
        print(f"\n{'='*60}")
        print(f"Turn finished. Run the script again to continue to {agent_status}.")
        print(f"{'='*60}\n")
        return False

async def main():
    """Main entry point - gets user input and runs one turn."""
    print("\n" + "="*60)
    print("Sequential Agent Workflow - Turn-by-Turn Execution")
    print("="*60)
    print(f"User ID: {USER_ID}")
    print(f"Database: {db_name}.db")
    print("="*60 + "\n")

    # Create session service
    session_service = DatabaseSessionService(db_url=DB_URL)

    try:
        # Get or create session
        session_id = await get_or_create_session(session_service, USER_ID)
        print()

        # Get user input
        user_input = input("Enter your message (or press Enter for default): ").strip()

        # Use default message if empty
        if not user_input:
            user_input = "Hi, I'd like help planning my weekend."
            print(f"Using default message: {user_input}\n")

        # Run one turn
        is_complete = await run_turn(session_service, user_input, USER_ID, session_id)

        if is_complete:
            print("Session complete!")
        else:
            print("Run the script again to execute the next agent in the sequence.")
    finally:
        # Close database connections
        if hasattr(session_service, 'close'):
            await session_service.close()
        elif hasattr(session_service, '_engine'):
            await session_service._engine.dispose()

if __name__ == "__main__":
    asyncio.run(main())
