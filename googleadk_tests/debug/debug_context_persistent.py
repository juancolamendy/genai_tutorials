"""
Simple ADK Context Debugger

A minimal example showing how to debug and inspect the LLM context
with datetime injection and session state visibility.
"""

import asyncio
from datetime import datetime

from google.genai import types
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.adk.runners import Runner
from google.adk.sessions.database_session_service import DatabaseSessionService
from google.adk.errors.already_exists_error import AlreadyExistsError


def debug_and_inject_context(
    callback_context: CallbackContext,
    llm_request: LlmRequest
):
    """
    Single callback that:
    1. Injects current datetime into system instruction
    2. Injects session state into system instruction
    3. Prints debug information about the request
    """

    # ===== INJECT DATETIME =====
    now = datetime.now()
    datetime_text = f"""
Current Date and Time: {now.strftime("%A, %B %d, %Y at %I:%M:%S %p")}
"""

    # ===== INJECT SESSION STATE =====
    state_text = "\nSession State:\n"
    state_dict = callback_context.state.to_dict() if callback_context.state else {}
    if state_dict:
        for key, value in state_dict.items():
            if not key.startswith("_"):  # Skip internal keys
                state_text += f"  - {key}: {value}\n"
    else:
        state_text += "  (empty)\n"

    state_text += "\n" + "="*70 + "\n"

    # ===== MODIFY SYSTEM INSTRUCTION =====
    original_instruction = llm_request.config.system_instruction or ""
    llm_request.config.system_instruction = (
        datetime_text + state_text + original_instruction
    )

    # ===== DEBUG OUTPUT =====
    print("\n" + "="*70)
    print("🔍 DEBUG: LLM REQUEST")
    print("="*70)

    print(f"\n📌 Agent: {callback_context.agent_name}")
    print(f"📌 User: {callback_context.user_id}")
    print(f"📌 Session: {callback_context.session.id if callback_context.session else 'N/A'}")

    print(f"\n📋 System Instruction:")
    print("-" * 70)
    # Show first 500 chars to avoid clutter
    instruction_preview = llm_request.config.system_instruction[:500]
    if len(llm_request.config.system_instruction) > 500:
        instruction_preview += "..."
    print(instruction_preview)
    print("-" * 70)

    print(f"\n💾 Session State: {callback_context.state.to_dict()}")

    print(f"\n💬 Messages ({len(llm_request.contents)}):")
    for i, content in enumerate(llm_request.contents):
        role_emoji = "👤" if content.role == "user" else "🤖"
        print(f"  [{i}] {role_emoji} {content.role}:", end=" ")

        if content.parts:
            for part in content.parts:
                if part.text:
                    text = part.text[:80] + "..." if len(part.text) > 80 else part.text
                    print(f"{text}")
                elif part.function_call:
                    print(f"[Tool: {part.function_call.name}]")
                elif part.function_response:
                    print(f"[Tool Result: {part.function_response.name}]")

    print("\n" + "="*70 + "\n")

    return None


async def create_session(
    session_service,
    app_name: str,
    user_id: str,
    session_id: str | None,
    state: dict
) -> str:
    """
    Create a session with error handling for existing sessions.

    Args:
        session_service: The session service to use
        app_name: The application name
        user_id: The user identifier
        session_id: The session identifier (or None for auto-generation)
        state: The initial state dictionary

    Returns:
        str: The session ID (either newly created or existing)

    Raises:
        Exception: If creation fails for reasons other than already exists
    """
    try:
        await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id,
            state=state
        )
        return session_id
    except AlreadyExistsError:
        # Session already exists, return the existing session_id
        print(f"ℹ️  Session '{session_id}' already exists. Using existing session.")
        return session_id
    except Exception as e:
        # Other errors should be raised
        raise e


async def run_turn(runner: Runner, user_id: str, session_id: str, query: str):
    """
    Execute a single conversation turn with the agent.
    
    Args:
        runner: The Runner instance to execute the query
        user_id: The user identifier
        session_id: The session identifier
        query: The user's query text
    """
    print(f"\n{'─'*70}")
    print(f"👤 USER: {query}")
    print('─'*70)

    msg = types.Content(role='user', parts=[types.Part(text=query)])

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=msg
    ):
        if event.is_final_response():
            response_text = event.content.parts[0].text if event.content and event.content.parts else "No response"
            print(f"\n🤖 AGENT: {response_text}\n")


async def main():
    """
    Simple example demonstrating context debugging.
    """

    print("\n" + "="*70)
    print("🚀 SIMPLE ADK CONTEXT DEBUGGER")
    print("="*70)
    print("This example shows how to debug the LLM context.")
    print("="*70 + "\n")

    # Create agent with debug callback
    agent = Agent(
        name="debug_agent",
        model="gemini-3-flash-preview",
        instruction="You are a helpful assistant. Use the information provided above to answer questions.",
        #include_contents='none',  # Stateless: no history
        before_model_callback=debug_and_inject_context
    )

    # Create runner with persistent SQLite sessions
    runner = Runner(
        app_name="simple_debug_app",
        agent=agent,
        session_service=DatabaseSessionService(
            db_url='sqlite+aiosqlite:///./simple_context_stateless_data.db'
        )
    )

    user_id = "user123"
    session_id = "session456"

    # Create session with some initial state
    session_id = await create_session(
        session_service=runner.session_service,
        app_name="simple_debug_app",
        user_id=user_id,
        session_id=session_id,
        state={
            "location": "Miami",
            "membership_level": "premium"
        }
    )

    # Interactive loop
    print("Type 'exit' to quit\n")
    
    while True:
        query = input("👤 YOU: ").strip()
        
        if query.lower() == 'exit':
            break
        
        if query:  # Only process non-empty input
            await run_turn(runner, user_id, session_id, query)


if __name__ == "__main__":
    asyncio.run(main())
