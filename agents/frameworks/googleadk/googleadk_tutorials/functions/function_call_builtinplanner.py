"""
Simple ADK Context Debugger with BuiltInPlanner

A minimal example showing how to debug and inspect the LLM context
with BuiltInPlanner (Gemini Thinking mode) enabled.
"""

import asyncio
from datetime import datetime

from google.genai import types
from google.genai.types import ThinkingConfig
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.adk.planners import BuiltInPlanner
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService


def add(a: float, b: float) -> float:
    """Add two numbers together.

    Args:
        a: The first number
        b: The second number

    Returns:
        The sum of a and b
    """
    return a + b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers together.

    Args:
        a: The first number
        b: The second number

    Returns:
        The product of a and b
    """
    return a * b


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

    # ===== DEBUG THINKING CONFIG =====
    if hasattr(llm_request.config, 'thinking_config') and llm_request.config.thinking_config:
        thinking_cfg = llm_request.config.thinking_config
        print(f"\n🧠 Thinking Mode Enabled:")
        if hasattr(thinking_cfg, 'include_thoughts'):
            print(f"  - Include Thoughts: {thinking_cfg.include_thoughts}")
        if hasattr(thinking_cfg, 'thinking_level'):
            print(f"  - Thinking Level: {thinking_cfg.thinking_level}")
        if hasattr(thinking_cfg, 'thinking_budget'):
            print(f"  - Thinking Budget: {thinking_cfg.thinking_budget}")

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
                elif hasattr(part, 'thought') and part.thought:
                    print(f"[🧠 Thinking...]")
                elif part.function_call:
                    print(f"[Tool: {part.function_call.name}]")
                elif part.function_response:
                    print(f"[Tool Result: {part.function_response.name}]")

    print("\n" + "="*70 + "\n")

    return None


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
            # Display thinking content if present
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, 'thought') and part.thought:
                        print(f"\n🧠 THINKING: {part.thought}\n")
                    elif part.text:
                        print(f"\n🤖 AGENT: {part.text}\n")
            else:
                print("\n🤖 AGENT: No response\n")


async def main():
    """
    Simple example demonstrating context debugging with BuiltInPlanner.
    """

    print("\n" + "="*70)
    print("🚀 ADK CONTEXT DEBUGGER WITH BUILTINPLANNER")
    print("="*70)
    print("This example shows BuiltInPlanner (Gemini Thinking Mode)")
    print("with tools and debug visibility.")
    print("="*70 + "\n")

    # Configure thinking mode for Gemini 3
    thinking_config = ThinkingConfig(
        include_thoughts=True,  # Include thinking summary in response
        thinking_level="high",  # Options: minimal, low, medium, high
    )

    # Create BuiltInPlanner with thinking config
    planner = BuiltInPlanner(thinking_config=thinking_config)

    # Create agent with BuiltInPlanner, debug callback, and tools
    agent = Agent(
        name="thinking_debug_agent",
        model="gemini-3-flash-preview",  # Use Gemini 3 for thinking mode
        instruction="You are a helpful assistant that carefully reasons about questions. Use the information provided above to answer questions. You have tools to perform mathematical operations.",
        before_model_callback=debug_and_inject_context,
        tools=[add, multiply],
        planner=planner,  # Add BuiltInPlanner
    )

    # Create runner with in-memory sessions
    runner = Runner(
        app_name="simple_debug_app",
        agent=agent,
        session_service=InMemorySessionService()
    )

    user_id = "user123"
    session_id = "session456"

    # Create session with some initial state
    await runner.session_service.create_session(
        app_name="simple_debug_app",
        user_id=user_id,
        session_id=session_id,
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
