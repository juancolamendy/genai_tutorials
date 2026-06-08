"""
Simple ADK Context Debugger with PlanReActPlanner

A minimal example showing how to debug and inspect the LLM context
with PlanReActPlanner (explicit ReAct structure: Plan/Action/Reasoning/Final).
"""

import asyncio
import re
from datetime import datetime

from google.genai import types
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.adk.planners import PlanReActPlanner
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

    print(f"\n💾 Session State: {callback_context.state.to_dict()}")

    print(f"\n💬 Messages ({len(llm_request.contents)}):")
    for i, content in enumerate(llm_request.contents):
        role_emoji = "👤" if content.role == "user" else "🤖"
        print(f"  [{i}] {role_emoji} {content.role}:", end=" ")

        if content.parts:
            for part in content.parts:
                if part.text:
                    text = part.text[:80] + "..." if len(part.text) > 80 else part.text
                    # Detect ReAct structure markers
                    if "/*PLANNING*/" in part.text or "/*ACTION*/" in part.text or "/*REASONING*/" in part.text or "/*FINAL_ANSWER*/" in part.text:
                        print(f"[📋 ReAct Structure] {text}")
                    else:
                        print(f"{text}")
                elif part.function_call:
                    print(f"[Tool: {part.function_call.name}]")
                elif part.function_response:
                    print(f"[Tool Result: {part.function_response.name}]")

    print("\n" + "="*70 + "\n")

    return None


def parse_react_response(text: str) -> dict:
    """
    Parse ReAct structure from response text.

    Args:
        text: The response text containing ReAct markers

    Returns:
        Dictionary with parsed sections
    """
    sections = {}

    # Extract PLANNING section
    planning_match = re.search(r'/\*PLANNING\*/(.*?)(?=/\*|$)', text, re.DOTALL)
    if planning_match:
        sections['planning'] = planning_match.group(1).strip()

    # Extract ACTION section
    action_match = re.search(r'/\*ACTION\*/(.*?)(?=/\*|$)', text, re.DOTALL)
    if action_match:
        sections['action'] = action_match.group(1).strip()

    # Extract REASONING section
    reasoning_match = re.search(r'/\*REASONING\*/(.*?)(?=/\*|$)', text, re.DOTALL)
    if reasoning_match:
        sections['reasoning'] = reasoning_match.group(1).strip()

    # Extract FINAL_ANSWER section
    final_match = re.search(r'/\*FINAL_ANSWER\*/(.*?)(?=/\*|$)', text, re.DOTALL)
    if final_match:
        sections['final_answer'] = final_match.group(1).strip()

    return sections


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
            if event.content and event.content.parts:
                response_text = event.content.parts[0].text if event.content.parts[0].text else "No response"

                # Check if response contains ReAct structure
                if "/*PLANNING*/" in response_text or "/*FINAL_ANSWER*/" in response_text:
                    sections = parse_react_response(response_text)

                    if 'planning' in sections:
                        print(f"\n📋 PLANNING:\n{sections['planning']}\n")

                    if 'action' in sections:
                        print(f"⚡ ACTION:\n{sections['action']}\n")

                    if 'reasoning' in sections:
                        print(f"💭 REASONING:\n{sections['reasoning']}\n")

                    if 'final_answer' in sections:
                        print(f"✅ FINAL ANSWER:\n{sections['final_answer']}\n")
                else:
                    # No ReAct structure, just print normal response
                    print(f"\n🤖 AGENT: {response_text}\n")
            else:
                print("\n🤖 AGENT: No response\n")


async def main():
    """
    Simple example demonstrating context debugging with PlanReActPlanner.
    """

    print("\n" + "="*70)
    print("🚀 ADK CONTEXT DEBUGGER WITH PLANREACTPLANNER")
    print("="*70)
    print("This example shows PlanReActPlanner (explicit ReAct structure)")
    print("with tools and debug visibility.")
    print("="*70 + "\n")

    # Create PlanReActPlanner
    planner = PlanReActPlanner()

    # Create agent with PlanReActPlanner, debug callback, and tools
    agent = Agent(
        name="react_debug_agent",
        model="gemini-3-flash-preview",
        instruction="""You are a planning agent that solves problems systematically.

For each question, you should:
1. First create a plan (/*PLANNING*/ section) - outline your approach
2. Then execute actions (/*ACTION*/ section) - call tools as needed
3. Then explain your reasoning (/*REASONING*/ section) - analyze the results
4. Finally give a concise answer (/*FINAL_ANSWER*/ section) - respond to the user

Use the information provided above to answer questions. You have tools to perform mathematical operations.""",
        before_model_callback=debug_and_inject_context,
        tools=[add, multiply],
        planner=planner,  # Add PlanReActPlanner
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
