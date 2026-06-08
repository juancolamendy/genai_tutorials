"""
ADK Context Mode Comparison

Demonstrates the difference between:
1. Default mode (with history)
2. Stateless mode (no history)
3. Custom history management
"""

import asyncio
from datetime import datetime
from google.genai import types
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService


def create_debug_callback(label: str):
    """Factory to create labeled debug callbacks"""

    def debug_callback(callback_context: CallbackContext, llm_request: LlmRequest):
        print(f"\n{'='*70}")
        print(f"🔍 {label}")
        print(f"{'='*70}")

        # Show datetime injection
        now = datetime.now()
        datetime_text = f"\nCurrent: {now.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Inject datetime
        original = llm_request.config.system_instruction or ""
        llm_request.config.system_instruction = datetime_text + original

        # Show what's being sent
        print(f"📊 Messages in context: {len(llm_request.contents)}")
        print(f"📋 System instruction length: {len(llm_request.config.system_instruction)} chars")

        print(f"\n💬 Message History:")
        for i, content in enumerate(llm_request.contents):
            role_icon = "👤" if content.role == "user" else "🤖"
            text = ""
            if content.parts and content.parts[0].text:
                text = content.parts[0].text[:60]
            print(f"  [{i}] {role_icon} {content.role}: {text}...")

        print(f"{'='*70}\n")

        return None

    return debug_callback


async def test_mode(title: str, agent: Agent, runner: Runner, user_id: str, session_id: str):
    """Test a specific agent/mode with a conversation"""

    print(f"\n{'█'*70}")
    print(f"  {title}")
    print(f"{'█'*70}")

    queries = [
        "Hi, my name is Alice.",
        "What's my name?",
        "Remember, I like pizza.",
        "What do I like?"
    ]

    for query in queries:
        print(f"\n👤 USER: {query}")

        msg = types.Content(role='user', parts=[types.Part(text=query)])

        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=msg
        ):
            if event.is_final_response():
                response = event.content.parts[0].text if event.content and event.content.parts else "No response"
                print(f"🤖 AGENT: {response}")

        await asyncio.sleep(0.3)


async def main():
    """Compare different context modes"""

    print("\n" + "="*70)
    print("🔬 ADK CONTEXT MODE COMPARISON")
    print("="*70)
    print("This demonstrates how different settings affect conversation history.")
    print("="*70 + "\n")

    session_service = InMemorySessionService()

    # ===== TEST 1: DEFAULT MODE (with history) =====
    agent_default = Agent(
        name="with_history",
        model="gemini-3-flash-preview",
        instruction="You are a helpful assistant.",
        include_contents='default',  # Default: includes history
        before_model_callback=create_debug_callback("TEST 1: WITH HISTORY (default)")
    )

    runner_default = Runner(
        app_name="test_default",
        agent=agent_default,
        session_service=session_service
    )

    # Create session
    await session_service.create_session(
        app_name="test_default",
        user_id="user1",
        session_id="session1"
    )

    await test_mode(
        "TEST 1: WITH CONVERSATION HISTORY (include_contents='default')",
        agent_default,
        runner_default,
        "user1",
        "session1"
    )

    print("\n" + "⏸️ "*35)
    await asyncio.sleep(1)

    # ===== TEST 2: STATELESS MODE (no history) =====
    agent_stateless = Agent(
        name="stateless",
        model="gemini-3-flash-preview",
        instruction="You are a helpful assistant.",
        include_contents='none',  # Stateless: no history
        before_model_callback=create_debug_callback("TEST 2: STATELESS (no history)")
    )

    runner_stateless = Runner(
        app_name="test_stateless",
        agent=agent_stateless,
        session_service=session_service
    )

    # Create session
    await session_service.create_session(
        app_name="test_stateless",
        user_id="user2",
        session_id="session2"
    )

    await test_mode(
        "TEST 2: STATELESS MODE (include_contents='none')",
        agent_stateless,
        runner_stateless,
        "user2",
        "session2"
    )

    # ===== COMPARISON SUMMARY =====
    print("\n" + "="*70)
    print("📊 SUMMARY OF DIFFERENCES")
    print("="*70)

    print("\n🟢 WITH HISTORY (include_contents='default'):")
    print("  ✓ Agent remembers previous messages")
    print("  ✓ Can reference earlier conversation")
    print("  ✓ Builds context over multiple turns")
    print("  ✓ Context grows with each message")
    print("  ⚠️ Higher token usage")

    print("\n🔴 STATELESS (include_contents='none'):")
    print("  ✗ Agent forgets after each turn")
    print("  ✗ Cannot remember names, preferences, etc.")
    print("  ✓ Each message is independent")
    print("  ✓ Consistent, predictable behavior")
    print("  ✓ Lower token usage")

    print("\n💡 USE CASES:")
    print("\n  With History:")
    print("    • Chatbots and conversational AI")
    print("    • Personal assistants")
    print("    • Customer support")
    print("    • Any multi-turn dialogue")

    print("\n  Stateless:")
    print("    • Text classification")
    print("    • Translation services")
    print("    • Single-shot Q&A")
    print("    • API endpoints where each call is independent")

    print("\n" + "="*70 + "\n")


async def demo_state_injection():
    """
    Bonus: Show how to inject state into context for stateless agents
    """

    print("\n" + "="*70)
    print("🎁 BONUS: Stateless + State Injection")
    print("="*70)
    print("Even without history, you can inject state into each request!")
    print("="*70 + "\n")

    def inject_state_callback(callback_context: CallbackContext, llm_request: LlmRequest):
        """Inject state even in stateless mode"""

        state_dict = callback_context.state.to_dict() if callback_context.state else {}

        # Build state section
        state_section = "\n=== USER CONTEXT ===\n"
        if state_dict:
            for key, value in state_dict.items():
                if not key.startswith("_"):
                    state_section += f"{key}: {value}\n"
        state_section += "===================\n\n"

        # Inject
        original = llm_request.config.system_instruction or ""
        llm_request.config.system_instruction = state_section + original

        print(f"🔍 Injected state: {state_dict}")
        print(f"📊 Messages in context: {len(llm_request.contents)} (current message only)")

        return None

    agent_hybrid = Agent(
        name="stateless_with_state",
        model="gemini-3-flash-preview",
        instruction="You are a helpful assistant. Use the user context above.",
        include_contents='none',  # Stateless
        before_model_callback=inject_state_callback
    )

    session_service = InMemorySessionService()

    runner_hybrid = Runner(
        app_name="test_hybrid",
        agent=agent_hybrid,
        session_service=session_service
    )

    # Create session with state
    await runner_hybrid.session_service.create_session(
        app_name="test_hybrid",
        user_id="user3",
        session_id="session3",
        state={
            "user_name": "Bob",
            "favorite_color": "green"
        }
    )

    queries = [
        "What's my name?",
        "What's my favorite color?"
    ]

    for query in queries:
        print(f"\n👤 USER: {query}")

        msg = types.Content(role='user', parts=[types.Part(text=query)])

        async for event in runner_hybrid.run_async(
            user_id="user3",
            session_id="session3",
            new_message=msg
        ):
            if event.is_final_response():
                response = event.content.parts[0].text if event.content and event.content.parts else "No response"
                print(f"🤖 AGENT: {response}")

        await asyncio.sleep(0.3)

    print("\n💡 Notice: Even without conversation history, the agent knows")
    print("   the user's name and favorite color from the injected state!")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
    asyncio.run(demo_state_injection())
