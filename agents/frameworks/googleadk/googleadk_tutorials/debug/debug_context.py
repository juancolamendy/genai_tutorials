import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

from rich.pretty import pprint
from rich.console import Console

from google.genai import types
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.events import Event, EventActions
from google.adk.tools import ToolContext

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Rich console for pretty printing
console = Console()

# Constants
# Note: Using InMemorySessionService for this demo
# For persistent storage, use a custom implementation or external database

# ============================================================================
# TOOLS - Functions for managing preferences in session state
# ============================================================================

def update_preference(preference_key: str, preference_value: str, tool_context: ToolContext) -> dict:
    """Stores a user preference.

    Args:
        preference_key: The preference key/category (e.g., "favorite_topic", "favorite_color").
        preference_value: The preference value.
        tool_context: Automatically provided by ADK.

    Returns:
        dict: Status of the operation.
    """
    preferences = tool_context.state.get("preferences", {})
    preferences[preference_key] = preference_value
    tool_context.state["preferences"] = preferences
    return {"status": "success", "message": f"Preference set: {preference_key} = {preference_value}"}


def list_all_preferences(tool_context: ToolContext) -> dict:
    """Retrieves all user preferences.
    
    Args:
        tool_context: Automatically provided by ADK.
        
    Returns:
        dict: The user's stored preferences.
    """
    preferences = tool_context.state.get("preferences", {})
    return {"status": "success", "preferences": preferences}


# ============================================================================
# DEBUG CALLBACK - Inspects everything sent to the LLM
# ============================================================================

def debug_llm_context(
    callback_context: CallbackContext,
    llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    Comprehensive callback to inspect and debug the LLM context.
    Shows: system instruction, datetime, messages, session state, and tools.
    """

    console.print("\n" + "="*80, style="bold cyan")
    console.print("🔍 DEBUG: LLM REQUEST CONTEXT", style="bold cyan")
    console.print("="*80, style="bold cyan")

    # Agent metadata
    console.print(f"\n📌 [bold]Agent:[/bold] {callback_context.agent_name}")
    console.print(f"📌 [bold]Invocation ID:[/bold] {callback_context.invocation_id}")
    console.print(f"📌 [bold]User ID:[/bold] {callback_context.user_id}")
    console.print(f"📌 [bold]Session ID:[/bold] {callback_context.session.id if callback_context.session else 'N/A'}")

    # System instruction (includes injected datetime and state)
    if llm_request.config and llm_request.config.system_instruction:
        console.print(f"\n📋 [bold yellow]SYSTEM INSTRUCTION:[/bold yellow]")
        console.print("─" * 80, style="yellow")
        console.print(llm_request.config.system_instruction, style="yellow")
        console.print("─" * 80, style="yellow")

    # Session state
    state_dict = callback_context.state.to_dict() if callback_context.state else {}
    if state_dict:
        console.print(f"\n💾 [bold magenta]SESSION STATE ({len(state_dict)} items):[/bold magenta]")
        console.print("─" * 80, style="magenta")
        for key, value in state_dict.items():
            console.print(f"  {key}: {value}", style="magenta")
        console.print("─" * 80, style="magenta")
    else:
        console.print("\n💾 [bold magenta]SESSION STATE:[/bold magenta] Empty")

    # Conversation history (contents)
    console.print(f"\n💬 [bold green]CONVERSATION HISTORY ({len(llm_request.contents)} messages):[/bold green]")
    console.print("─" * 80, style="green")

    for i, content in enumerate(llm_request.contents):
        role_emoji = "👤" if content.role == "user" else "🤖"
        console.print(f"\n  [{i}] {role_emoji} [bold]{content.role.upper()}[/bold]:", style="green")

        if content.parts:
            for j, part in enumerate(content.parts):
                if part.text:
                    # Truncate long text for readability
                    text = part.text
                    if len(text) > 200:
                        text = text[:200] + "..."
                    console.print(f"      💭 Text: {text}", style="green")

                elif part.function_call:
                    console.print(f"      🔧 Function Call: [bold]{part.function_call.name}[/bold]", style="cyan")
                    console.print(f"         Args: {dict(part.function_call.args)}", style="cyan")

                elif part.function_response:
                    console.print(f"      ✅ Function Response: [bold]{part.function_response.name}[/bold]", style="blue")
                    response_str = str(part.function_response.response)
                    if len(response_str) > 150:
                        response_str = response_str[:150] + "..."
                    console.print(f"         Response: {response_str}", style="blue")

    console.print("─" * 80, style="green")

    # Tools available
    if llm_request.config and llm_request.config.tools:
        console.print(f"\n🔨 [bold blue]AVAILABLE TOOLS:[/bold blue]")
        console.print("─" * 80, style="blue")
        for tool in llm_request.config.tools:
            if tool.function_declarations:
                for func in tool.function_declarations:
                    console.print(f"  • {func.name}", style="blue")
                    if func.description:
                        desc = func.description.split('\n')[0]  # First line only
                        if len(desc) > 70:
                            desc = desc[:70] + "..."
                        console.print(f"    └─ {desc}", style="dim blue")
        console.print("─" * 80, style="blue")

    console.print("="*80 + "\n", style="bold cyan")

    # Return None to allow the LLM call to proceed
    return None


# ============================================================================
# DATETIME + STATE INJECTION
# ============================================================================

def inject_datetime_and_state(
    callback_context: CallbackContext,
    llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    Injects current datetime and session state into the system instruction.
    This mimics Agno's add_datetime_to_context and add_session_state_to_context.
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

    return None


# ============================================================================
# COMBINED CALLBACK - Both debug and injection
# ============================================================================

def combined_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest
) -> Optional[LlmResponse]:
    """
    Combines both datetime/state injection and debugging.
    First injects, then debugs to show final state.
    """
    # First inject datetime and state
    inject_datetime_and_state(callback_context, llm_request)

    # Then debug to show everything
    debug_llm_context(callback_context, llm_request)

    return None


# ============================================================================
# AGENT CREATION
# ============================================================================

def create_agent():
    """
    Creates an ADK agent with similar functionality to the Agno agent.
    """

    agent = Agent(
        name="context_debug_agent",
        model="gemini-3-flash-preview",

        # System instruction (role/instructions)
        instruction="""You are a helpful AI assistant with access to user preferences.

IMPORTANT GUIDELINES:
- When a user asks about their preferences, use the list_all_preferences tool.
- When a user tells you about a new preference or wants to update one, use the update_preference tool.
- Provide personalized responses based on stored preferences shown in the SESSION STATE
- Be conversational and friendly.
- If asked about something not in preferences, politely say you don't have that information stored.

TOOL USAGE:
- update_preference: Updates a single preference (takes preference_key and preference_value)
- list_all_preferences: Lists all preferences in a formatted way

RESPONSE GUIDELINES:
- Always evaluate your confidence in your responses
- Provide clear reasoning for your answers
- Use the preference information to personalize your responses
""",

        # Tools
        tools=[update_preference, list_all_preferences],

        # Callbacks for debugging and context injection
        before_model_callback=combined_callback,
    )

    return agent


# ============================================================================
# CONVERSATION EXECUTION
# ============================================================================

async def execute_turn(runner, turn: str, user_id: str, session_id: str):
    """
    Executes a single conversation turn and displays the response.
    """

    console.print(f"\n[bold blue]You:[/bold blue] {turn}")

    # Create user message
    msg = types.Content(
        role='user',
        parts=[types.Part(text=turn)]
    )

    # Track response
    final_response = None
    run_id = None

    # Execute and stream events
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=msg
    ):
        # Get final response
        if event.is_final_response():
            final_response = event.content.parts[0].text if event.content and event.content.parts else "No response"
            run_id = event.invocation_id

    # Display response
    console.print(f"[bold green]Agent:[/bold green] {final_response}\n")

    # Display metadata
    console.print("─" * 80, style="dim")
    console.print(f"[dim]Run ID: {run_id} | Session: {session_id}[/dim]")
    console.print("─" * 80 + "\n", style="dim")


# ============================================================================
# MAIN INTERACTIVE LOOP
# ============================================================================

async def main():
    """
    Main function that creates the agent, runner, and interactive loop.
    """

    console.print("\n" + "="*80, style="bold magenta")
    console.print("🚀 ADK CONTEXT DEBUGGER - Interactive Chat", style="bold magenta")
    console.print("="*80, style="bold magenta")
    console.print("\nFeatures:", style="bold")
    console.print("  • Full context debugging (datetime, state, messages, tools)")
    console.print("  • Persistent session state with preferences")
    console.print("  • Rich formatted output")
    console.print("\nCommands:", style="bold")
    console.print("  • Type your message to chat")
    console.print("  • Type '/exit' to quit")
    console.print("="*80 + "\n", style="bold magenta")

    # Create agent
    agent = create_agent()

    # Create session service (in-memory for this demo)
    # Note: Sessions will not persist after the program exits
    session_service = InMemorySessionService()

    # Create runner
    runner = Runner(
        app_name="context_debug_app",
        agent=agent,
        session_service=session_service
    )

    # Session IDs
    user_id = "user_01"
    session_id = "session_01"

    # Initialize session with default preferences (like Agno's session_state)
    await session_service.create_session(
        app_name="context_debug_app",
        user_id=user_id,
        session_id=session_id,
        state={
            "preferences": {
                "favorite_color": "blue",
                "favorite_food": "pizza",
                "preferred_language": "English"
            }
        }
    )

    console.print("[green]✓ Session initialized with default preferences[/green]\n")

    # Interactive loop
    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ").strip()

            if user_input == "exit":
                console.print("\n[yellow]👋 Goodbye![/yellow]\n")
                break
            elif not user_input:
                continue

            # Execute conversation turn
            await execute_turn(runner, user_input, user_id, session_id)

        except KeyboardInterrupt:
            console.print("\n\n[yellow]👋 Goodbye![/yellow]\n")
            break
        except Exception as e:
            console.print(f"\n[red]❌ Error: {e}[/red]\n", style="bold")


if __name__ == "__main__":
    asyncio.run(main())
