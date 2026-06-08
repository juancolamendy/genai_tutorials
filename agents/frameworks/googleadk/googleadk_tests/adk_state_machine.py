"""
ADK — State Machine Pattern: Ticket Triage
─────────────────────────────────────────────────────────────────────────────
Shape:  INTAKE ↔ CLARIFY ↔ RESOLVE ↔ CONFIRM → CLOSED
        Interactive, stateful, supports back-transitions and re-routing.
        CustomAgent owns the state machine via _run_async_impl.
        LLM calls are narrow extraction/generation tasks only.

Install:
    uv add google-adk python-dotenv

Usage:
    uv run adk_state_machine.py
"""

import asyncio
import json
from enum import Enum
from typing import AsyncGenerator

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.genai import types

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL = "gemini-2.5-flash"
APP   = "ticket-triage-sm"

# Max times we'll ask the user for more info before proceeding with what we
# have. Guards against an over-strict completeness model trapping the machine.
MAX_INFO_ROUNDS = 2


# ── States ────────────────────────────────────────────────────────────────────

class TicketState(str, Enum):
    INTAKE   = "intake"     # Gather initial ticket description
    CLARIFY  = "clarify"    # Ask follow-up questions if info is incomplete
    RESOLVE  = "resolve"    # Generate a resolution
    CONFIRM  = "confirm"    # Ask user to confirm or revise
    CLOSED   = "closed"     # Done


# ── Transition tables (code owns these — no LLM involved) ────────────────────

FORWARD: dict[TicketState, TicketState] = {
    TicketState.INTAKE:  TicketState.CLARIFY,
    TicketState.CLARIFY: TicketState.RESOLVE,
    TicketState.RESOLVE: TicketState.CONFIRM,
    TicketState.CONFIRM: TicketState.CLOSED,
}

BACKWARD: dict[TicketState, TicketState] = {
    TicketState.CLARIFY: TicketState.INTAKE,
    TicketState.RESOLVE: TicketState.CLARIFY,
    TicketState.CONFIRM: TicketState.RESOLVE,
}


# ── Focused LLM agents (each is a single, narrow task) ───────────────────────
# Built via a factory so every state-machine instance gets its OWN agents.
# ADK forbids one agent having two parents, so module-level singletons cannot be
# reused across multiple TicketTriageStateMachine() instances.

def _build_agents() -> dict[str, LlmAgent]:
    # Intent detector: what does the user want to do right now?
    intent_agent = LlmAgent(
        name="IntentAgent",
        model=MODEL,
        description="Detects user intent from a message.",
        instruction="""
Current step in the support flow: {ticket_state}

Detect the user's intent given that step. Output ONLY this JSON:

{
  "action": "<provide_info|clarify|go_back|confirm|reject|cancel>",
  "notes": "<one-sentence summary of what the user said>"
}

Definitions:
  provide_info — user is giving new information about their ticket
  clarify      — user is asking a question or needs more guidance
  go_back      — user wants to revise or change something
  confirm      — user accepts the proposed resolution
  reject       — user rejects the resolution and wants a different one
  cancel       — user wants to abandon the ticket

IMPORTANT: when the current step is "confirm", the user is replying to a
yes/no confirmation prompt. Affirmative replies ("yes", "looks good", "that
works") → confirm. Negative replies ("no", "not quite", "still broken") →
reject.
""",
        output_key="intent",
    )

    # Completeness checker: do we have enough info to resolve?
    completeness_agent = LlmAgent(
        name="CompletenessAgent",
        model=MODEL,
        description="Checks whether a ticket has enough information to resolve.",
        instruction="""
Assess whether the ticket has enough information to ATTEMPT a resolution.
Be pragmatic and decisive: a support agent can resolve or escalate with the
core facts. Mark complete=true once you know the category and the basic
problem plus any one concrete specific (e.g. an order number, an amount, an
error/behaviour, a version). Do NOT demand exhaustive detail. Only mark
complete=false when the ticket is too vague to act on at all.

Ticket so far: {ticket_description}

Output ONLY this JSON:

{
  "complete": <true|false>,
  "missing": "<the single most important missing item, or empty string if complete>",
  "category": "<billing|technical|general>"
}
""",
        output_key="completeness",
    )

    # Clarification agent: ask a targeted follow-up question.
    clarify_agent = LlmAgent(
        name="ClarifyAgent",
        model=MODEL,
        description="Asks a targeted follow-up question to complete ticket info.",
        instruction="""
You need more information to resolve this ticket.

Ticket so far: {ticket_description}
What's missing: {missing_info}

Ask ONE clear, specific follow-up question to get the missing information.
Be concise and friendly. Output only the question — no labels, no JSON.
""",
        output_key="clarification_question",
    )

    # Resolver: generate a resolution based on category and description.
    resolver_agent = LlmAgent(
        name="ResolverAgent",
        model=MODEL,
        description="Generates a resolution for a complete ticket.",
        instruction="""
You are a support specialist. Generate a resolution.

Ticket category: {ticket_category}
Ticket description: {ticket_description}

Output ONLY this JSON:

{
  "response": "<your resolution text>",
  "escalate": <true if human escalation is needed, else false>,
  "escalation_reason": "<reason if escalate is true, else empty string>"
}
""",
        output_key="resolution",
    )

    # Confirmation presenter: format resolution for user approval.
    confirm_agent = LlmAgent(
        name="ConfirmAgent",
        model=MODEL,
        description="Presents the resolution to the user for confirmation.",
        instruction="""
Present the proposed resolution to the user and ask them to confirm or reject.

Resolution JSON: {resolution}

Write a friendly message that:
1. Summarises the proposed resolution clearly
2. Asks: "Does this resolve your issue? Reply 'yes' to confirm or 'no' to request changes."

If escalate is true in the resolution, explain that their case will be
escalated to a specialist.

Output only the message — no JSON, no labels.
""",
        output_key="confirmation_prompt",
    )

    return {
        "intent_agent": intent_agent,
        "completeness_agent": completeness_agent,
        "clarify_agent": clarify_agent,
        "resolver_agent": resolver_agent,
        "confirm_agent": confirm_agent,
    }


# ── Custom Agent: owns the state machine ─────────────────────────────────────

class TicketTriageStateMachine(BaseAgent):
    """
    State machine implemented as a CustomAgent.

    _run_async_impl holds all orchestration logic.
    LLM agents are called as narrow, typed sub-tasks.
    Code makes every state transition decision.
    """

    intent_agent:      LlmAgent
    completeness_agent: LlmAgent
    clarify_agent:     LlmAgent
    resolver_agent:    LlmAgent
    confirm_agent:     LlmAgent

    def __init__(self):
        agents = _build_agents()
        super().__init__(
            name="TicketTriageStateMachine",
            description="Interactive ticket triage with forward/back state transitions.",
            sub_agents=list(agents.values()),
            **agents,
        )

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        State machine loop.

        State changes are accumulated in `pending` and committed via the
        state_delta of the events we yield — that is the only thing ADK
        persists across turns. `ctx.session.state` is per-turn scratch (a fresh
        copy each turn), used here only for within-turn reads.
        """
        state = ctx.session.state
        pending: dict = {}

        def set_state(key: str, value) -> None:
            """Set state for both this turn (read) and future turns (persist)."""
            state[key] = value
            pending[key] = value

        # ── Initialise state on first turn ────────────────────────────────────
        if "ticket_state" not in state:
            set_state("ticket_state", TicketState.INTAKE.value)
            set_state("ticket_description", "")
            set_state("ticket_category", "")
            set_state("missing_info", "")
            set_state("info_rounds", 0)

        current = TicketState(state["ticket_state"])
        user_msg = ctx.user_content.parts[0].text if ctx.user_content else ""

        # ── Accumulate ticket description ─────────────────────────────────────
        if user_msg:
            set_state(
                "ticket_description",
                (state.get("ticket_description", "") + "\n" + user_msg).strip(),
            )

        # ── Detect intent (LLM call 1 — narrow extraction) ───────────────────
        await self._call(self.intent_agent, ctx, pending)
        intent_data = _parse_json(state.get("intent", "{}"))
        action      = intent_data.get("action", "provide_info")

        print(f"\n[STATE: {current.value.upper()}] [ACTION: {action}]")

        # ── Handle cancel (always available) ─────────────────────────────────
        if action == "cancel":
            set_state("ticket_state", TicketState.INTAKE.value)
            set_state("ticket_description", "")
            yield _text_event(ctx, "Ticket cancelled. Start a new one whenever you're ready.", pending)
            return

        # ── Handle go_back (always available except at INTAKE) ───────────────
        if action == "go_back" and current in BACKWARD:
            prev = BACKWARD[current]
            set_state("ticket_state", prev.value)
            yield _text_event(ctx, f"Going back to {prev.value}. What would you like to change?", pending)
            return

        # ── State-specific logic ──────────────────────────────────────────────

        if current == TicketState.INTAKE:
            # Check if we have enough info to move forward
            await self._call(self.completeness_agent, ctx, pending)

            completeness = _parse_json(state.get("completeness", "{}"))
            is_complete  = completeness.get("complete", False)
            missing      = completeness.get("missing", "")
            category     = completeness.get("category", "general")

            set_state("ticket_category", category)
            set_state("missing_info", missing)

            if is_complete or state.get("info_rounds", 0) >= MAX_INFO_ROUNDS:
                # Enough info (or we've asked enough) — advance to CLARIFY and
                # fall through this turn.
                set_state("ticket_state", TicketState.CLARIFY.value)
                current = TicketState.CLARIFY
            else:
                set_state("info_rounds", state.get("info_rounds", 0) + 1)
                yield _text_event(
                    ctx,
                    f"Thanks for reaching out. To help you, I need a bit more information.\n\n"
                    f"Missing: {missing}\n\nCould you provide those details?",
                    pending,
                )
                return

        if current == TicketState.CLARIFY:
            # Re-check completeness with accumulated description
            await self._call(self.completeness_agent, ctx, pending)

            completeness = _parse_json(state.get("completeness", "{}"))
            is_complete  = completeness.get("complete", True)
            missing      = completeness.get("missing", "")
            category     = completeness.get("category", state.get("ticket_category", "general"))

            set_state("ticket_category", category)

            if is_complete or state.get("info_rounds", 0) >= MAX_INFO_ROUNDS:
                # Advance to RESOLVE
                set_state("ticket_state", TicketState.RESOLVE.value)
                current = TicketState.RESOLVE
            else:
                # Stay — ask a targeted follow-up question
                set_state("info_rounds", state.get("info_rounds", 0) + 1)
                set_state("missing_info", missing)
                question = await self._call(self.clarify_agent, ctx, pending)
                yield _text_event(ctx, question, pending)
                return

        if current == TicketState.RESOLVE:
            # Generate resolution (LLM call)
            await self._call(self.resolver_agent, ctx, pending)

            # Advance to CONFIRM
            set_state("ticket_state", TicketState.CONFIRM.value)

            # Present resolution for user confirmation
            message = await self._call(self.confirm_agent, ctx, pending)
            yield _text_event(ctx, message, pending)
            return

        if current == TicketState.CONFIRM:
            # A yes/no confirmation is a binary the code can read directly;
            # the LLM intent is a fallback for richer phrasing.
            reply  = user_msg.strip().lower().rstrip(".!")
            affirm = reply in {"yes", "y", "yep", "yeah", "yup", "ok", "okay", "confirm", "correct"}
            deny   = reply in {"no", "n", "nope", "nah"}

            if action == "confirm" or affirm:
                resolution = _parse_json(state.get("resolution", "{}"))
                set_state("ticket_state", TicketState.CLOSED.value)

                if resolution.get("escalate"):
                    yield _text_event(
                        ctx,
                        "Your ticket has been escalated to a specialist. "
                        f"Reason: {resolution.get('escalation_reason', 'requires human review')}. "
                        "You will hear back within 24 hours. Ticket closed.",
                        pending,
                    )
                else:
                    yield _text_event(ctx, "Great — your ticket is resolved and closed. Have a good day!", pending)

            elif action in ("reject", "go_back") or deny:
                # Go back to resolve for a different resolution
                set_state("ticket_state", TicketState.RESOLVE.value)
                set_state(
                    "ticket_description",
                    state.get("ticket_description", "") + "\n[User requested a different resolution]",
                )
                yield _text_event(
                    ctx,
                    "Understood. Let me find a better solution. "
                    "Can you tell me what didn't work for you?",
                    pending,
                )
            else:
                # Ambiguous — re-present the confirmation
                message = await self._call(self.confirm_agent, ctx, pending)
                yield _text_event(ctx, message, pending)

        elif current == TicketState.CLOSED:
            yield _text_event(ctx, "This ticket is already closed. Open a new one if you need further help.", pending)

    async def _call(self, agent: LlmAgent, ctx: InvocationContext, pending: dict) -> str:
        """Run a sub-agent, applying its state_delta both live (so this turn can
        read its output_key) and into `pending` (so it persists). Returns the
        agent's final text — without yielding its events, so intermediate JSON
        stays out of the user-facing transcript.
        """
        text = ""
        async for event in agent.run_async(ctx):
            if event.actions and event.actions.state_delta:
                ctx.session.state.update(event.actions.state_delta)
                pending.update(event.actions.state_delta)
            if event.content and event.content.parts:
                chunk = "".join(p.text for p in event.content.parts if p.text)
                if chunk:
                    text = chunk
        return text


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    """Safely parse JSON from LLM output, stripping markdown fences."""
    try:
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(clean)
    except Exception:
        return {}


def _text_event(ctx: InvocationContext, text: str, state_delta: dict | None = None) -> Event:
    """Create a text event from the orchestrator.

    Any state_delta is what actually PERSISTS the orchestrator's state changes:
    ADK only commits state via event.actions.state_delta as events bubble up
    through the runner. Direct ctx.session.state mutation is per-turn scratch
    only — the session service hands each turn a fresh copy.
    """
    from google.adk.events import EventActions
    from google.genai.types import Content, Part
    return Event(
        author=ctx.agent.name,
        content=Content(parts=[Part(text=text)]),
        actions=EventActions(state_delta=dict(state_delta or {})),
    )


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_conversation(turns: list[str]) -> None:
    """Simulate a multi-turn conversation."""
    agent   = TicketTriageStateMachine()
    runner  = InMemoryRunner(agent=agent, app_name=APP)

    user_id    = "user-1"
    session_id = "session-sm-1"

    # Create the session on the runner's own service before the first turn.
    await runner.session_service.create_session(
        app_name=APP, user_id=user_id, session_id=session_id
    )

    print(f"\n{'═'*60}")
    print("NEW CONVERSATION")
    print(f"{'═'*60}")

    for turn in turns:
        print(f"\nUSER: {turn}")
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=types.Content(role="user", parts=[types.Part(text=turn)]),
        ):
            if hasattr(event, "content") and event.content:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        print(f"AGENT: {part.text}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Scenario 1: Happy path — billing ticket resolved without escalation
    asyncio.run(run_conversation([
        "I was charged twice last month.",
        "My email is alice@example.com, order #12345, charge was $49.99 on June 1.",
        "yes",
    ]))

    # Scenario 2: User goes back and revises
    asyncio.run(run_conversation([
        "App is broken.",
        "It crashes when I export to PDF. macOS 14, version 3.2.1.",
        "no",    # reject resolution
        "Actually the crash happens only with large files over 100MB.",
        "yes",
    ]))
