"""
ADK — Generic State Machine Framework
─────────────────────────────────────────────────────────────────────────────
A reusable state machine framework for ADK CustomAgents.

Any scenario is expressed as configuration:
  • FORWARD / BACKWARD transition tables  (code-owned, no LLM)
  • StateConfig per state — guard type, executor, stay behavior, confirm logic
  • StateMachineConfig — wires states + tables + accumulation rules

Guard types
  • GuardType.LLM    — caller supplies an async guard_fn; LLM decides advance/stay
  • GuardType.STAMPS — heuristic: all required_fields must be in session state
  • GuardType.NONE   — no guard; always advances (auto-advance states)

Cancel and go_back are global handlers — always available, no config needed.

ADK specifics vs Agno
  • All non-trivial callbacks are async coroutines
  • Callbacks receive a `call(agent) -> str` helper and a `set_s(key, val)` setter
    so they can invoke sub-LLM-agents and persist state without knowing about ctx
  • State persists via event.actions.state_delta — direct ctx.session.state mutation
    is per-turn scratch only; set_s() writes to both scratch and pending delta
  • intent_agent must be a fresh instance per StateMachineAgent (ADK single-parent rule)
  • domain_agents are passed into StateMachineConfig and included in sub_agents

Install:
    uv add google-adk python-dotenv

Usage:
    uv run adk_generic_state_machine.py
"""

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Optional

from dotenv import load_dotenv
from google.adk.agents import LlmAgent
from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.genai import types
from pydantic import ConfigDict

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# GENERIC STATE MACHINE FRAMEWORK
# ══════════════════════════════════════════════════════════════════════════════

# ── Guard types ───────────────────────────────────────────────────────────────

class GuardType(str, Enum):
    LLM    = "llm"     # caller provides async guard_fn; LLM decides
    STAMPS = "stamps"  # heuristic: all required_fields must be in session state
    NONE   = "none"    # always advances


# ── Core dataclasses ──────────────────────────────────────────────────────────

@dataclass
class GuardResult:
    advance: bool
    message: str = ""


@dataclass
class StateConfig:
    # Guard behaviour — controls whether this state advances or stays
    guard_type:      GuardType = GuardType.NONE
    # async (ss, user_msg, call, set_s) -> GuardResult
    guard_fn:        Optional[Callable] = None
    required_fields: list[str] = field(default_factory=list)   # for STAMPS

    # Runs just before advancing (mutates session state via set_s); async.
    # Signature: async (ss, user_msg, call, set_s) -> None
    executor_fn: Optional[Callable] = None

    # Called when guard fails (stay behaviour). Returns user-facing string.
    # Signature: async (ss, user_msg, guard_message, call, set_s) -> str
    # If None, guard_message is returned directly.
    on_stay_fn: Optional[Callable] = None

    # Confirm-state behaviour (intent-based branching instead of guard)
    is_confirm: bool = False
    # async (ss, call, set_s) -> str  — present current output to user
    on_present_fn: Optional[Callable] = None
    # (ss) -> str  — sync; return closing message after confirm
    on_confirm_fn: Optional[Callable] = None
    # (ss, set_s) -> None  — sync; clean up session state on reject
    on_reject_fn:  Optional[Callable] = None
    on_reject_msg: str = "Let me find a different approach. What didn't work for you?"

    # Terminal state — just echo terminal_msg, no transitions
    is_terminal:  bool = False
    terminal_msg: str  = "This session is closed."

    # Session-state keys to clear (set to "") when navigating BACKWARD from this state
    clears_on_back: list[str] = field(default_factory=list)


@dataclass
class StateMachineConfig:
    states:        dict[str, StateConfig]  # state_name → config
    forward:       dict[str, str]          # state_name → next state name
    backward:      dict[str, str]          # state_name → previous state name
    initial_state: str

    # All LlmAgents that domain callbacks will invoke via call().
    # These are registered as sub_agents so ADK sets up the correct context chain.
    domain_agents: list = field(default_factory=list)    # list[LlmAgent]

    state_key:                   str           = "current_state"
    accumulation_key:            Optional[str] = None    # ss key to append user_msg into
    accumulation_exclude_states: list[str]     = field(default_factory=list)
    cancel_message:              str           = "Session cancelled. Start a new one whenever you're ready."

    # Safety: force advance from a guarded state after this many consecutive stays.
    # Set to 0 to disable. Prevents over-strict LLM guards trapping the machine.
    max_stay_rounds: int = 2


# ── Framework-level helpers ───────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    """Safely parse JSON from LLM output, stripping markdown fences."""
    try:
        clean = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        return json.loads(clean)
    except Exception:
        return {}


def _text_event(ctx: InvocationContext, text: str, state_delta: dict | None = None) -> Event:
    """Create a user-facing text event.

    state_delta is what actually PERSISTS the orchestrator's state changes:
    ADK only commits state via event.actions.state_delta as events bubble up
    through the runner. Direct ctx.session.state mutation is per-turn scratch.
    """
    from google.adk.events import EventActions
    from google.genai.types import Content, Part
    return Event(
        author=ctx.agent.name,
        content=Content(parts=[Part(text=text)]),
        actions=EventActions(state_delta=dict(state_delta or {})),
    )


def _build_intent_agent(name_prefix: str = "") -> LlmAgent:
    """Build a fresh intent-detection agent (fresh instance required per StateMachineAgent)."""
    agent_name = f"{name_prefix}IntentAgent" if name_prefix else "IntentAgent"
    return LlmAgent(
        name=agent_name,
        model="gemini-2.5-flash",
        description="Detects user intent from a message.",
        instruction="""
Detect the user's intent from their message. Output ONLY this JSON:

{
  "action": "<provide_info|go_back|confirm|reject|cancel>"
}

Definitions:
  provide_info — user is giving new information or asking a question
  go_back      — user wants to revise or undo the previous step
  confirm      — user accepts the current proposal (yes, looks good, that works, etc.)
  reject       — user rejects the proposal and wants a different one (no, not quite, etc.)
  cancel       — user wants to abandon this session entirely

When the user is responding to a yes/no confirmation prompt, treat affirmative
replies (yes, y, yep, ok, correct) as confirm and negative replies as reject.
""",
        output_key="intent",
    )


# ── Generic StateMachineAgent ─────────────────────────────────────────────────

class StateMachineAgent(BaseAgent):
    """
    Generic state machine implemented as an ADK BaseAgent.

    Use make_state_machine_agent() to create instances.
    _run_async_impl owns all orchestration. LLM agents are narrow sub-tasks.
    Code makes every state-transition decision.
    """

    # arbitrary_types_allowed lets Pydantic accept StateConfig's Callable fields
    model_config = ConfigDict(arbitrary_types_allowed=True)

    sm_config:    Any        # StateMachineConfig — Any avoids Pydantic callable validation
    intent_agent: LlmAgent

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """
        Generic state machine loop.

        set_s() writes to both ctx.session.state (this-turn readable) and
        `pending` (accumulated into event.actions.state_delta for persistence).
        """
        state = ctx.session.state
        pending: dict = {}

        def set_s(key: str, value: Any) -> None:
            state[key] = value
            pending[key] = value

        # ── 1. Initialize session state on first turn ─────────────────────
        if self.sm_config.state_key not in state:
            set_s(self.sm_config.state_key, self.sm_config.initial_state)
            set_s("stay_rounds", 0)
            if self.sm_config.accumulation_key:
                set_s(self.sm_config.accumulation_key, "")

        current  = state[self.sm_config.state_key]
        user_msg = ctx.user_content.parts[0].text if ctx.user_content else ""

        # ── 2. Accumulate user message (if configured) ────────────────────
        if (
            self.sm_config.accumulation_key
            and user_msg
            and current not in self.sm_config.accumulation_exclude_states
        ):
            existing = state.get(self.sm_config.accumulation_key, "")
            set_s(self.sm_config.accumulation_key, (existing + "\n" + user_msg).strip())

        # ── 3. Detect intent (one LLM call per turn, always) ─────────────
        await self._call(self.intent_agent, ctx, pending)
        intent_data = _parse_json(state.get("intent", "{}"))
        action      = intent_data.get("action", "provide_info")

        print(f"\n[STATE: {current.upper()}] [ACTION: {action}]")

        # ── 4. Cancel (global, always available) ─────────────────────────
        if action == "cancel":
            set_s(self.sm_config.state_key, self.sm_config.initial_state)
            set_s("stay_rounds", 0)
            if self.sm_config.accumulation_key:
                set_s(self.sm_config.accumulation_key, "")
            yield _text_event(ctx, self.sm_config.cancel_message, pending)
            return

        # ── 5. Go back (global, when backward table has an entry) ─────────
        if action == "go_back" and current in self.sm_config.backward:
            prev      = self.sm_config.backward[current]
            state_cfg = self.sm_config.states.get(current)
            if state_cfg:
                for key in state_cfg.clears_on_back:
                    set_s(key, "")
            set_s(self.sm_config.state_key, prev)
            set_s("stay_rounds", 0)
            yield _text_event(
                ctx, f"Going back to {prev}. What would you like to change?", pending
            )
            return

        # ── call helper: runs a sub-agent, applies its state_delta ─────────
        async def call(agent: LlmAgent) -> str:
            return await self._call(agent, ctx, pending)

        # ── 6. Main state-machine loop (handles fall-through via iteration)─
        MAX_TRANSITIONS = 10
        for _ in range(MAX_TRANSITIONS):
            current   = state[self.sm_config.state_key]
            state_cfg = self.sm_config.states.get(current)

            if state_cfg is None:
                yield _text_event(ctx, f"Unknown state '{current}'. Configuration error.", pending)
                return

            # ── 6a. Terminal state ────────────────────────────────────────
            if state_cfg.is_terminal:
                yield _text_event(ctx, state_cfg.terminal_msg, pending)
                return

            # ── 6b. Confirm state (intent + keyword branching) ────────────
            if state_cfg.is_confirm:
                # Keyword check matches concrete yes/no; LLM intent handles richer phrasing
                reply  = user_msg.strip().lower().rstrip(".!")
                affirm = reply in {"yes", "y", "yep", "yeah", "yup", "ok", "okay", "confirm", "correct"}
                deny   = reply in {"no", "n", "nope", "nah"}

                if action == "confirm" or affirm:
                    next_state = self.sm_config.forward.get(current)
                    if next_state is None:
                        yield _text_event(ctx, "No forward transition from confirm state.", pending)
                        return
                    set_s(self.sm_config.state_key, next_state)
                    set_s("stay_rounds", 0)
                    msg = state_cfg.on_confirm_fn(state) if state_cfg.on_confirm_fn else "Confirmed."
                    yield _text_event(ctx, msg, pending)
                    return

                if action == "reject" or deny:
                    prev = self.sm_config.backward.get(current)
                    if prev is None:
                        yield _text_event(ctx, "Cannot go back from here.", pending)
                        return
                    if state_cfg.on_reject_fn:
                        state_cfg.on_reject_fn(state, set_s)
                    set_s(self.sm_config.state_key, prev)
                    set_s("stay_rounds", 0)
                    yield _text_event(ctx, state_cfg.on_reject_msg, pending)
                    return

                # Default: present current output and wait for confirmation
                msg = (
                    await state_cfg.on_present_fn(state, call, set_s)
                    if state_cfg.on_present_fn
                    else "Please confirm or reject."
                )
                yield _text_event(ctx, msg, pending)
                return

            # ── 6c. Guard check ───────────────────────────────────────────
            if state_cfg.guard_type == GuardType.LLM:
                guard_result = (
                    await state_cfg.guard_fn(state, user_msg, call, set_s)
                    if state_cfg.guard_fn
                    else GuardResult(advance=True)
                )
            elif state_cfg.guard_type == GuardType.STAMPS:
                all_present  = all(state.get(f) for f in state_cfg.required_fields)
                missing_keys = [f for f in state_cfg.required_fields if not state.get(f)]
                guard_result = GuardResult(
                    advance=all_present,
                    message="" if all_present else f"Missing required fields: {missing_keys}",
                )
            else:  # GuardType.NONE
                guard_result = GuardResult(advance=True)

            # Safety: force advance after max_stay_rounds consecutive stays
            stay_rounds = state.get("stay_rounds", 0)
            if (
                not guard_result.advance
                and self.sm_config.max_stay_rounds > 0
                and stay_rounds >= self.sm_config.max_stay_rounds
            ):
                guard_result = GuardResult(advance=True)

            # ── 6d. Guard failed → stay ───────────────────────────────────
            if not guard_result.advance:
                set_s("stay_rounds", stay_rounds + 1)
                set_s(self.sm_config.state_key, current)
                if state_cfg.on_stay_fn:
                    msg = await state_cfg.on_stay_fn(
                        state, user_msg, guard_result.message, call, set_s
                    )
                else:
                    msg = guard_result.message or "More information needed."
                yield _text_event(ctx, msg, pending)
                return

            # ── 6e. Guard passed → executor, then advance (fall-through) ──
            set_s("stay_rounds", 0)
            if state_cfg.executor_fn:
                await state_cfg.executor_fn(state, user_msg, call, set_s)

            next_state = self.sm_config.forward.get(current)
            if next_state is None:
                yield _text_event(
                    ctx, f"No forward transition for state '{current}'.", pending
                )
                return
            set_s(self.sm_config.state_key, next_state)
            # Continue loop — next iteration picks up next_state

        yield _text_event(ctx, "State machine loop limit exceeded. Configuration error.", pending)

    async def _call(self, agent: LlmAgent, ctx: InvocationContext, pending: dict) -> str:
        """Run a sub-agent, applying its state_delta both live and into pending.

        Returns the agent's final text without yielding intermediate events,
        keeping internal JSON out of the user-facing transcript.
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


def make_state_machine_agent(
    config: StateMachineConfig,
    name: str,
    description: str,
) -> StateMachineAgent:
    """
    Create a StateMachineAgent from config.

    Builds a fresh intent_agent per call (ADK forbids shared agent parents).
    config.domain_agents must list every LlmAgent that domain callbacks invoke.
    """
    intent = _build_intent_agent(name)
    return StateMachineAgent(
        name=name,
        description=description,
        sm_config=config,
        intent_agent=intent,
        sub_agents=[intent] + config.domain_agents,
    )


# ══════════════════════════════════════════════════════════════════════════════
# DEMO: Ticket Triage (rewritten using the generic framework)
# ══════════════════════════════════════════════════════════════════════════════

APP   = "ticket-triage-sm"
MODEL = "gemini-2.5-flash"


# ── States ────────────────────────────────────────────────────────────────────

class TicketState(str, Enum):
    INTAKE  = "intake"
    CLARIFY = "clarify"
    RESOLVE = "resolve"
    CONFIRM = "confirm"
    CLOSED  = "closed"


# ── Domain agent factory ──────────────────────────────────────────────────────
# Returns fresh instances — ADK forbids one agent having two parents, so this
# must be called once per StateMachineAgent instance.

def _build_ticket_agents() -> dict[str, LlmAgent]:
    completeness_agent = LlmAgent(
        name="CompletenessAgent",
        model=MODEL,
        description="Checks whether a ticket has enough information to resolve.",
        instruction="""
Assess whether the ticket has enough information to ATTEMPT a resolution.
Be pragmatic: a support agent can resolve or escalate with core facts.
Mark complete=true once you know the category and the basic problem plus
any one concrete specific (order number, error, version, amount).
Only mark complete=false when the ticket is too vague to act on at all.

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

    clarify_agent = LlmAgent(
        name="ClarifyAgent",
        model=MODEL,
        description="Asks a targeted follow-up question to complete ticket info.",
        instruction="""
You need more information to resolve this ticket.

Ticket so far: {ticket_description}
What's missing: {missing_info}

Ask ONE clear, specific follow-up question. Be concise and friendly.
Output only the question — no labels, no JSON.
""",
        output_key="clarification_question",
    )

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

    confirm_agent = LlmAgent(
        name="ConfirmAgent",
        model=MODEL,
        description="Presents the resolution to the user for confirmation.",
        instruction="""
Present the proposed resolution to the user and ask them to confirm or reject.

Resolution JSON: {resolution}

Write a friendly message that:
1. Summarises the proposed resolution clearly
2. Ends with: "Does this resolve your issue? Reply 'yes' to confirm or 'no' to request changes."

If escalate is true, explain the case will be escalated to a specialist.

Output only the message — no JSON, no labels.
""",
        output_key="confirmation_prompt",
    )

    return {
        "completeness_agent": completeness_agent,
        "clarify_agent":      clarify_agent,
        "resolver_agent":     resolver_agent,
        "confirm_agent":      confirm_agent,
    }


# ── Domain callables ──────────────────────────────────────────────────────────
# All LLM-calling callbacks are async and receive (ss, ..., call, set_s).
# `call(agent)` runs the agent and returns its text; ss is updated automatically
# via output_key + state_delta. `set_s(key, val)` persists additional state.

def _make_ticket_sm_config(agents: dict[str, LlmAgent]) -> StateMachineConfig:
    """Build a StateMachineConfig for ticket triage, closing over the given agents."""

    completeness_agent = agents["completeness_agent"]
    clarify_agent      = agents["clarify_agent"]
    resolver_agent     = agents["resolver_agent"]
    confirm_agent      = agents["confirm_agent"]

    # ── Guard functions ───────────────────────────────────────────────────────

    async def check_completeness(ss: dict, user_msg: str, call, set_s) -> GuardResult:
        """LLM guard: is the ticket complete enough to resolve?"""
        await call(completeness_agent)       # writes to ss["completeness"] via output_key
        data        = _parse_json(ss.get("completeness", "{}"))
        is_complete = data.get("complete", False)
        missing     = data.get("missing", "")
        category    = data.get("category", "general")
        set_s("ticket_category", category)
        set_s("missing_info", missing)
        msg = (
            f"Thanks for reaching out. To help you, I need a bit more information.\n\n"
            f"Missing: {missing}\n\nCould you provide those details?"
            if missing
            else "More information needed."
        )
        return GuardResult(advance=is_complete, message=msg)

    # ── Stay functions ────────────────────────────────────────────────────────

    async def ask_clarification(ss: dict, user_msg: str, guard_msg: str, call, set_s) -> str:
        """on_stay_fn for CLARIFY: ask a targeted follow-up via clarify_agent."""
        # missing_info was set by check_completeness; clarify_agent reads it from ss
        question = await call(clarify_agent)   # reads {ticket_description} + {missing_info}
        return question

    # ── Executor functions ────────────────────────────────────────────────────

    async def generate_resolution(ss: dict, user_msg: str, call, set_s) -> None:
        """executor_fn for RESOLVE: call resolver_agent, persist result."""
        await call(resolver_agent)   # reads {ticket_category} + {ticket_description}
        # resolver_agent writes to ss["resolution"] via output_key

    # ── Confirm-state functions ───────────────────────────────────────────────

    async def present_resolution(ss: dict, call, set_s) -> str:
        """on_present_fn for CONFIRM: format resolution for user approval."""
        message = await call(confirm_agent)   # reads {resolution} from ss
        return message

    def close_ticket(ss: dict) -> str:
        """on_confirm_fn for CONFIRM: return closing message."""
        resolution = _parse_json(ss.get("resolution", "{}"))
        if resolution.get("escalate"):
            reason = resolution.get("escalation_reason", "requires human review")
            return (
                f"Your ticket has been escalated to a specialist. "
                f"Reason: {reason}. You will hear back within 24 hours. Ticket closed."
            )
        return "Great — your ticket is resolved and closed. Have a good day!"

    def reject_resolution(ss: dict, set_s) -> None:
        """on_reject_fn for CONFIRM: annotate description, clear resolution."""
        existing = ss.get("ticket_description", "")
        set_s("ticket_description", existing + "\n[User requested a different resolution]")

    # ── StateConfig objects ───────────────────────────────────────────────────
    #
    # INTAKE:  LLM guard (completeness); on_stay=None → returns guard_msg directly
    # CLARIFY: LLM guard (same check); on_stay=ask_clarification (extra LLM call)
    # RESOLVE: No guard (GuardType.NONE) — always runs executor and advances
    # CONFIRM: Confirm state — intent + keyword branching (confirm / reject / present)
    # CLOSED:  Terminal — no transitions

    intake_cfg = StateConfig(
        guard_type=GuardType.LLM,
        guard_fn=check_completeness,
        on_stay_fn=None,               # returns guard_msg directly (includes preamble)
    )

    clarify_cfg = StateConfig(
        guard_type=GuardType.LLM,
        guard_fn=check_completeness,
        on_stay_fn=ask_clarification,  # calls clarify_agent for a targeted question
    )

    resolve_cfg = StateConfig(
        guard_type=GuardType.NONE,
        executor_fn=generate_resolution,
        clears_on_back=["resolution"],
    )

    confirm_cfg = StateConfig(
        is_confirm=True,
        on_present_fn=present_resolution,
        on_confirm_fn=close_ticket,
        on_reject_fn=reject_resolution,
        on_reject_msg="Understood. Let me find a better solution. What didn't work for you?",
        clears_on_back=["resolution"],
    )

    closed_cfg = StateConfig(
        is_terminal=True,
        terminal_msg="This ticket is already closed. Open a new one if you need further help.",
    )

    return StateMachineConfig(
        states={
            TicketState.INTAKE:  intake_cfg,
            TicketState.CLARIFY: clarify_cfg,
            TicketState.RESOLVE: resolve_cfg,
            TicketState.CONFIRM: confirm_cfg,
            TicketState.CLOSED:  closed_cfg,
        },
        forward={
            TicketState.INTAKE:  TicketState.CLARIFY,
            TicketState.CLARIFY: TicketState.RESOLVE,
            TicketState.RESOLVE: TicketState.CONFIRM,
            TicketState.CONFIRM: TicketState.CLOSED,
        },
        backward={
            TicketState.CLARIFY: TicketState.INTAKE,
            TicketState.RESOLVE: TicketState.CLARIFY,
            TicketState.CONFIRM: TicketState.RESOLVE,
        },
        initial_state=TicketState.INTAKE,
        domain_agents=list(agents.values()),
        state_key="ticket_state",
        accumulation_key="ticket_description",
        # ADK original accumulates on every turn — no exclusion for CONFIRM/CLOSED
        accumulation_exclude_states=[],
        cancel_message="Ticket cancelled. Start a new one whenever you're ready.",
        max_stay_rounds=2,
    )


def make_ticket_triage_agent() -> StateMachineAgent:
    """
    Factory: build a fresh TicketTriage StateMachineAgent.

    Call once per run_conversation() to get isolated agent instances
    (ADK forbids sharing agents across multiple parent agents).
    """
    agents = _build_ticket_agents()
    config = _make_ticket_sm_config(agents)
    return make_state_machine_agent(
        config,
        name="TicketTriageStateMachine",
        description="Interactive ticket triage with forward/back state transitions.",
    )


# ── Runner ────────────────────────────────────────────────────────────────────

async def run_conversation(title: str, turns: list[str]) -> None:
    """Simulate a multi-turn conversation using a fresh agent instance."""
    agent   = make_ticket_triage_agent()
    runner  = InMemoryRunner(agent=agent, app_name=APP)

    user_id    = "user-1"
    session_id = "session-sm-1"

    await runner.session_service.create_session(
        app_name=APP, user_id=user_id, session_id=session_id
    )

    print(f"\n{'═'*60}")
    print(f"SCENARIO: {title}")
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


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Scenario 1: Happy path — billing ticket resolved without escalation
    asyncio.run(run_conversation(
        "Happy path — billing ticket",
        [
            "I was charged twice last month.",
            "My email is alice@example.com, order #12345, charge was $49.99 on June 1.",
            "yes",
        ],
    ))

    # Scenario 2: User revises their ticket
    asyncio.run(run_conversation(
        "Revision — technical ticket",
        [
            "App is broken.",
            "It crashes when I export to PDF. macOS 14, version 3.2.1.",
            "no",
            "Actually the crash happens only with large files over 100MB.",
            "yes",
        ],
    ))
