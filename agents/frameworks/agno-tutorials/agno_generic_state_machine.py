"""
Agno — Generic State Machine Framework
─────────────────────────────────────────────────────────────────────────────
A reusable state machine framework for Agno Workflows.

Any scenario is expressed as configuration:
  • FORWARD / BACKWARD transition tables  (code-owned, no LLM)
  • StateConfig per state — guard type, executor, stay behavior, confirm logic
  • StateMachineConfig — wires states + tables + accumulation rules

Guard types
  • GuardType.LLM    — caller supplies a guard_fn; LLM decides advance/stay
  • GuardType.STAMPS — heuristic: all required_fields must be in session_state
  • GuardType.NONE   — no guard; always advances (auto-advance states)

Cancel and go_back are global handlers — always available, no config needed.

Install:
    uv add agno python-dotenv

Usage:
    uv run agno_generic_state_machine.py
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from dotenv import load_dotenv

from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.models.anthropic import Claude
from agno.workflow import Step, Workflow
from agno.workflow.types import StepInput, StepOutput

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════════
# GENERIC STATE MACHINE FRAMEWORK
# ══════════════════════════════════════════════════════════════════════════════

# ── Guard types ───────────────────────────────────────────────────────────────

class GuardType(str, Enum):
    LLM    = "llm"     # caller provides guard_fn; LLM decides
    STAMPS = "stamps"  # heuristic: all required_fields must be in session_state
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
    guard_fn:        Optional[Callable[[dict, str], GuardResult]] = None  # GuardType.LLM
    required_fields: list[str] = field(default_factory=list)              # GuardType.STAMPS

    # Runs just before advancing (mutates session_state in-place); args: (ss, user_msg)
    executor_fn: Optional[Callable[[dict, str], None]] = None

    # Called when guard fails (stay). args: (ss, user_msg, guard_message) → response str.
    # If None, guard_message is returned directly.
    on_stay_fn: Optional[Callable[[dict, str, str], str]] = None

    # Confirm-state behaviour (intent-based branching instead of guard)
    is_confirm:    bool = False
    on_present_fn: Optional[Callable[[dict], str]] = None   # default branch — present output
    on_confirm_fn: Optional[Callable[[dict], str]] = None   # confirm branch — return closing msg
    on_reject_fn:  Optional[Callable[[dict], None]] = None  # reject branch — cleanup ss
    on_reject_msg: str = "Let me find a different approach. What didn't work for you?"

    # Terminal state — just echo terminal_msg, no transitions
    is_terminal:  bool = False
    terminal_msg: str  = "This session is closed."

    # Session-state keys to clear when navigating BACKWARD from this state
    clears_on_back: list[str] = field(default_factory=list)


@dataclass
class StateMachineConfig:
    states:        dict[str, StateConfig]  # state_name → config
    forward:       dict[str, str]          # state_name → next state name
    backward:      dict[str, str]          # state_name → previous state name
    initial_state: str

    state_key:                   str            = "current_state"
    accumulation_key:            Optional[str]  = None   # ss key to append user_msg into
    accumulation_exclude_states: list[str]      = field(default_factory=list)
    cancel_message:              str            = "Session cancelled. Start a new one whenever you're ready."


# ── Framework-level intent agent ──────────────────────────────────────────────
# Intentionally generic — the five actions map directly to the global handlers
# (cancel, go_back) and confirm-state branches (confirm, reject, provide_info).

_intent_agent = Agent(
    name="IntentAgent",
    model=Claude(id="claude-sonnet-4-6"),
    instructions=[
        "Detect the user's intent from their message.",
        "Output ONLY valid JSON (no markdown):",
        '{"action": "<provide_info|go_back|confirm|reject|cancel>"}',
        "provide_info: user giving new information or asking a question",
        "go_back: user wants to revise or undo the last step",
        "confirm: user accepts the current proposal",
        "reject: user rejects the current proposal",
        "cancel: user wants to abandon this session entirely",
    ],
)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _parse(raw: str) -> dict:
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


def _llm(agent: Agent, prompt: str) -> str:
    result = agent.run(prompt)
    return result.content if hasattr(result, "content") else str(result)


def _detect_intent(user_msg: str) -> str:
    raw  = _llm(_intent_agent, user_msg)
    data = _parse(raw)
    return data.get("action", "provide_info")


# ── Factory ───────────────────────────────────────────────────────────────────

def make_state_machine_executor(config: StateMachineConfig) -> Callable:
    """
    Return a Step executor that drives a generic state machine.

    Usage:
        Step(name="SM", executor=make_state_machine_executor(my_config))

    The returned function has the signature Agno expects:
        (step_input: StepInput, session_state: dict) -> StepOutput

    Fall-through is handled by an internal loop: when a guarded or auto-advance
    state passes its check, the loop immediately processes the next state in the
    same call, matching the original chained-if pattern without recursion.
    """

    def _executor(step_input: StepInput, session_state: dict) -> StepOutput:
        ss       = session_state
        user_msg = step_input.input or ""

        # ── 1. Current state ─────────────────────────────────────────────────
        current = ss.get(config.state_key, config.initial_state)
        print(f"\n[STATE: {current.upper()}]")

        # ── 2. Intent detection (one LLM call per turn, always) ──────────────
        action = _detect_intent(user_msg)
        print(f"[INTENT] → {action}")

        # ── 3. Cancel — global, always available ─────────────────────────────
        if action == "cancel":
            ss.clear()
            ss[config.state_key] = config.initial_state
            return StepOutput(content=config.cancel_message)

        # ── 4. Go back — global, when backward table has an entry ────────────
        if action == "go_back" and current in config.backward:
            prev      = config.backward[current]
            state_cfg = config.states.get(current)
            if state_cfg:
                for key in state_cfg.clears_on_back:
                    ss.pop(key, None)
            ss[config.state_key] = prev
            return StepOutput(
                content=f"Going back to {prev}. What would you like to change?"
            )

        # ── 5. Accumulate user message (if configured) ────────────────────────
        if (
            config.accumulation_key
            and user_msg
            and current not in config.accumulation_exclude_states
        ):
            existing = ss.get(config.accumulation_key, "")
            ss[config.accumulation_key] = (existing + "\n" + user_msg).strip()

        # ── 6. Main state-machine loop (handles fall-through via iteration) ───
        MAX_TRANSITIONS = 10
        for _ in range(MAX_TRANSITIONS):
            current   = ss.get(config.state_key, config.initial_state)
            state_cfg = config.states.get(current)

            if state_cfg is None:
                return StepOutput(content=f"Unknown state '{current}'. Configuration error.")

            # ── 6a. Terminal state ────────────────────────────────────────────
            if state_cfg.is_terminal:
                return StepOutput(content=state_cfg.terminal_msg)

            # ── 6b. Confirm state — intent-based branching ───────────────────
            if state_cfg.is_confirm:
                if action == "confirm":
                    next_state = config.forward.get(current)
                    if next_state is None:
                        return StepOutput(content="No forward transition from confirm state.")
                    ss[config.state_key] = next_state
                    msg = state_cfg.on_confirm_fn(ss) if state_cfg.on_confirm_fn else "Confirmed."
                    return StepOutput(content=msg)

                if action == "reject":
                    prev = config.backward.get(current)
                    if prev is None:
                        return StepOutput(content="Cannot go back from here.")
                    if state_cfg.on_reject_fn:
                        state_cfg.on_reject_fn(ss)
                    ss[config.state_key] = prev
                    return StepOutput(content=state_cfg.on_reject_msg)

                # Default — present current output and wait for confirmation
                msg = (
                    state_cfg.on_present_fn(ss)
                    if state_cfg.on_present_fn
                    else "Please confirm or reject."
                )
                ss[config.state_key] = current
                return StepOutput(content=msg)

            # ── 6c. Guard check ───────────────────────────────────────────────
            if state_cfg.guard_type == GuardType.LLM:
                if state_cfg.guard_fn is None:
                    guard_result = GuardResult(advance=True)
                else:
                    guard_result = state_cfg.guard_fn(ss, user_msg)

            elif state_cfg.guard_type == GuardType.STAMPS:
                all_present  = all(ss.get(f) for f in state_cfg.required_fields)
                missing_keys = [f for f in state_cfg.required_fields if not ss.get(f)]
                guard_result = GuardResult(
                    advance=all_present,
                    message="" if all_present else f"Missing required fields: {missing_keys}",
                )

            else:  # GuardType.NONE
                guard_result = GuardResult(advance=True)

            # ── 6d. Guard failed → stay ───────────────────────────────────────
            if not guard_result.advance:
                ss[config.state_key] = current
                if state_cfg.on_stay_fn:
                    return StepOutput(
                        content=state_cfg.on_stay_fn(ss, user_msg, guard_result.message)
                    )
                return StepOutput(content=guard_result.message or "More information needed.")

            # ── 6e. Guard passed → run executor, then advance (fall-through) ──
            if state_cfg.executor_fn:
                state_cfg.executor_fn(ss, user_msg)

            next_state = config.forward.get(current)
            if next_state is None:
                return StepOutput(
                    content=f"No forward transition configured for state '{current}'."
                )
            ss[config.state_key] = next_state
            # Continue loop — next iteration picks up next_state

        return StepOutput(content="State machine loop limit exceeded. Configuration error.")

    return _executor


# ══════════════════════════════════════════════════════════════════════════════
# DEMO: Ticket Triage (rewritten using the generic framework)
# ══════════════════════════════════════════════════════════════════════════════

# ── States ────────────────────────────────────────────────────────────────────

class TicketState(str, Enum):
    INTAKE  = "intake"
    CLARIFY = "clarify"
    RESOLVE = "resolve"
    CONFIRM = "confirm"
    CLOSED  = "closed"


# ── Domain LLM agents ─────────────────────────────────────────────────────────

completeness_agent = Agent(
    name="CompletenessAgent",
    model=Claude(id="claude-sonnet-4-6"),
    instructions=[
        "Assess whether the ticket has enough information to resolve.",
        "Output ONLY valid JSON (no markdown):",
        '{"complete": true/false, "missing": "<what is needed or empty string>", "category": "<billing|technical|general>"}',
    ],
)

clarify_agent = Agent(
    name="ClarifyAgent",
    model=Claude(id="claude-sonnet-4-6"),
    instructions=[
        "Ask ONE targeted follow-up question to get missing ticket information.",
        "Be concise and friendly.",
        "Output only the question — no JSON, no labels.",
    ],
)

resolver_agent = Agent(
    name="ResolverAgent",
    model=Claude(id="claude-sonnet-4-6"),
    instructions=[
        "Generate a support resolution.",
        "Output ONLY valid JSON (no markdown):",
        '{"response": "<resolution>", "escalate": false, "escalation_reason": ""}',
        "Set escalate=true only if human specialist intervention is needed.",
    ],
)

confirm_presenter_agent = Agent(
    name="ConfirmPresenterAgent",
    model=Claude(id="claude-sonnet-4-6"),
    instructions=[
        "Present the resolution to the user and ask for confirmation.",
        "Write a friendly message summarising the resolution.",
        'End with: "Does this resolve your issue? Reply yes to confirm or no for changes."',
        "Output only the message — no JSON.",
    ],
)


# ── Domain callables ──────────────────────────────────────────────────────────

def check_completeness(ss: dict, user_msg: str) -> GuardResult:
    """LLM guard: is the ticket complete enough to resolve?"""
    completeness = _parse(_llm(
        completeness_agent,
        f"Ticket: {ss.get('ticket_description', '')}"
    ))
    is_complete = completeness.get("complete", False)
    missing     = completeness.get("missing", "")
    cat         = completeness.get("category", "general")
    ss["ticket_category"] = cat
    msg = f"Thanks for reaching out. I need a bit more information.\n\n{missing}" if missing else "More information needed."
    return GuardResult(advance=is_complete, message=msg)


def ask_clarification(ss: dict, user_msg: str, guard_msg: str) -> str:
    """on_stay_fn for CLARIFY: call clarify_agent to ask a targeted follow-up."""
    description = ss.get("ticket_description", "")
    missing     = guard_msg  # guard_msg contains the missing-info explanation
    return _llm(
        clarify_agent,
        f"Ticket so far: {description}\nWhat's missing: {missing}"
    )


def generate_resolution(ss: dict, user_msg: str) -> None:
    """executor_fn for RESOLVE: call resolver and store result in session_state."""
    res_raw = _llm(
        resolver_agent,
        f"Category: {ss.get('ticket_category', 'general')}\nTicket: {ss.get('ticket_description', '')}"
    )
    ss["resolution"] = json.dumps(_parse(res_raw))


def present_resolution(ss: dict) -> str:
    """on_present_fn for CONFIRM: format the resolution for the user."""
    res_data = _parse(ss.get("resolution", "{}"))
    return _llm(confirm_presenter_agent, json.dumps(res_data))


def close_ticket(ss: dict) -> str:
    """on_confirm_fn for CONFIRM: return the appropriate closing message."""
    res_data = _parse(ss.get("resolution", "{}"))
    if res_data.get("escalate"):
        reason = res_data.get("escalation_reason", "requires human review")
        return (
            f"Your ticket has been escalated to a specialist. "
            f"Reason: {reason}. You will hear back within 24 hours. Ticket closed."
        )
    return "Ticket resolved and closed. Have a great day!"


def reject_resolution(ss: dict) -> None:
    """on_reject_fn for CONFIRM: clear resolution and annotate description."""
    ss["resolution"] = ""
    existing = ss.get("ticket_description", "")
    ss["ticket_description"] = existing + "\n[User requested a different resolution]"


# ── StateConfig objects ───────────────────────────────────────────────────────
#
# INTAKE:  LLM guard (completeness); on_stay=None → returns guard_msg directly
# CLARIFY: LLM guard (same completeness check); on_stay=ask_clarification (extra LLM call)
# RESOLVE: No guard (GuardType.NONE) — always runs executor and advances
# CONFIRM: Confirm state — intent-based branching (confirm / reject / present)
# CLOSED:  Terminal — no transitions

intake_cfg = StateConfig(
    guard_type=GuardType.LLM,
    guard_fn=check_completeness,
    on_stay_fn=None,               # guard_msg returned directly (includes preamble)
)

clarify_cfg = StateConfig(
    guard_type=GuardType.LLM,
    guard_fn=check_completeness,
    on_stay_fn=ask_clarification,  # calls clarify_agent for a targeted follow-up
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
    on_reject_msg="Understood. Let me find a different approach. What didn't work for you?",
    clears_on_back=["resolution"],
)

closed_cfg = StateConfig(
    is_terminal=True,
    terminal_msg="This ticket is already closed. Open a new one if you need help.",
)


# ── StateMachineConfig ────────────────────────────────────────────────────────

ticket_sm_config = StateMachineConfig(
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
    state_key="ticket_state",
    accumulation_key="ticket_description",
    accumulation_exclude_states=[TicketState.CONFIRM, TicketState.CLOSED],
    cancel_message="Ticket cancelled. Start a new one whenever you're ready.",
)


# ── Workflow ───────────────────────────────────────────────────────────────────

state_machine_workflow = Workflow(
    name="TicketTriageStateMachine",
    db=InMemoryDb(),
    session_state={},
    steps=[
        Step(
            name="StateMachine",
            executor=make_state_machine_executor(ticket_sm_config),
        )
    ],
)


# ── Simulate multi-turn conversation ──────────────────────────────────────────

def run_conversation(title: str, turns: list[str], session_id: str) -> None:
    print(f"\n{'═'*60}")
    print(f"SCENARIO: {title}")
    print(f"{'═'*60}")
    for msg in turns:
        print(f"\nUSER: {msg}")
        response = state_machine_workflow.run(msg, session_id=session_id)
        content  = response.content if hasattr(response, "content") else response
        print(f"AGENT: {content}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Scenario 1: Happy path
    run_conversation(
        "Happy path — billing ticket",
        [
            "I was charged twice last month.",
            "Order #12345, charge $49.99 on June 1st, email alice@example.com.",
            "yes",
        ],
        session_id="scenario-1-happy-path",
    )

    # Scenario 2: User revises their ticket
    run_conversation(
        "Revision — technical ticket",
        [
            "App is broken.",
            "It crashes on PDF export. macOS 14, app version 3.2.1.",
            "no",
            "The crash only happens with files over 100MB. Here's the error: OOM exception.",
            "yes",
        ],
        session_id="scenario-2-revision",
    )

    # Scenario 3: Back transition then cancel
    run_conversation(
        "Back transition then cancel",
        [
            "I need help with my account.",
            "I want to change my subscription plan from monthly to annual.",
            "go back",
            "cancel",
        ],
        session_id="scenario-3-back-cancel",
    )
