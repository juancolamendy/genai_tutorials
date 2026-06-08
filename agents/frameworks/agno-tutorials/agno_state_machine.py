"""
Agno — State Machine Pattern: Ticket Triage
─────────────────────────────────────────────────────────────────────────────
Shape:  INTAKE ↔ CLARIFY ↔ RESOLVE ↔ CONFIRM → CLOSED
        Interactive, stateful, supports back-transitions and revision.
        Agno Workflow v2 with session state + a single Python function
        executor that owns the entire state machine loop.
        LLM agents are narrow extraction/generation tasks only.

Install:
    uv add agno python-dotenv

Usage:
    uv run agno_state_machine.py
"""

import json
from enum import Enum

from dotenv import load_dotenv
from pydantic import BaseModel

from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.models.anthropic import Claude
from agno.workflow import Step, Workflow
from agno.workflow.types import StepInput, StepOutput

load_dotenv()


# ── States ────────────────────────────────────────────────────────────────────

class TicketState(str, Enum):
    INTAKE  = "intake"    # Gather initial ticket description
    CLARIFY = "clarify"   # Ask follow-up questions if info is incomplete
    RESOLVE = "resolve"   # Generate a resolution
    CONFIRM = "confirm"   # Ask user to confirm or revise
    CLOSED  = "closed"    # Done


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

# Stamps: what fields must be non-empty to advance from each state
STAMPS: dict[TicketState, list[str]] = {
    TicketState.CLARIFY: ["ticket_description", "ticket_category"],
    TicketState.RESOLVE: ["ticket_description", "ticket_category"],
    TicketState.CONFIRM: ["resolution"],
    TicketState.CLOSED:  ["resolution"],
}


# ── Focused LLM agents ────────────────────────────────────────────────────────

intent_agent = Agent(
    name="IntentAgent",
    model=Claude(id="claude-sonnet-4-6"),
    instructions=[
        "Detect the user's intent from their message.",
        "Output ONLY valid JSON (no markdown):",
        '{"action": "<provide_info|go_back|confirm|reject|cancel>"}',
        "provide_info: user giving new information",
        "go_back: user wants to revise something",
        "confirm: user accepts the resolution",
        "reject: user rejects the resolution",
        "cancel: user wants to abandon the ticket",
    ],
)

completeness_agent = Agent(
    name="CompletenessAgent",
    model=Claude(id="claude-sonnet-4-6"),
    instructions=[
        "Assess whether the ticket has enough information to resolve.",
        "Output ONLY valid JSON (no markdown):",
        '{"complete": true/false, "missing": "<what is needed or empty>", "category": "<billing|technical|general>"}',
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


# ── Helper: safe JSON parse ────────────────────────────────────────────────────

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


def _has_stamp(state: TicketState, ctx_state: dict) -> bool:
    """Check if all required fields for a state are present."""
    required = STAMPS.get(state, [])
    return all(ctx_state.get(f) for f in required)


# ── State Machine: single Python function owns all orchestration ──────────────
#
# This is the Agno equivalent of ADK's _run_async_impl.
# One function, explicit state transitions, LLM calls only where necessary.
# Session state persists across workflow turns via the `session_state` dict,
# which Agno injects (and persists to the workflow `db`) when the executor
# declares a `session_state` parameter. Mutating it in place is what persists.

def ticket_state_machine(step_input: StepInput, session_state: dict) -> StepOutput:
    """
    Stateful ticket triage.

    State lives in the `session_state` dict, which persists across turns via the
    workflow's db (keyed by session_id). Code owns all transitions. LLMs extract
    data only.
    """
    # ── Load persisted state (mutate `ss` in place — it IS session_state) ─────
    ss = session_state

    current_raw = ss.get("ticket_state", TicketState.INTAKE)
    current     = TicketState(current_raw)
    description = ss.get("ticket_description", "")
    category    = ss.get("ticket_category", "")
    resolution  = ss.get("resolution", "")
    user_msg    = step_input.input or ""

    print(f"\n[STATE: {current.value.upper()}]")

    # ── Accumulate ticket description ─────────────────────────────────────────
    if user_msg and current not in (TicketState.CONFIRM, TicketState.CLOSED):
        description = (description + "\n" + user_msg).strip()
        ss["ticket_description"] = description

    # ── Detect intent (LLM call — narrow extraction) ─────────────────────────
    intent_raw  = _llm(intent_agent, user_msg)
    intent_data = _parse(intent_raw)
    action      = intent_data.get("action", "provide_info")

    print(f"[INTENT] → {action}")

    # ── Cancel: always available ──────────────────────────────────────────────
    if action == "cancel":
        ss.clear()
        ss["ticket_state"] = TicketState.INTAKE
        return StepOutput(content="Ticket cancelled. Start a new one whenever you're ready.")

    # ── Go back: always available (except at INTAKE) ──────────────────────────
    if action == "go_back" and current in BACKWARD:
        prev = BACKWARD[current]
        ss["ticket_state"] = prev
        # Invalidate stamps for current state
        if current == TicketState.CONFIRM:
            ss["resolution"] = ""
        if current == TicketState.RESOLVE:
            ss["resolution"] = ""
        return StepOutput(content=f"Going back to {prev.value}. What would you like to change?")

    # ─────────────────────────────────────────────────────────────────────────
    # INTAKE state
    # ─────────────────────────────────────────────────────────────────────────
    if current == TicketState.INTAKE:
        # Check completeness (LLM call)
        completeness = _parse(_llm(
            completeness_agent,
            f"Ticket: {description}"
        ))

        is_complete = completeness.get("complete", False)
        missing     = completeness.get("missing", "")
        cat         = completeness.get("category", "general")

        ss["ticket_category"] = cat

        if not is_complete:
            ss["ticket_state"] = TicketState.INTAKE
            return StepOutput(
                content=f"Thanks for reaching out. I need a bit more information.\n\n{missing}"
            )

        # Stamp earned — advance to CLARIFY (which will immediately pass through to RESOLVE)
        ss["ticket_state"] = TicketState.CLARIFY
        current = TicketState.CLARIFY

    # ─────────────────────────────────────────────────────────────────────────
    # CLARIFY state
    # ─────────────────────────────────────────────────────────────────────────
    if current == TicketState.CLARIFY:
        completeness = _parse(_llm(
            completeness_agent,
            f"Ticket: {description}"
        ))

        is_complete = completeness.get("complete", True)
        missing     = completeness.get("missing", "")
        cat         = completeness.get("category", ss.get("ticket_category", "general"))

        ss["ticket_category"] = cat

        if not is_complete:
            # Stay — ask a follow-up question
            question = _llm(
                clarify_agent,
                f"Ticket so far: {description}\nWhat's missing: {missing}"
            )
            ss["ticket_state"] = TicketState.CLARIFY
            return StepOutput(content=question)

        # Stamp earned — advance to RESOLVE
        ss["ticket_state"] = TicketState.RESOLVE
        current = TicketState.RESOLVE

    # ─────────────────────────────────────────────────────────────────────────
    # RESOLVE state
    # ─────────────────────────────────────────────────────────────────────────
    if current == TicketState.RESOLVE:
        res_raw = _llm(
            resolver_agent,
            f"Category: {ss.get('ticket_category', 'general')}\nTicket: {description}"
        )
        res_data = _parse(res_raw)

        ss["resolution"]    = json.dumps(res_data)
        ss["ticket_state"]  = TicketState.CONFIRM
        current = TicketState.CONFIRM

    # ─────────────────────────────────────────────────────────────────────────
    # CONFIRM state
    # ─────────────────────────────────────────────────────────────────────────
    if current == TicketState.CONFIRM:
        if action == "confirm":
            # Ticket resolved — advance to CLOSED
            res_data = _parse(ss.get("resolution", "{}"))
            ss["ticket_state"] = TicketState.CLOSED
            if res_data.get("escalate"):
                return StepOutput(
                    content=(
                        "Your ticket has been escalated to a specialist. "
                        f"Reason: {res_data.get('escalation_reason', 'requires human review')}. "
                        "You will hear back within 24 hours. Ticket closed."
                    )
                )
            return StepOutput(content="Ticket resolved and closed. Have a great day!")

        if action in ("reject", "go_back"):
            # User wants a different resolution — go back to RESOLVE
            ss["ticket_state"]  = TicketState.RESOLVE
            ss["resolution"]    = ""
            description        += "\n[User requested a different resolution]"
            ss["ticket_description"] = description
            return StepOutput(
                content="Understood. Let me find a different approach. What didn't work for you?"
            )

        # Present the resolution for confirmation
        res_data = _parse(ss.get("resolution", "{}"))
        prompt   = _llm(
            confirm_presenter_agent,
            json.dumps(res_data)
        )
        ss["ticket_state"] = TicketState.CONFIRM
        return StepOutput(content=prompt)

    # ─────────────────────────────────────────────────────────────────────────
    # CLOSED state
    # ─────────────────────────────────────────────────────────────────────────
    if current == TicketState.CLOSED:
        return StepOutput(content="This ticket is already closed. Open a new one if you need help.")

    return StepOutput(content="Something unexpected happened. Please try again.")


# ── Workflow ───────────────────────────────────────────────────────────────────

state_machine_workflow = Workflow(
    name="TicketTriageStateMachine",
    # A db is required for session_state to persist across run() calls.
    db=InMemoryDb(),
    session_state={},
    steps=[
        Step(
            name="StateMachine",
            executor=ticket_state_machine,
        )
    ],
)


# ── Simulate multi-turn conversation ──────────────────────────────────────────

def run_conversation(title: str, turns: list[str], session_id: str) -> None:
    print(f"\n{'═'*60}")
    print(f"SCENARIO: {title}")
    print(f"{'═'*60}")

    # A stable session_id ties every turn in this scenario to one persisted
    # session, so session_state accumulates across run() calls. A distinct
    # session_id per scenario keeps the scenarios isolated from each other.
    for msg in turns:
        print(f"\nUSER: {msg}")
        response = state_machine_workflow.run(msg, session_id=session_id)
        content = response.content if hasattr(response, "content") else response
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
            "no",   # reject resolution
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
            "go back",   # triggers backward transition
            "cancel",
        ],
        session_id="scenario-3-back-cancel",
    )
