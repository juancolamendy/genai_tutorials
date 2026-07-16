"""Domain-specific guardrails for the HR helpdesk chatbot."""

from __future__ import annotations

from typing import Dict

from src.engine.guardrail import (
    GuardrailFn,
    GuardrailResult,
    make_guardrail,
    make_transition_guardrail,
)

from .session_state import HelpdeskState
from .state_transitions import State, is_transition_allowed

check_transition_allowed = make_transition_guardrail(State, is_transition_allowed, State.ERROR)


def _booking_confirmed(state: HelpdeskState) -> bool:
    topic_data = state.get("topic_data") or {}
    return bool(topic_data.get("booking_confirmed"))


def check_booking_sticky(state: HelpdeskState) -> GuardrailResult:
    """Self-loop TOPIC_BOOKING until confirmed; otherwise proceed to IDLE."""
    if state.get("active_topic") != "booking":
        return GuardrailResult(passed=True)
    if _booking_confirmed(state):
        return GuardrailResult(passed=True)
    return GuardrailResult(
        passed=False,
        reason="booking incomplete",
        fallback=State.TOPIC_BOOKING,
    )


guardrails: Dict[State, GuardrailFn] = {
    # handler_status → ERROR is enforced centrally in EngineGraph._guardrail_node.
    State.CLARIFY: make_guardrail(check_transition_allowed),
    State.TOPIC_FAQ: make_guardrail(check_transition_allowed),
    State.TOPIC_ESCALATE: make_guardrail(check_transition_allowed),
    State.TOPIC_BOOKING: make_guardrail(
        check_booking_sticky,
        check_transition_allowed,
    ),
    State.NOTIFY_USER: make_guardrail(check_transition_allowed),
    State.IDLE: make_guardrail(check_transition_allowed),
    State.ERROR: lambda _: GuardrailResult(passed=True),
}

__all__ = [
    "check_transition_allowed",
    "check_booking_sticky",
    "guardrails",
]
