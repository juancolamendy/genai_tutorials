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


def check_handler_status(state: HelpdeskState) -> GuardrailResult:
    """Divert to ERROR when the previous handler stamped handler_status=\"error\"."""
    if state.get("handler_status") == "error":
        return GuardrailResult(
            passed=False,
            reason=state.get("error_message") or "handler failed",
            fallback=State.ERROR,
        )
    return GuardrailResult(passed=True)


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
    State.HUB_CLARIFY: make_guardrail(check_handler_status, check_transition_allowed),
    State.TOPIC_FAQ: make_guardrail(check_handler_status, check_transition_allowed),
    State.TOPIC_ESCALATE: make_guardrail(check_handler_status, check_transition_allowed),
    State.TOPIC_BOOKING: make_guardrail(
        check_handler_status,
        check_booking_sticky,
        check_transition_allowed,
    ),
    State.NOTIFY_USER: make_guardrail(check_transition_allowed),
    State.IDLE: make_guardrail(check_transition_allowed),
    State.ERROR: lambda _: GuardrailResult(passed=True),
}

__all__ = [
    "check_transition_allowed",
    "check_handler_status",
    "check_booking_sticky",
    "guardrails",
]
