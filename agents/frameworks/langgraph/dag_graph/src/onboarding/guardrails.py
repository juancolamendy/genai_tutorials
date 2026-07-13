"""Domain-specific guardrails for the onboarding pipeline.

Guardrails validate conditions before entering a state and can redirect
to a fallback state if the check fails (e.g., ESCALATED, ERROR).
"""

from __future__ import annotations

from typing import Dict

from src.engine.guardrail import (
    GuardrailFn,
    GuardrailResult,
    make_guardrail,
    make_transition_guardrail,
)

from .session_state import OnboardingState
from .state_transitions import State, is_transition_allowed

# ─────────────────────────────────────────────────────────────────────────────
# GENERIC CHECKS (built from src.engine.guardrail factories)
# ─────────────────────────────────────────────────────────────────────────────

check_transition_allowed = make_transition_guardrail(State, is_transition_allowed, State.ERROR)

# ─────────────────────────────────────────────────────────────────────────────
# ONBOARDING-SPECIFIC CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def check_new_hire_details_complete(state: OnboardingState) -> GuardrailResult:
    """Gate for COLLECT -> WELCOME_SENT: the router always proposes
    WELCOME_SENT from COLLECT (code routing, no LLM in the transition
    decision — design spec §2.3), so this guardrail is what actually
    implements COLLECT's self-loop ("collect --> collect: guardrail
    fallback (details incomplete)", design spec §3) — falls back to
    COLLECT, not ERROR, since incomplete details on an ordinary turn is
    expected, not a failure.
    """
    details = state.get("new_hire_details") or {}
    if all(details.get(k) for k in ("full_name", "role", "start_date")):
        return GuardrailResult(passed=True)
    return GuardrailResult(
        passed=False, reason="new hire details incomplete", fallback=State.COLLECT
    )


def check_not_timeout_escalation(state: OnboardingState) -> GuardrailResult:
    """Diverts to ESCALATED when a park state resumed via a timeout event
    rather than the happy event (design spec §8). Zero new engine
    mechanism — the router's happy_path always proposes the normal next
    state regardless of which legal event resumed the park state, so
    diversion is just an ordinary guardrail fallback check, run first in
    the composed guardrail so it short-circuits check_transition_allowed.

    Attached to IT_PROVISIONED and SCHEDULE_SENT — the happy-path targets
    of AWAIT_DOCUMENTS_SIGNED and AWAIT_HARDWARE_DELIVERED respectively,
    the two states this can actually divert away from.
    """
    if state.get("current_event_type") == "timeout_escalation":
        return GuardrailResult(passed=False, reason="timeout escalation", fallback=State.ESCALATED)
    return GuardrailResult(passed=True)


def check_handler_status(state: OnboardingState) -> GuardrailResult:
    """Divert to ERROR when the previous handler stamped handler_status="error".

    Business failures are caught in the handler (not via session status — that
    field is overwritten by _dispatch_handler). Run this check FIRST on the
    happy-path successor of any fallible handler.
    """
    if state.get("handler_status") == "error":
        return GuardrailResult(
            passed=False,
            reason=state.get("error_message") or "handler failed",
            fallback=State.ERROR,
        )
    return GuardrailResult(passed=True)


# ─────────────────────────────────────────────────────────────────────────────
# GUARDRAIL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

guardrails: Dict[State, GuardrailFn] = {
    # COLLECT deliberately does NOT include a cascade-detection check
    # (found in adversarial review, design spec §8): check_fallback_depth
    # is meant to catch automated retry loops, but COLLECT's self-fallback
    # fires on every ordinary turn where the human hasn't yet supplied all
    # required fields — a normal multi-turn clarifying conversation would
    # otherwise get killed to ERROR even though nothing is actually looping.
    State.COLLECT: make_guardrail(check_transition_allowed),
    # check_handler_status first: collect LLM/tool failure → ERROR, not the
    # incomplete-details self-loop. Then details-complete → COLLECT loop.
    State.WELCOME_SENT: make_guardrail(
        check_handler_status,
        check_new_hire_details_complete,
        check_transition_allowed,
    ),
    State.AWAIT_DOCUMENTS_SIGNED: make_guardrail(check_handler_status, check_transition_allowed),
    # check_not_timeout_escalation runs FIRST so it short-circuits
    # check_transition_allowed when a timeout resume diverts to ESCALATED.
    State.IT_PROVISIONED: make_guardrail(check_not_timeout_escalation, check_transition_allowed),
    # username_chain failure in IT_PROVISIONED is detected here on the way in.
    State.AWAIT_HARDWARE_DELIVERED: make_guardrail(
        check_handler_status, check_transition_allowed
    ),
    State.SCHEDULE_SENT: make_guardrail(check_not_timeout_escalation, check_transition_allowed),
    State.COMPLETE: make_guardrail(check_transition_allowed),
    State.ESCALATED: make_guardrail(check_transition_allowed),
    State.ERROR: lambda _: GuardrailResult(passed=True),  # error is always reachable
}

__all__ = [
    "check_transition_allowed",
    "check_new_hire_details_complete",
    "check_not_timeout_escalation",
    "check_handler_status",
    "guardrails",
]
