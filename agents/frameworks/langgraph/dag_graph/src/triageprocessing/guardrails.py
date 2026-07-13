"""Domain guardrails for triageprocessing.

Reject diversion mirrors onboarding's timeout diversion: happy_path always
proposes PUBLISH from AWAIT_APPROVAL; the PUBLISH guardrail redirects to
REJECT when approval_decision == reject. WRITE_REPORT → COMPLETE unless
the run was rejected, then → REJECTED.
"""

from __future__ import annotations

from typing import Dict

from src.engine.guardrail import (
    GuardrailFn,
    GuardrailResult,
    make_guardrail,
    make_transition_guardrail,
)

from .session_state import TriageState
from .state_transitions import State, is_transition_allowed

check_transition_allowed = make_transition_guardrail(State, is_transition_allowed, State.ERROR)


def check_not_rejected(state: TriageState) -> GuardrailResult:
    """On entry to PUBLISH: require explicit approve; divert rejects."""
    decision = state.get("approval_decision")
    if decision == "reject":
        return GuardrailResult(passed=False, reason="approval rejected", fallback=State.REJECT)
    if decision != "approve":
        return GuardrailResult(
            passed=False, reason="missing approve decision", fallback=State.ERROR
        )
    return GuardrailResult(passed=True)


def check_report_terminal(state: TriageState) -> GuardrailResult:
    """On entry to COMPLETE: rejected runs land in REJECTED instead."""
    if state.get("run_status") == "rejected" or state.get("approval_decision") == "reject":
        return GuardrailResult(passed=False, reason="run rejected", fallback=State.REJECTED)
    return GuardrailResult(passed=True)


guardrails: Dict[State, GuardrailFn] = {
    State.INTAKE: make_guardrail(check_transition_allowed),
    State.GATHER_CONTEXT: make_guardrail(check_transition_allowed),
    State.EXECUTE_PATCH: make_guardrail(check_transition_allowed),
    State.COLLECT_DIFF: make_guardrail(check_transition_allowed),
    State.AWAIT_APPROVAL: make_guardrail(check_transition_allowed),
    State.PUBLISH: make_guardrail(check_not_rejected, check_transition_allowed),
    State.REJECT: make_guardrail(check_transition_allowed),
    State.WRITE_REPORT: make_guardrail(check_transition_allowed),
    State.COMPLETE: make_guardrail(check_report_terminal, check_transition_allowed),
    State.REJECTED: make_guardrail(check_transition_allowed),
    State.ERROR: lambda _: GuardrailResult(passed=True),
}

__all__ = [
    "check_transition_allowed",
    "check_not_rejected",
    "check_report_terminal",
    "guardrails",
]
