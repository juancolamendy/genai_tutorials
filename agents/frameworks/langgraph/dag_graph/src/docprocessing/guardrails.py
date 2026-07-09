"""Domain-specific guardrails for document processing pipeline.

Guardrails validate conditions before entering a state and can redirect
to a fallback state if the check fails (e.g., RETRY, HUMAN_REVIEW, ERROR).
"""

from __future__ import annotations

from typing import Dict

from src.engine.guardrail import (
    GuardrailFn,
    GuardrailResult,
    make_fallback_depth_guardrail,
    make_guardrail,
    make_retry_budget_guardrail,
    make_timeout_guardrail,
    make_transition_guardrail,
)

from .session_state import SessionState
from .state_transitions import State, is_transition_allowed

# ─────────────────────────────────────────────────────────────────────────────
# GENERIC CHECKS (built from src.engine.guardrail factories)
# ─────────────────────────────────────────────────────────────────────────────

check_transition_allowed = make_transition_guardrail(State, is_transition_allowed, State.ERROR)
check_retry_budget = make_retry_budget_guardrail(max_retries=3, error_state=State.ERROR)
check_pipeline_timeout = make_timeout_guardrail(error_state=State.ERROR)
check_fallback_depth = make_fallback_depth_guardrail(max_depth=2, error_state=State.ERROR)

# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT-SPECIFIC CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def check_raw_data_present(state: SessionState) -> GuardrailResult:
    """Check that raw_data is present before validation.

    Args:
        state: SessionState with raw_data field

    Returns:
        GuardrailResult with passed=True or fallback to RETRY
    """
    if state.get("raw_data"):
        return GuardrailResult(passed=True)

    return GuardrailResult(
        passed=False,
        reason="raw_data is missing; cannot proceed to validate.",
        fallback=State.RETRY,
    )


def check_validated_data_present(state: SessionState) -> GuardrailResult:
    """Check that validated_data is present before enrichment.

    Args:
        state: SessionState with validated_data field

    Returns:
        GuardrailResult with passed=True or fallback to HUMAN_REVIEW
    """
    if state.get("validated_data"):
        return GuardrailResult(passed=True)

    return GuardrailResult(
        passed=False,
        reason="validated_data is missing; document may need human review.",
        fallback=State.HUMAN_REVIEW,
    )


def check_enriched_data_present(state: SessionState) -> GuardrailResult:
    """Check that enriched_data is present before storage.

    Args:
        state: SessionState with enriched_data field

    Returns:
        GuardrailResult with passed=True or fallback to RETRY
    """
    if state.get("enriched_data"):
        return GuardrailResult(passed=True)

    return GuardrailResult(
        passed=False,
        reason="enriched_data is missing; cannot store.",
        fallback=State.RETRY,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GUARDRAIL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

guardrails: Dict[State, GuardrailFn] = {
    State.FETCH: make_guardrail(
        check_transition_allowed,
        check_pipeline_timeout,
        check_retry_budget,
        check_fallback_depth,
    ),
    State.UPLOAD_DOCUMENTS: make_guardrail(
        check_transition_allowed,
        check_pipeline_timeout,
        check_fallback_depth,
    ),
    State.VALIDATE: make_guardrail(
        check_transition_allowed,
        check_pipeline_timeout,
        check_raw_data_present,
        check_fallback_depth,
    ),
    State.ENRICH: make_guardrail(
        check_transition_allowed,
        check_pipeline_timeout,
        check_validated_data_present,
        check_fallback_depth,
    ),
    State.STORE: make_guardrail(
        check_transition_allowed,
        check_pipeline_timeout,
        check_enriched_data_present,
    ),
    State.COMPLETE: make_guardrail(
        check_transition_allowed,
        check_pipeline_timeout,
    ),
    State.RETRY: make_guardrail(
        check_transition_allowed,
        check_pipeline_timeout,
        check_retry_budget,
        check_fallback_depth,
    ),
    State.HUMAN_REVIEW: make_guardrail(
        check_transition_allowed,
        check_pipeline_timeout,
        check_fallback_depth,
    ),
    State.ERROR: lambda _: GuardrailResult(passed=True),  # error is always reachable
}

__all__ = [
    "check_transition_allowed",
    "check_retry_budget",
    "check_raw_data_present",
    "check_validated_data_present",
    "check_enriched_data_present",
    "check_pipeline_timeout",
    "check_fallback_depth",
    "guardrails",
]
