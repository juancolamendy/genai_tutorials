"""Domain-specific guardrails for document processing pipeline.

Guardrails validate conditions before entering a state and can redirect
to a fallback state if the check fails (e.g., RETRY, HUMAN_REVIEW, ERROR).
"""

from __future__ import annotations

import time
from typing import Callable, Dict

from src.engine.guardrail import make_guardrail
from .state_machine import State, PipelineState, GuardrailResult, is_transition_allowed

# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def check_transition_allowed(state: PipelineState) -> GuardrailResult:
    """Validate that proposed transition is allowed by state machine.

    Args:
        state: PipelineState with current_state and proposed_next

    Returns:
        GuardrailResult with passed=True or fallback state
    """
    current = State(state["current_state"])
    proposed = State(state["proposed_next"])

    if is_transition_allowed(current, proposed):
        return GuardrailResult(passed=True)

    return GuardrailResult(
        passed=False,
        reason=f"Transition {current.value} → {proposed.value} is not in the state machine.",
        fallback=State.ERROR,
    )


def check_retry_budget(state: PipelineState) -> GuardrailResult:
    """Check retry budget exhaustion.

    Allows up to MAX_RETRIES (3) before rejecting further retries.

    Args:
        state: PipelineState with retry_count

    Returns:
        GuardrailResult with passed=True or fallback to ERROR
    """
    MAX_RETRIES = 3
    if state["retry_count"] <= MAX_RETRIES:
        return GuardrailResult(passed=True)

    return GuardrailResult(
        passed=False,
        reason=f"Retry budget exhausted ({state['retry_count']} attempts).",
        fallback=State.ERROR,
    )


def check_raw_data_present(state: PipelineState) -> GuardrailResult:
    """Check that raw_data is present before validation.

    Args:
        state: PipelineState with raw_data field

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


def check_validated_data_present(state: PipelineState) -> GuardrailResult:
    """Check that validated_data is present before enrichment.

    Args:
        state: PipelineState with validated_data field

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


def check_enriched_data_present(state: PipelineState) -> GuardrailResult:
    """Check that enriched_data is present before storage.

    Args:
        state: PipelineState with enriched_data field

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


def check_pipeline_timeout(state: PipelineState) -> GuardrailResult:
    """Check that pipeline execution has not exceeded timeout.

    Args:
        state: PipelineState with started_at and timeout_seconds

    Returns:
        GuardrailResult with passed=True or fallback to ERROR
    """
    started_at = state.get("started_at")
    if started_at is None:
        return GuardrailResult(passed=True)  # Skip if not set

    timeout_seconds = state.get("timeout_seconds", 300)  # 5 minute default
    elapsed = time.time() - started_at

    if elapsed > timeout_seconds:
        return GuardrailResult(
            passed=False,
            reason=f"Pipeline timeout ({elapsed:.1f}s > {timeout_seconds}s)",
            fallback=State.ERROR,
        )
    return GuardrailResult(passed=True)


def check_fallback_depth(state: PipelineState) -> GuardrailResult:
    """Detect fallback cascade loops (max depth = 2).

    Args:
        state: PipelineState with fallback_depth field

    Returns:
        GuardrailResult with passed=True or fallback to ERROR
    """
    MAX_DEPTH = 2

    depth = state.get("fallback_depth", 0)
    if depth > MAX_DEPTH:
        return GuardrailResult(
            passed=False,
            reason=f"Fallback cascade detected (depth {depth} > {MAX_DEPTH})",
            fallback=State.ERROR,
        )
    return GuardrailResult(passed=True)


# ─────────────────────────────────────────────────────────────────────────────
# GUARDRAIL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

GUARDRAILS: Dict[State, GuardrailFn] = {
    State.FETCH: make_guardrail(
        check_transition_allowed,
        check_pipeline_timeout,
        check_retry_budget,
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
    "GuardrailFn",
    "check_transition_allowed",
    "check_retry_budget",
    "check_raw_data_present",
    "check_validated_data_present",
    "check_enriched_data_present",
    "check_pipeline_timeout",
    "check_fallback_depth",
    "GUARDRAILS",
]
