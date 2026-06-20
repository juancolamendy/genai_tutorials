"""Guardrail framework with composable checks."""

import json
import time
from typing import Callable, Dict
from .state_machine import State, PipelineState, GuardrailResult, is_transition_allowed

GuardrailFn = Callable[[PipelineState], GuardrailResult]


def make_guardrail(*checks: GuardrailFn) -> GuardrailFn:
    """Compose multiple guardrail checks with short-circuit evaluation.

    Args:
        *checks: Variable number of GuardrailFn check functions

    Returns:
        A composed guardrail function that runs all checks in order
        and returns on first failure (short-circuit).
    """

    def _combined(state: PipelineState) -> GuardrailResult:
        for check in checks:
            result = check(state)
            if not result.passed:
                return result
        return GuardrailResult(passed=True)

    return _combined


def check_transition_allowed(state: PipelineState) -> GuardrailResult:
    """Validate that proposed transition is in ALLOWED_TRANSITIONS.

    Args:
        state: PipelineState with current_state and proposed_next

    Returns:
        GuardrailResult with fallback=State.ERROR if transition invalid
    """
    current = State(state["current_state"])
    proposed = State(state["proposed_next"])

    if not is_transition_allowed(current, proposed):
        return GuardrailResult(
            passed=False,
            reason=f"Transition not allowed: {current.value} → {proposed.value}",
            fallback=State.ERROR,
        )
    return GuardrailResult(passed=True)


def check_retry_budget_with_error_type(state: PipelineState) -> GuardrailResult:
    """Check retry budget and error type classification.

    Permanent errors are rejected immediately (no retry).
    Transient errors allowed up to MAX_RETRIES (3).

    Args:
        state: PipelineState with retry_count and error_type

    Returns:
        GuardrailResult with fallback=State.ERROR if budget exceeded or permanent
    """
    MAX_RETRIES = 3

    if state.get("error_type") == "permanent":
        return GuardrailResult(
            passed=False,
            reason="Permanent error; no retry allowed",
            fallback=State.ERROR,
        )

    if state["retry_count"] > MAX_RETRIES:
        return GuardrailResult(
            passed=False,
            reason=f"Retry budget exhausted ({state['retry_count']} > {MAX_RETRIES})",
            fallback=State.ERROR,
        )

    return GuardrailResult(passed=True)


def check_raw_data_present(state: PipelineState) -> GuardrailResult:
    """Check that raw_data is present (not None).

    Args:
        state: PipelineState with raw_data field

    Returns:
        GuardrailResult with fallback based on state
    """
    if state.get("raw_data") is None:
        return GuardrailResult(
            passed=False,
            reason="raw_data is absent",
            fallback=State.RETRY,
        )
    return GuardrailResult(passed=True)


def check_validated_data_present(state: PipelineState) -> GuardrailResult:
    """Check that validated_data is present (not None).

    Args:
        state: PipelineState with validated_data field

    Returns:
        GuardrailResult with fallback=State.HUMAN_REVIEW if missing
    """
    if state.get("validated_data") is None:
        return GuardrailResult(
            passed=False,
            reason="validated_data is absent",
            fallback=State.HUMAN_REVIEW,
        )
    return GuardrailResult(passed=True)


def check_enriched_data_present(state: PipelineState) -> GuardrailResult:
    """Check that enriched_data is present (not None).

    Args:
        state: PipelineState with enriched_data field

    Returns:
        GuardrailResult with fallback=State.RETRY if missing
    """
    if state.get("enriched_data") is None:
        return GuardrailResult(
            passed=False,
            reason="enriched_data is absent",
            fallback=State.RETRY,
        )
    return GuardrailResult(passed=True)


def check_document_size(state: PipelineState) -> GuardrailResult:
    """Check that document size does not exceed limit (10MB).

    Args:
        state: PipelineState with raw_data field

    Returns:
        GuardrailResult with fallback=State.ERROR if document too large
    """
    MAX_SIZE_BYTES = 10_000_000

    doc_size = len(json.dumps(state.get("raw_data", {})))
    if doc_size > MAX_SIZE_BYTES:
        return GuardrailResult(
            passed=False,
            reason=f"Document too large ({doc_size} bytes > {MAX_SIZE_BYTES})",
            fallback=State.ERROR,
        )
    return GuardrailResult(passed=True)


def check_fallback_depth(state: PipelineState) -> GuardrailResult:
    """Detect fallback cascade loops (max depth = 2).

    Args:
        state: PipelineState with fallback_depth field

    Returns:
        GuardrailResult with fallback=State.ERROR if depth exceeded
    """
    MAX_DEPTH = 2

    if state.get("fallback_depth", 0) > MAX_DEPTH:
        return GuardrailResult(
            passed=False,
            reason=f"Fallback cascade detected (depth {state['fallback_depth']} > {MAX_DEPTH})",
            fallback=State.ERROR,
        )
    return GuardrailResult(passed=True)


def check_pipeline_timeout(state: PipelineState) -> GuardrailResult:
    """Check that pipeline execution has not exceeded timeout.

    Args:
        state: PipelineState with started_at and node_timeout_seconds

    Returns:
        GuardrailResult with fallback=State.ERROR if timeout exceeded
    """
    timeout = state.get("node_timeout_seconds", 60)
    elapsed = time.time() - state["started_at"]

    if elapsed > timeout:
        return GuardrailResult(
            passed=False,
            reason=f"Pipeline timeout ({elapsed:.1f}s > {timeout}s)",
            fallback=State.ERROR,
        )
    return GuardrailResult(passed=True)


GUARDRAILS: Dict[State, GuardrailFn] = {
    State.FETCH: make_guardrail(
        check_transition_allowed,
        check_pipeline_timeout,
        check_retry_budget_with_error_type,
    ),
    State.VALIDATE: make_guardrail(
        check_transition_allowed,
        check_pipeline_timeout,
        check_raw_data_present,
        check_document_size,
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
        check_retry_budget_with_error_type,
    ),
    State.HUMAN_REVIEW: make_guardrail(
        check_transition_allowed,
        check_pipeline_timeout,
        check_fallback_depth,
    ),
    State.ERROR: lambda _: GuardrailResult(passed=True),
}
