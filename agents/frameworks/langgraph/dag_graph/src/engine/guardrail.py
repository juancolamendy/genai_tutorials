"""Generic guardrail framework with composable checks.

Provides guardrail composition patterns independent of domain-specific types.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

# ─────────────────────────────────────────────────────────────────────────────
# GUARDRAIL RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    """Result of guardrail check."""

    passed: bool
    reason: str = ""
    fallback: Optional[Any] = None

# Generic guardrail result type
GuardrailFn = Callable[[dict[str, Any]], Any]


def make_guardrail(*checks: GuardrailFn) -> GuardrailFn:
    """Compose multiple guardrail checks with short-circuit evaluation.

    Args:
        *checks: Variable number of GuardrailFn check functions

    Returns:
        A composed guardrail function that runs all checks in order
        and returns on first failure (short-circuit).
    """

    def _combined(state: dict[str, Any]) -> Any:
        for check in checks:
            result = check(state)
            if not result.passed:
                return result
        # Return a passing result (domain-specific result type)
        from dataclasses import dataclass

        @dataclass
        class PassResult:
            passed: bool = True
            reason: str = ""

        return PassResult()

    return _combined


# ─────────────────────────────────────────────────────────────────────────────
# GENERIC CONTROL-PLANE CHECKS
#
# These only touch fields defined on the generic EngineSessionState
# (current_state, proposed_next, retry_count, started_at, timeout_seconds,
# fallback_depth), so any EngineGraph subclass can reuse them by supplying
# its own state enum / transition table / error fallback state.
# ─────────────────────────────────────────────────────────────────────────────

def make_transition_guardrail(
    state_enum: type,
    is_transition_allowed: Callable[[Any, Any], bool],
    error_state: Any,
) -> GuardrailFn:
    """Build a guardrail that rejects proposed_next if it isn't an allowed transition."""

    def check_transition_allowed(state: dict[str, Any]) -> GuardrailResult:
        current = state_enum(state["current_state"])
        proposed = state_enum(state["proposed_next"])

        if is_transition_allowed(current, proposed):
            return GuardrailResult(passed=True)

        return GuardrailResult(
            passed=False,
            reason=f"Transition {current.value} → {proposed.value} is not in the state machine.",
            fallback=error_state,
        )

    return check_transition_allowed


def make_retry_budget_guardrail(max_retries: int, error_state: Any) -> GuardrailFn:
    """Build a guardrail that rejects once retry_count exceeds max_retries."""

    def check_retry_budget(state: dict[str, Any]) -> GuardrailResult:
        if state["retry_count"] <= max_retries:
            return GuardrailResult(passed=True)

        return GuardrailResult(
            passed=False,
            reason=f"Retry budget exhausted ({state['retry_count']} attempts).",
            fallback=error_state,
        )

    return check_retry_budget


def make_timeout_guardrail(error_state: Any, default_timeout_seconds: float = 300.0) -> GuardrailFn:
    """Build a guardrail that rejects once elapsed time exceeds timeout_seconds."""

    def check_pipeline_timeout(state: dict[str, Any]) -> GuardrailResult:
        started_at = state.get("started_at")
        if started_at is None:
            return GuardrailResult(passed=True)  # Skip if not set

        timeout_seconds = state.get("timeout_seconds", default_timeout_seconds)
        elapsed = time.time() - started_at

        if elapsed > timeout_seconds:
            return GuardrailResult(
                passed=False,
                reason=f"Pipeline timeout ({elapsed:.1f}s > {timeout_seconds}s)",
                fallback=error_state,
            )
        return GuardrailResult(passed=True)

    return check_pipeline_timeout


def make_fallback_depth_guardrail(max_depth: int, error_state: Any) -> GuardrailFn:
    """Build a guardrail that rejects once fallback_depth exceeds max_depth (cascade detection)."""

    def check_fallback_depth(state: dict[str, Any]) -> GuardrailResult:
        depth = state.get("fallback_depth", 0)
        if depth > max_depth:
            return GuardrailResult(
                passed=False,
                reason=f"Fallback cascade detected (depth {depth} > {max_depth})",
                fallback=error_state,
            )
        return GuardrailResult(passed=True)

    return check_fallback_depth
