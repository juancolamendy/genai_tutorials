"""Generic guardrail framework with composable checks.

Provides guardrail composition patterns independent of domain-specific types.
"""

from typing import Any, Callable, Optional
from dataclasses import dataclass

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
