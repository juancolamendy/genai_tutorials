from dataclasses import dataclass
from typing import Any, Callable, Optional

# structures
@dataclass
class GuardrailResult:
    passed:   bool
    reason:   str           = ""
    fallback: Optional = None    # where to route when check fails


# types
# Callable alias so type annotations are concise everywhere.
GuardrailFn = Callable[[Any], GuardrailResult]


# variables
GUARDRAIL_PASS = GuardrailResult(passed=True)    # singleton for passing results


# functions
def make_guardrail(*checks: GuardrailFn) -> GuardrailFn:
    """
    Compose multiple checks into one GuardrailFn.

    Evaluation is short-circuit: the first failing check wins.
    If all pass, a single PASS sentinel is returned.

    Example:
        guard = make_guardrail(check_transition_allowed, check_retry_budget)
        result = guard(pipeline_state)
    """
    def _combined(state: Any) -> GuardrailResult:
        for check in checks:
            result = check(state)
            if not result.passed:
                return result
        return GUARDRAIL_PASS

    return _combined