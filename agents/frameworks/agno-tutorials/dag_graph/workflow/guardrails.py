from dataclasses import dataclass
from typing import Callable, Optional

from engine.guardrail import GuardrailResult, GuardrailFn, GUARDRAIL_PASS, make_guardrail

from pipeline.pipeline_state import PipelineState
from pipeline.state_machine import State, is_transition_allowed

# functions
# ── Individual checks ─────────────────────────────────────────────────────────
def check_transition_allowed(state: PipelineState) -> GuardrailResult:
    """Reject any transition not declared in ALLOWED_TRANSITIONS."""
    current  = State(state["current_state"])
    proposed = State(state["proposed_next"])
    if is_transition_allowed(current, proposed):
        return GUARDRAIL_PASS
    return GuardrailResult(
        passed   = False,
        reason   = f"Illegal transition {current.value} → {proposed.value}.",
        fallback = State.ERROR,
    )


def check_retry_budget(state: PipelineState, max_retries: int = 3) -> GuardrailResult:
    """Reject when retry_count has exceeded the budget."""
    if state["retry_count"] <= max_retries:
        return GUARDRAIL_PASS
    return GuardrailResult(
        passed   = False,
        reason   = f"Retry budget exhausted ({state['retry_count']} / {max_retries}).",
        fallback = State.ERROR,
    )


def check_raw_data_present(state: PipelineState) -> GuardrailResult:
    """Require raw_data before proceeding to VALIDATE."""
    if state.get("raw_data"):
        return GUARDRAIL_PASS
    return GuardrailResult(
        passed   = False,
        reason   = "raw_data is absent; fetch may have failed.",
        fallback = State.RETRY,
    )


def check_validated_data_present(state: PipelineState) -> GuardrailResult:
    """Require validated_data before proceeding to ENRICH."""
    if state.get("validated_data"):
        return GUARDRAIL_PASS
    return GuardrailResult(
        passed   = False,
        reason   = "validated_data is absent; document needs review.",
        fallback = State.HUMAN_REVIEW,
    )


def check_enriched_data_present(state: PipelineState) -> GuardrailResult:
    """Require enriched_data before proceeding to STORE."""
    if state.get("enriched_data"):
        return GUARDRAIL_PASS
    return GuardrailResult(
        passed   = False,
        reason   = "enriched_data is absent; enrichment may have failed.",
        fallback = State.RETRY,
    )


# ── GUARDRAILS registry ───────────────────────────────────────────────────────
# Maps each destination state to a composed GuardrailFn.
# The guardrail for state X is evaluated BEFORE entering state X.

GUARDRAILS: dict[State, GuardrailFn] = {
    State.FETCH:        make_guardrail(check_transition_allowed, check_retry_budget),
    State.VALIDATE:     make_guardrail(check_transition_allowed, check_raw_data_present),
    State.ENRICH:       make_guardrail(check_transition_allowed, check_validated_data_present),
    State.STORE:        make_guardrail(check_transition_allowed, check_enriched_data_present),
    State.COMPLETE:     make_guardrail(check_transition_allowed),
    State.RETRY:        make_guardrail(check_transition_allowed, check_retry_budget),
    State.HUMAN_REVIEW: make_guardrail(check_transition_allowed),
    State.ERROR:        lambda _: GUARDRAIL_PASS,    # error is always reachable
}


# ── Runner ────────────────────────────────────────────────────────────────────

def run_guardrail(state: PipelineState) -> tuple[PipelineState, GuardrailResult]:
    """
    Run the guardrail for state["proposed_next"].

    Returns the (possibly mutated) state and the GuardrailResult.
    On failure, proposed_next is overwritten with the fallback state so the
    workflow dispatcher can proceed without any extra branching.
    """
    proposed = State(state["proposed_next"])
    guard    = GUARDRAILS.get(proposed, lambda _: GUARDRAIL_PASS)
    result   = guard(state)

    if result.passed:
        entry = f"guardrail PASS → {proposed.value}"
        return {**state, "guardrail_ok": True,  "audit_trail": state["audit_trail"] + [entry]}, result

    fallback = (result.fallback or State.ERROR).value
    entry    = f"guardrail FAIL → {proposed.value} ({result.reason}) → fallback {fallback}"
    new_state = {
        **state,
        "proposed_next": fallback,
        "error_message": result.reason,
        "guardrail_ok":  False,
        "audit_trail":   state["audit_trail"] + [entry],
    }
    return new_state, result