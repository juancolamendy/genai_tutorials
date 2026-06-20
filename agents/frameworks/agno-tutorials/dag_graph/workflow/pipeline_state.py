"""
pipeline_state.py
────────────────────────────────────────────────────────────────────────────
PipelineState — the single typed dict that flows through every step.

Split into two logical planes:
  • Control plane  — routing, guardrails, retries, audit log
  • Business plane — document payload at each processing stage

Also exports helpers for common mutations so callers never format audit
entries by hand.
"""

from typing import Any, Optional, TypedDict

from .state_machine import State, TERMINAL_STATES


# structures
class PipelineState(TypedDict):
    # ── Control plane ─────────────────────────────────────────────────────────
    current_state:  str           # current active state (mirrors the node just executed)
    proposed_next:  str           # router's candidate for the next state
    retry_count:    int
    error_message:  Optional[str]
    guardrail_ok:   bool          # True after guardrail passed; False after fallback override
    audit_trail:    list[str]     # append-only chronological log

    # ── Business plane ────────────────────────────────────────────────────────
    document_id:    str
    raw_data:       Optional[dict[str, Any]]    # set by FETCH
    validated_data: Optional[dict[str, Any]]    # set by VALIDATE / HUMAN_REVIEW
    enriched_data:  Optional[dict[str, Any]]    # set by ENRICH


# functions
# ── Constructor ───────────────────────────────────────────────────────────────
def new_pipeline(document_id: str) -> PipelineState:
    """Return a fresh PipelineState ready to start at INIT."""
    return PipelineState(
        current_state  = State.INIT.value,
        proposed_next  = State.FETCH.value,
        retry_count    = 0,
        error_message  = None,
        guardrail_ok   = True,
        audit_trail    = [f"init  doc_id={document_id}"],
        document_id    = document_id,
        raw_data       = None,
        validated_data = None,
        enriched_data  = None,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────
def audit(state: PipelineState, entry: str) -> PipelineState:
    """Return state with `entry` appended to audit_trail (immutable-style update)."""
    return {**state, "audit_trail": state["audit_trail"] + [entry]}


def is_terminal(state: PipelineState) -> bool:
    """True when the pipeline has reached a terminal state."""
    return State(state["current_state"]) in TERMINAL_STATES


def pretty_audit(state: PipelineState) -> str:
    """Format the audit trail for human-readable display."""
    lines = [f"\n  Audit Trail ({len(state['audit_trail'])} entries):"]
    for entry in state["audit_trail"]:
        lines.append(f"    • {entry}")
    lines.append(f"  Final State: {state['current_state'].upper()}")
    if state.get("retry_count"):
        lines.append(f"  Retries: {state['retry_count']}")
    return "\n".join(lines)
