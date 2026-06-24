"""
workflow/pipeline_state.py
────────────────────────────────────────────────────────────────────────────
PipelineState — document processing state combining control and business planes.

Inherits EngineState (control plane) and adds business-specific fields:
  • document_id: Document being processed
  • raw_data: Document as fetched
  • validated_data: Document after validation
  • enriched_data: Document after enrichment

Also exports helpers for common mutations.
"""

from typing import Any, Optional

from engine.engine_state import EngineState, init_engine_state
from .state_machine import State, TERMINAL_STATES


class PipelineState(EngineState):
    """
    Full pipeline state = control plane (EngineState) + business payload.

    Adds document-specific fields to the base EngineState.
    """

    document_id: str
    raw_data: Optional[dict[str, Any]]       # set by FETCH
    validated_data: Optional[dict[str, Any]]  # set by VALIDATE / HUMAN_REVIEW
    enriched_data: Optional[dict[str, Any]]   # set by ENRICH


def new_pipeline(document_id: str) -> PipelineState:
    """Return a fresh PipelineState ready to start at INIT."""
    base = init_engine_state()
    return {
        **base,
        "current_state": State.INIT.value,
        "proposed_next": State.FETCH.value,
        "audit_trail": [f"init  doc_id={document_id}"],
        "document_id": document_id,
        "raw_data": None,
        "validated_data": None,
        "enriched_data": None,
    }


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
