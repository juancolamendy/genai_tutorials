from typing import TypedDict, Optional, Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STATE
# ─────────────────────────────────────────────────────────────────────────────

# structures
class PipelineState(TypedDict):
    """Central state dict for pipeline execution.

    Fields organized into three categories:
      • Control Plane: routing and error tracking
      • Business Payload: document data
    """

    # ─ Control Plane ─────────────────────────────────────────────────────
    current_state: str  # mirrors the active graph node
    proposed_next: str  # router's suggestion
    retry_count: int
    error_message: Optional[str]
    audit_trail: list[str]  # append-only log of every step

    # ─ Business Payload ──────────────────────────────────────────────────
    document_id: str
    raw_data: Optional[Dict[str, Any]]
    validated_data: Optional[Dict[str, Any]]
    enriched_data: Optional[Dict[str, Any]]

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
