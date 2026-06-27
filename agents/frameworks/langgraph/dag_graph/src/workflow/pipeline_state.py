from typing import Any, Dict, Optional

from src.engine.engine_state import EngineState
from .state_machine import State

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STATE
# ─────────────────────────────────────────────────────────────────────────────

# structures
class PipelineState(EngineState):
    """Document processing pipeline state.

    Inherits common control plane and multi-turn fields from EngineState.
    Adds business-specific payload for document processing.

    Business Payload Fields:
      • document_id: Unique document identifier
      • raw_data: Raw document content from fetch
      • validated_data: Validated schema and content
      • enriched_data: Enriched with metadata, tags, summary
    """

    # ─ Business Payload (document-specific) ───────────────────────────────
    document_id: str
    """Unique document identifier being processed."""

    raw_data: Optional[Dict[str, Any]]
    """Raw document content fetched from source. Set by fetch handler."""

    validated_data: Optional[Dict[str, Any]]
    """Validated document content. Set by validate handler."""

    enriched_data: Optional[Dict[str, Any]]
    """Enriched document with metadata. Set by enrich handler."""

# functions
# ── Constructor ───────────────────────────────────────────────────────────────
def new_pipeline(document_id: str, timeout_seconds: float = 300.0) -> PipelineState:
    """Return a fresh PipelineState ready to start at INIT.

    Args:
        document_id: Document identifier
        timeout_seconds: Max execution time (default 300s = 5 min)

    Returns:
        Fresh PipelineState with all fields initialized
    """
    import time

    return PipelineState(
        # Control Plane (EngineState)
        current_state=State.INIT.value,
        proposed_next=State.FETCH.value,
        retry_count=0,
        error_message=None,
        guardrail_ok=True,
        audit_trail=[f"init  doc_id={document_id}"],
        # Multi-turn Support (EngineState)
        turn_input=None,
        turn_number=0,
        conversation_history=[],
        max_history_turns=10,
        router_timeout_sec=10.0,
        user_id=None,
        session_id=None,
        semantic_context={},
        router_confidence=0.0,
        # Checkpointing Support (EngineState)
        started_at=time.time(),
        timeout_seconds=timeout_seconds,
        # Business Payload (PipelineState)
        document_id=document_id,
        raw_data=None,
        validated_data=None,
        enriched_data=None,
    )
