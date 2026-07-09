from typing import Any, Dict, Optional

from src.engine.engine_session_state import EngineSessionState, new_engine_session_state

from .state_transitions import State

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STATE
# ─────────────────────────────────────────────────────────────────────────────

# structures
class SessionState(EngineSessionState):
    """Document processing pipeline state.

    Inherits common control plane and multi-turn fields from EngineSessionState.
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

    supporting_docs: Optional[list[Dict[str, Any]]]
    """Supporting documents uploaded by user. Set by upload_documents handler."""

# functions
# ── Constructor ───────────────────────────────────────────────────────────────
def new_session_state() -> SessionState:
    """Return a fresh SessionState ready to start at INIT.

    Args:
        document_id: Document identifier
        timeout_seconds: Max execution time (default 300s = 5 min)

    Returns:
        Fresh SessionState with all fields initialized
    """

    engine_session_state = new_engine_session_state()
    session_state: SessionState = {
        **engine_session_state,
        "current_state": State.INIT.value,
        "proposed_next": State.FETCH.value,
        "document_id": "",
        "raw_data": None,
        "validated_data": None,
        "enriched_data": None,
        "supporting_docs": None,
    }
    return session_state
