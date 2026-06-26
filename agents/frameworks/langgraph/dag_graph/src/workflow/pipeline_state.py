from typing import Any, Dict, Optional, TypedDict

from .state_machine import State

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STATE
# ─────────────────────────────────────────────────────────────────────────────

# structures
class PipelineState(TypedDict):
    """Central state dict for pipeline execution.

    Fields organized into four categories:
      • Control Plane: routing and error tracking
      • Business Payload: document data
      • Multi-turn: conversation history and session tracking
    """

    # ─ Control Plane ─────────────────────────────────────────────────────
    current_state: str  # mirrors the active graph node
    proposed_next: str  # router's suggestion
    retry_count: int
    error_message: Optional[str]
    audit_trail: list[str]  # append-only log of every step
    guardrail_ok: bool

    # ─ Business Payload ──────────────────────────────────────────────────
    document_id: str
    raw_data: Optional[Dict[str, Any]]
    validated_data: Optional[Dict[str, Any]]
    enriched_data: Optional[Dict[str, Any]]

    # ─ Multi-turn Support ────────────────────────────────────────────────
    turn_input: Optional[str]  # Current turn's user input (escaped)
    turn_number: int  # Turn counter (0, 1, 2, ...)
    conversation_history: list[Dict[str, Any]]  # Turns: {role, content, semantic_context, state}
    max_history_turns: int  # Max turns to keep (default 10)
    router_timeout_sec: float  # Timeout for semantic router (default 10.0)
    user_id: Optional[str]  # Caller identity (for audit)
    session_id: Optional[str]  # Multi-turn session ID
    semantic_context: Dict[str, Any]  # {entities, intents} from router
    router_confidence: float  # [0.0, 1.0]

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
    return PipelineState(
        # Control Plane
        current_state=State.INIT.value,
        proposed_next=State.FETCH.value,
        retry_count=0,
        error_message=None,
        guardrail_ok=True,
        audit_trail=[f"init  doc_id={document_id}"],
        # Business Payload
        document_id=document_id,
        raw_data=None,
        validated_data=None,
        enriched_data=None,
        # Multi-turn Support
        turn_input=None,
        turn_number=0,
        conversation_history=[],
        max_history_turns=10,
        router_timeout_sec=10.0,
        user_id=None,
        session_id=None,
        semantic_context={},
        router_confidence=0.0,
    )
