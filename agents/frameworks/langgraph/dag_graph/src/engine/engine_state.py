"""Generic state machine state definition for LangGraph.

EngineState defines the common control plane and multi-turn support fields
used by all state machine workflows. Domain-specific states (e.g., PipelineState)
inherit from this and add their own business payload fields.
"""

from typing import Any, Dict, Optional
from typing_extensions import TypedDict


class EngineState(TypedDict, total=False):
    """Generic state machine state with control plane and multi-turn support.

    This TypedDict provides the foundation for all state machine workflows.
    It includes:
      • Control plane: state tracking, routing, error handling
      • Multi-turn support: conversation history, user context
      • Semantic routing: context and confidence

    Domain-specific states (e.g., PipelineState for documents) inherit from
    this class and add business-specific payload fields.

    Required Fields:
      • current_state: Current state in the state machine
      • proposed_next: Router's suggestion for next state
      • retry_count: Number of retries attempted
      • audit_trail: Append-only log of state transitions
      • turn_number: Multi-turn counter
      • conversation_history: Accumulated turns
      • max_history_turns: Max turns to retain
      • router_timeout_sec: Timeout for semantic router
      • semantic_context: Extracted entities and intents
      • router_confidence: Confidence of router decision

    Optional Fields:
      • error_message: Error description if state=error
      • guardrail_ok: Guardrail validation result
      • turn_input: Current turn's user input (escaped)
      • user_id: Caller identity (for audit)
      • session_id: Multi-turn session ID
    """

    # ─ Control Plane ─────────────────────────────────────────────────────
    current_state: str
    """Current state node (e.g., 'init', 'fetch', 'validate', 'complete', 'error')."""

    proposed_next: str
    """Router's proposed next state. Set by router node, used by guardrail/dispatcher."""

    retry_count: int
    """Number of retries attempted for current operation. Incremented by retry handler."""

    error_message: Optional[str]
    """Error description if state='error'. None otherwise."""

    guardrail_ok: bool
    """Guardrail validation result. True if proposed_next passed guardrails."""

    audit_trail: list[str]
    """Append-only log of every step. One entry per state transition."""

    # ─ Multi-turn Support ────────────────────────────────────────────────
    turn_input: Optional[str]
    """Current turn's user input (already escaped for LLM safety)."""

    turn_number: int
    """Turn counter. 0 = initial state, 1+ = multi-turn turns."""

    conversation_history: list[Dict[str, Any]]
    """Accumulated conversation history across turns.

    Each entry: {
        "role": "user" | "assistant",
        "content": str,
        "turn_number": int,
        "state": str (optional),
        "semantic_context": dict (optional)
    }
    """

    max_history_turns: int
    """Maximum number of turns to retain in conversation_history (default: 10)."""

    router_timeout_sec: float
    """Timeout in seconds for semantic router LLM call (default: 10.0)."""

    user_id: Optional[str]
    """Caller identity. Used for audit trail and session management."""

    session_id: Optional[str]
    """Multi-turn session ID. Groups related turns together."""

    # ─ Semantic Routing ──────────────────────────────────────────────────
    semantic_context: Dict[str, Any]
    """Semantic entities and intents extracted by router.

    Format: {
        "entities": {...},        # Domain entities extracted from input
        "intents": [...]          # User intents identified
    }
    """

    router_confidence: float
    """Confidence of router's proposed_next decision. Range [0.0, 1.0]."""

    router_reasoning: Optional[str]
    """Optional explanation from semantic router for its decision."""

    # ─ Checkpointing Support ─────────────────────────────────────────────
    started_at: Optional[float]
    """Unix timestamp when workflow started. Used for timeout calculation."""

    timeout_seconds: float
    """Maximum execution time for entire workflow (default: 300.0)."""
