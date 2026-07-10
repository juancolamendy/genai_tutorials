"""Generic state machine state definition for LangGraph.

EngineSessionState defines the common control plane and multi-turn support fields
used by all state machine workflows. Domain-specific states (e.g., SessionState)
inherit from this and add their own business payload fields.
"""

import operator
import time
from typing import Annotated, Any, Dict, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class EngineSessionState(TypedDict, total=False):
    """Generic state machine state with control plane and multi-turn support.

    This TypedDict provides the foundation for all state machine workflows.
    It includes:
      • Control plane: state tracking, routing, error handling
      • Multi-turn support: message history, user context
      • Semantic routing: context and confidence

    Domain-specific states (e.g., SessionState for documents) inherit from
    this class and add business-specific payload fields.

    Fields:
      • current_state: Current state in the state machine
      • proposed_next: Router's suggestion for next state
      • retry_count: Number of retries attempted
      • error_message: Error description if status='error'
      • error_type: Exception class name if status='error'
      • status: 'ok' or 'error' — set centrally (safe_node, handler dispatch),
        never by individual handlers
      • guardrail_ok: Guardrail validation result
      • fallback_depth: Consecutive guardrail-fallback count (cascade detection)
      • audit_trail: Append-only log of state transitions (reducer: operator.add)
      • input_message: Current turn's user input (escaped)
      • turn_number: Multi-turn counter
      • messages: Accumulated conversation messages (reducer: add_messages)
      • user_id: Caller identity (for audit)
      • session_id: Multi-turn session ID
      • semantic_context: Extracted entities and intents
      • router_confidence: Confidence of router decision
      • router_reasoning: Optional explanation from semantic router
      • started_at: Workflow start timestamp
      • timeout_seconds: Max execution time for entire workflow

    Note: max_history_turns and router_timeout_sec are NOT here — they're
    run configuration, not session data. max_history_turns lives on
    EngineGraph (doesn't vary per-session); router_timeout_sec is threaded
    per-invocation via LangGraph's config["configurable"], not persisted.
    """

    # ─ Control Plane ─────────────────────────────────────────────────────
    current_state: str
    """Current state node (e.g., 'init', 'fetch', 'validate', 'complete', 'error')."""

    proposed_next: str
    """Router's proposed next state. Set by router node, used by guardrail/dispatcher."""

    retry_count: int
    """Number of retries attempted for current operation. Incremented by retry handler."""

    error_message: Optional[str]
    """Error description if status='error'. None otherwise."""

    error_type: Optional[str]
    """Exception class name if status='error' (set by safe_node). None otherwise."""

    status: Literal["ok", "error"]
    """Overall session health. Set centrally — by safe_node on uncaught exceptions,
    and by handler dispatch when the state entered is the domain's ERROR state.
    Individual handlers never set this themselves."""

    guardrail_ok: bool
    """Guardrail validation result. True if proposed_next passed guardrails."""

    fallback_depth: int
    """Consecutive guardrail-fallback count. Incremented on fallback, reset on pass."""

    audit_trail: Annotated[list[str], operator.add]
    """Append-only log of every step. Reducer-backed: nodes return only the new
    entry/entries (e.g. ["fetch OK"]), LangGraph appends them automatically."""

    # ─ Multi-turn Support ────────────────────────────────────────────────
    input_message: Optional[str]
    """Current turn's user input (already escaped for LLM safety)."""

    turn_number: int
    """Turn counter. 0 = initial state, 1+ = multi-turn turns."""

    messages: Annotated[list[BaseMessage], add_messages]
    """Accumulated conversation messages across turns. Reducer-backed: nodes
    return only new messages, LangGraph appends/merges them automatically.

    Per-turn bookkeeping (turn_number, state, semantic_context) is carried in
    each message's additional_kwargs, since BaseMessage has no room for
    app-specific fields otherwise.
    """

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


def new_engine_session_state() -> EngineSessionState:
    """Create fresh session state."""
    return EngineSessionState(
        current_state="init",
        proposed_next="init",
        retry_count=0,
        error_message=None,
        error_type=None,
        status="ok",
        guardrail_ok=True,
        fallback_depth=0,
        audit_trail=["init session state"],
        input_message=None,
        turn_number=0,
        messages=[],
        user_id="",
        session_id="",
        semantic_context={},
        router_confidence=0.0,
        router_reasoning=None,
        started_at=time.time(),
        timeout_seconds=300.0,
    )
