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
      • current_event_source: Which kind of event resumed this turn
        ("human" or "system"); absent until an event-gate call stamps it
      • current_event_type: The specific event type for this turn (e.g.
        "message", "document_signed", "timeout_escalation")
      • output_messages: Per-turn messages a handler wants said to a human,
        if any (reducer: operator.add)

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

    handler_status: Literal["ok", "error"]
    """Per-handler business outcome for guardrail diversion. Fallible handlers
    stamp "error" on caught business failures (e.g. LLM/tool exceptions) and
    "ok" on success. A domain guardrail (check_handler_status) reading this
    can divert to ERROR — do not use session status for that; dispatch resets
    status to "ok" on every non-ERROR handler."""

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

    # ─ Event Extension ───────────────────────────────────────────────────
    current_event_source: Literal["human", "system"]
    """Which kind of event resumed this turn. Stamped fresh by aemit_event's
    human/system branches on every turn — never carried over from a prior
    turn. Absent on turns that never went through aemit_event (e.g. every
    existing docprocessing call); read via
    state.get("current_event_source", "human") so those default to
    human-sourced."""

    current_event_type: str
    """The specific event type for this turn (e.g. "message",
    "document_signed", "timeout_escalation"). Stamped alongside
    current_event_source. Read via state.get("current_event_type",
    "message") when absent."""

    output_messages: Annotated[list[str], operator.add]
    """Per-turn messages a handler wants said to a human, if any.
    Reducer-backed like audit_trail: handlers return only the new
    entry/entries, never the accumulated list. Empty by default — the
    engine's message convention falls back to a generic "Transitioned to X"
    message for human-sourced turns when this is empty, and stays silent
    entirely for system-sourced turns with nothing to say."""


def new_engine_session_state() -> EngineSessionState:
    """Create fresh session state.

    current_event_source/current_event_type are deliberately omitted here
    (left absent, not pre-seeded) — they are stamped fresh by aemit_event's
    branches on every turn that goes through it, and pre-seeding a value
    here would be indistinguishable from a stale value carried over from a
    turn that never actually set it.
    """
    return EngineSessionState(
        current_state="init",
        proposed_next="init",
        retry_count=0,
        error_message=None,
        error_type=None,
        status="ok",
        handler_status="ok",
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
        output_messages=[],
    )


class ChatEngineSessionState(EngineSessionState):
    """Base session state for hub + sticky-topic conversational workflows.

    Linear pipelines keep using EngineSessionState. Chatbot domains
    (e.g. hrhelpdesk) inherit this and add business payload fields.

    Fields:
      • schema_version: Durable schema revision for migration shims
      • active_topic: Sticky lane id while a topic is open; None at hub
      • topic_started_at: ISO timestamp when the sticky topic opened (sweep)
      • topic_data: Lean per-lane scratch (slots, etc.) — not document blobs
    """

    schema_version: int
    active_topic: Optional[str]
    topic_started_at: Optional[str]
    topic_data: Dict[str, Any]


def new_chat_session_state(
    *,
    current_state: str = "idle",
    proposed_next: str = "route",
) -> ChatEngineSessionState:
    """Create fresh chat session state with sticky-lane fields initialized."""
    base = new_engine_session_state()
    chat_state: ChatEngineSessionState = {
        **base,
        "current_state": current_state,
        "proposed_next": proposed_next,
        "schema_version": 1,
        "active_topic": None,
        "topic_started_at": None,
        "topic_data": {},
    }
    return chat_state
