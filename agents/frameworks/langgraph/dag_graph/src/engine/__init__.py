"""LangGraph state machine engine (reusable, domain-agnostic)."""

# Generic engine utilities (framework-agnostic, reusable across projects)
from .chains import (
    ainvoke_agent,
    astream_agent,
    chain_field,
    get_agent,
    get_chain,
    get_model,
    invoke_agent,
    make_chain,
    make_llm_agent,
    make_llm_chain,
    render_as_xml,
)
from .engine_graph import (
    END,  # re-export for convenience
    EngineGraph,
    safe_node,
)
from .chat_engine_graph import ChatEngineGraph, TopicDecision
from .chat_engine_session_state import ChatEngineSessionState, new_chat_session_state
from .engine_session_state import EngineSessionState
from .event_ledger import EventLedger, effect_key, event_key
from .json_checkpointer import JsonCheckpointer
from .utils import (
    close_topic_delta,
    content_hash,
    find_tool_call,
    find_tool_message,
    last_ai_content,
    ledger_is_processed_sync,
    ledger_mark_processed_sync,
    log_handler_enter,
    log_handler_exit,
    segment_reset_messages,
    trim_messages,
)
from .sqlite_checkpointing import SqliteCheckpointer

# Note: Domain-specific state machine, handlers, and guardrails are in src/docprocessing/
# This module provides only the generic engine utilities for reuse across projects.

__all__ = [
    # Generic engine (reusable across projects)
    "EngineSessionState",
    "ChatEngineSessionState",
    "new_chat_session_state",
    "EngineGraph",
    "ChatEngineGraph",
    "TopicDecision",
    "safe_node",
    "make_chain",
    "get_chain",
    "get_model",
    "make_llm_agent",
    "get_agent",
    "invoke_agent",
    "ainvoke_agent",
    "astream_agent",
    "make_llm_chain",
    "render_as_xml",
    "chain_field",
    "END",
    # Chat / handler helpers
    "trim_messages",
    "segment_reset_messages",
    "close_topic_delta",
    "ledger_is_processed_sync",
    "ledger_mark_processed_sync",
    "content_hash",
    "log_handler_enter",
    "log_handler_exit",
    "last_ai_content",
    "find_tool_message",
    "find_tool_call",
    # Ledger
    "EventLedger",
    "event_key",
    "effect_key",
    # Checkpointing
    "SqliteCheckpointer",
    "JsonCheckpointer",
]
