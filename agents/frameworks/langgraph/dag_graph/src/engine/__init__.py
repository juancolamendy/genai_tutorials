"""LangGraph state machine engine (reusable, domain-agnostic)."""

# Generic engine utilities (framework-agnostic, reusable across projects)
from .chain import (
    get_chain,
    make_chain,
    make_llm_chain,
    render_as_xml,
)
from .engine_session_state import EngineSessionState
from .engine_graph import (
    END,  # re-export for convenience
    EngineGraph,
    safe_node,
)
from .json_checkpointer import JsonCheckpointer
from .sqlite_checkpointing import SqliteCheckpointer

# Note: Domain-specific state machine, handlers, and guardrails are in src/workflow/
# This module provides only the generic engine utilities for reuse across projects.

__all__ = [
    # Generic engine (reusable across projects)
    "EngineSessionState",
    "EngineGraph",
    "safe_node",
    "make_chain",
    "get_chain",
    "make_llm_chain",
    "render_as_xml",
    "END",
    # Checkpointing
    "SqliteCheckpointer",
    "JsonCheckpointer",
]
