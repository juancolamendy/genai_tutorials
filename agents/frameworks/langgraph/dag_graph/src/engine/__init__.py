"""LangGraph state machine engine (reusable, domain-agnostic)."""

# Generic engine utilities (framework-agnostic, reusable across projects)
from .graph import (
    StateMachineGraph,
    serialize_session_state,
    deserialize_to_session_state,
    safe_node,
    END,  # re-export for convenience
)
from .checkpointing import (
    SqliteCheckpointer,
    init_checkpointer,
    get_checkpointer,
)
from .chain import (
    make_chain,
    get_chain,
    make_llm_chain,
    render_as_xml,
)
from .session import (
    append_turn,
    build_history_prompt,
    get_execution_context,
    init_session_defaults,
)

# Note: Domain-specific state machine, handlers, and guardrails are in src/workflow/
# This module provides only the generic engine utilities for reuse across projects.

__all__ = [
    # Generic engine (reusable across projects)
    "StateMachineGraph",
    "serialize_session_state",
    "deserialize_to_session_state",
    "safe_node",
    "make_chain",
    "get_chain",
    "make_llm_chain",
    "render_as_xml",
    "append_turn",
    "build_history_prompt",
    "get_execution_context",
    "init_session_defaults",
    "END",
    # Checkpointing
    "SqliteCheckpointer",
    "init_checkpointer",
    "get_checkpointer",
]
