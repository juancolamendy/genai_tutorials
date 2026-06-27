"""Domain-specific LangGraph configuration for document processing.

DocumentPipelineGraph inherits from StateMachineGraph and defines:
  • State machine (states + transitions)
  • Routing table (happy path)
  • Guardrails
  • Handlers
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from src.engine.graph import StateMachineGraph

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

from .guardrails import GUARDRAILS
from .handlers import (
    handle_complete,
    handle_enrich,
    handle_error,
    handle_fetch,
    handle_human_review,
    handle_retry,
    handle_store,
    handle_validate,
)
from .pipeline_state import PipelineState
from .state_machine import State, HAPPY_PATH, TERMINAL_STATES

# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN-SPECIFIC CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

HANDLER_MAP = {
    State.FETCH: handle_fetch,
    State.VALIDATE: handle_validate,
    State.ENRICH: handle_enrich,
    State.STORE: handle_store,
    State.RETRY: handle_retry,
    State.HUMAN_REVIEW: handle_human_review,
    State.COMPLETE: handle_complete,
    State.ERROR: handle_error,
}


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT PIPELINE GRAPH (inherits from StateMachineGraph)
# ─────────────────────────────────────────────────────────────────────────────

class DocumentPipelineGraph(StateMachineGraph):
    """Document processing pipeline using generic StateMachineGraph.

    Implements the production pattern:
      Router → Guardrail → Handler → (loop or end)

    All generic logic (router, guardrail, graph building) is inherited from
    StateMachineGraph. This class only defines domain-specific configuration.

    Can optionally use semantic routing (LLM-powered) via set_semantic_router().
    """

    # Domain-specific configuration
    _STATE_KEYS = (
        "current_state",
        "proposed_next",
        "retry_count",
        "error_message",
        "audit_trail",
        "document_id",
        "raw_data",
        "validated_data",
        "enriched_data",
    )
    _STATE_ENUM = State
    _TERMINAL_STATES = TERMINAL_STATES
    HANDLER_MAP = HANDLER_MAP

    def __init__(self, semantic_router: Optional[Any] = None):
        """Initialize graph with optional semantic router.

        Args:
            semantic_router: Optional semantic router instance (e.g., DocPipelineRouter)
        """
        self.semantic_router = semantic_router

    def _build_routing_table(self) -> dict[Any, Any]:
        """Return happy path routing table."""
        return HAPPY_PATH

    def _get_current_state(self, state: dict[str, Any]) -> State:
        """Extract current state from state dict."""
        return State(state.get("current_state", State.INIT.value))

    def _get_proposed_state(self, state: dict[str, Any]) -> State:
        """Extract proposed next state from state dict."""
        return State(state.get("proposed_next", State.FETCH.value))

    def _get_guardrails(self) -> dict[Any, Callable]:
        """Return guardrail registry."""
        return GUARDRAILS

    def set_semantic_router(self, router: Any) -> None:
        """Set the semantic router for LLM-powered routing.

        Args:
            router: Semantic router instance (e.g., DocPipelineRouter, DefaultSemanticRouter)
        """
        self.semantic_router = router


def build_graph(
    checkpointer: Optional[Any] = None,
    semantic_router: Optional[Any] = None,
) -> Any:
    """Build and compile the state machine graph.

    Args:
        checkpointer: Optional BaseCheckpointSaver for checkpointing
        semantic_router: Optional semantic router for LLM-powered routing
                        (e.g., DocPipelineRouter instance)

    Returns:
        Compiled StateGraph ready for invocation
    """
    graph = DocumentPipelineGraph(semantic_router=semantic_router)
    return graph.build_graph(PipelineState, checkpointer=checkpointer)


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTIONS (backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def router_node(state: PipelineState) -> PipelineState:
    """Convenience function wrapping DocumentPipelineGraph._router_node."""
    graph = DocumentPipelineGraph()
    return graph._router_node(state)


def guardrail_node(state: PipelineState) -> PipelineState:
    """Convenience function wrapping DocumentPipelineGraph._guardrail_node."""
    graph = DocumentPipelineGraph()
    return graph._guardrail_node(state)


def guardrail_router(state: PipelineState) -> str:
    """Convenience function wrapping DocumentPipelineGraph._guardrail_router."""
    graph = DocumentPipelineGraph()
    return graph._guardrail_router(state)

