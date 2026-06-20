"""Domain-specific LangGraph configuration for document processing.

DocumentPipelineGraph inherits from StateMachineGraph and defines:
  • State machine (states + transitions)
  • Routing table (happy path)
  • Guardrails
  • Handlers
"""

from __future__ import annotations

from typing import Any, Callable

from src.engine.graph import StateMachineGraph

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
from .state_machine import State

# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN-SPECIFIC CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

HAPPY_PATH: dict[State, State] = {
    State.INIT: State.FETCH,
    State.FETCH: State.VALIDATE,
    State.VALIDATE: State.ENRICH,
    State.ENRICH: State.STORE,
    State.STORE: State.COMPLETE,
    State.RETRY: State.FETCH,
    State.HUMAN_REVIEW: State.ENRICH,
}

TERMINAL_STATES = {State.COMPLETE, State.ERROR}

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


def build_graph() -> Any:
    """Build and compile the state machine graph."""
    graph = DocumentPipelineGraph()
    return graph.build_graph(PipelineState)


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

