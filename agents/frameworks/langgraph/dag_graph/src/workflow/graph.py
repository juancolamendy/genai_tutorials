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
from src.engine.json_checkpointer import JsonCheckpointer
from src.workflow.router import DocPipelineRouter

if TYPE_CHECKING:
    pass

from .guardrails import GUARDRAILS
from .handlers import HANDLER_MAP
from .pipeline_state import PipelineState
from .state_machine import HAPPY_PATH, TERMINAL_STATES, State

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

    def _get_allowed_states(self, current_state: State) -> list[str]:
        """Get allowed next states for current state from ALLOWED_TRANSITIONS.

        Args:
            current_state: Current state enum

        Returns:
            List of allowed state strings
        """
        from .state_machine import ALLOWED_TRANSITIONS

        # Get allowed states from the state machine's transition table
        allowed = ALLOWED_TRANSITIONS.get(current_state, set())
        return [s.value if hasattr(s, "value") else s for s in allowed]

    def new_pipeline(
        self, entity_id: str, timeout_seconds: float = 300.0
    ) -> PipelineState:
        """Create a fresh pipeline state for document processing.

        Args:
            entity_id: Document identifier
            timeout_seconds: Max execution time (default 300s = 5 min)

        Returns:
            Fresh PipelineState with all fields initialized
        """
        from .pipeline_state import new_pipeline as create_pipeline

        return create_pipeline(entity_id, timeout_seconds)


def build_graph(
    sessions_dir: str = ".doc_sessions",
    semantic_router: Optional[Any] = None,
) -> DocumentPipelineGraph:
    """Build and compile the state machine graph.

    Args:
        sessions_dir: Directory for checkpoint storage
        semantic_router: Optional semantic router for LLM-powered routing
                        (e.g., DocPipelineRouter instance)

    Returns:
        DocumentPipelineGraph with compiled_graph set and ready for invoke_turn()
    """
    checkpointer = JsonCheckpointer(sessions_dir=sessions_dir)
    semantic_router = DocPipelineRouter()
    graph = DocumentPipelineGraph(semantic_router=semantic_router)
    # Set the compiled graph on the wrapper
    graph.compiled_graph = graph.build_graph(PipelineState, checkpointer=checkpointer)
    return graph

