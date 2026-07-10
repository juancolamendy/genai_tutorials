"""Domain-specific LangGraph configuration for document processing.

Graph inherits from EngineGraph and defines:
  • State machine (states + transitions)
  • Routing table (happy path)
  • Guardrails
  • Handlers
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from src.docprocessing.router import DocPipelineRouter
from src.engine.engine_graph import EngineGraph
from src.engine.json_checkpointer import JsonCheckpointer

if TYPE_CHECKING:
    pass

from .guardrails import guardrails
from .handlers import handler_map
from .session_state import SessionState, new_session_state
from .state_transitions import (
    State,
    happy_path,
    terminal_states,
)

# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT PIPELINE GRAPH (inherits from EngineGraph)
# ─────────────────────────────────────────────────────────────────────────────

class Graph(EngineGraph):
    """Document processing pipeline using generic EngineGraph.

    Implements the production pattern:
      Router → Guardrail → Handler → (loop or end)

    All generic logic (router, guardrail, graph building) is inherited from
    EngineGraph. This class only defines domain-specific configuration.

    Can optionally use semantic routing (LLM-powered) via set_semantic_router().
    """

    # Domain-specific configuration
    state_enum = State
    terminal_states = terminal_states
    handler_map = handler_map

    def __init__(
        self,
        semantic_router: Optional[Any] = None,
        handler_map: Optional[dict[Any, Callable]] = None,
    ):
        """Initialize graph with optional semantic router.

        Args:
            semantic_router: Optional semantic router instance (e.g., DocPipelineRouter)
        """
        self.semantic_router = semantic_router

    def _build_routing_table(self) -> dict[Any, Any]:
        """Return happy path routing table."""
        return happy_path

    def _get_current_state(self, state: dict[str, Any]) -> State:
        """Extract current state from state dict."""
        return State(state.get("current_state", State.INIT.value))

    def _get_proposed_state(self, state: dict[str, Any]) -> State:
        """Extract proposed next state from state dict."""
        return State(state.get("proposed_next", State.FETCH.value))

    def _get_guardrails(self) -> dict[Any, Callable]:
        """Return guardrail registry."""
        return guardrails

    def _get_allowed_states(self, current_state: State) -> list[str]:
        """Get allowed next states for current state from ALLOWED_TRANSITIONS.

        Args:
            current_state: Current state enum

        Returns:
            List of allowed state strings
        """
        from .state_transitions import allowed_transitions

        # Get allowed states from the state machine's transition table
        allowed = allowed_transitions.get(current_state, set())
        return [s.value if hasattr(s, "value") else s for s in allowed]


    def _new_session_state(self) -> dict[str, Any]:
        """Create fresh session state."""
        return new_session_state()


    def set_semantic_router(self, router: Any) -> None:
        """Set the semantic router for LLM-powered routing.

        Args:
            router: Semantic router instance (e.g., DocPipelineRouter, DefaultSemanticRouter)
        """
        self.semantic_router = router


def build_graph(
    sessions_dir: str = ".doc_sessions",
    semantic_router: Optional[Any] = None,
    use_semantic_routing: bool = False,
) -> Graph:
    """Build and compile the state machine graph.

    Args:
        sessions_dir: Directory for checkpoint storage
        semantic_router: Optional semantic router for LLM-powered routing
                        (e.g., DocPipelineRouter instance)
        use_semantic_routing: Enable LLM-powered semantic routing (default: False).
                             If True and semantic_router is None, uses DocPipelineRouter.
                             If False, uses code-based routing only (faster, more predictable).

    Returns:
        Graph with compiled_graph set and ready for invoke()

    Note:
        Code-based routing is enabled by default. Enable semantic routing for LLM-powered
        state transitions, but note that this requires API calls and may make different
        routing decisions than the code router.
    """
    checkpointer = JsonCheckpointer(sessions_dir=sessions_dir)

    # Only use semantic router if explicitly enabled
    if use_semantic_routing and semantic_router is None:
        semantic_router = DocPipelineRouter()
    elif not use_semantic_routing:
        semantic_router = None

    graph = Graph(semantic_router=semantic_router)
    # Set the compiled graph on the wrapper
    graph.compiled_graph = graph.build_graph(SessionState, checkpointer=checkpointer)
    return graph

