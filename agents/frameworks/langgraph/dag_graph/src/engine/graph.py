"""Generic state machine graph base class for LangGraph.

Provides:
  • StateMachineGraph — base class for state machine workflows
  • State serialization/deserialization helpers
  • Router, guardrail, and handler dispatch patterns
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

from langgraph.graph import END, StateGraph

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

log = logging.getLogger(__name__)

StateT = TypeVar("StateT", bound=dict[str, Any])
StateEnum = TypeVar("StateEnum")


# ─────────────────────────────────────────────────────────────────────────
# ERROR HANDLING & SAFETY
# ─────────────────────────────────────────────────────────────────────────

def safe_node(func: Callable) -> Callable:
    """Wrap node function with error handling.

    Catches exceptions and returns error state without propagating.
    """
    @wraps(func)
    def wrapper(state: dict[str, Any]) -> dict[str, Any]:
        try:
            return func(state)
        except Exception as e:
            log.error(f"Node {func.__name__} failed: {e}", exc_info=True)
            return {
                **state,
                "error_message": str(e),
                "error_type": type(e).__name__,
                "proposed_next": "error",  # Route to error state
            }
    return wrapper


def serialize_session_state(
    session_state: dict[str, Any],
    keys: tuple[str, ...],
    defaults: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Extract specified keys from session_state into a new dict.

    Args:
        session_state: The LangGraph state dict
        keys: Tuple of keys to extract (order matters for positional access)
        defaults: Optional dict of {key: default_value} for missing keys

    Returns:
        A new dict containing only the specified keys with their values
        from session_state (or defaults if missing).
    """
    defaults = defaults or {}
    return {k: session_state.get(k, defaults.get(k)) for k in keys}


def deserialize_to_session_state(
    state_dict: dict[str, Any],
    session_state: dict[str, Any],
    keys: tuple[str, ...],
) -> None:
    """Write state_dict values back into session_state for specified keys.

    Args:
        state_dict: The state dict to read from
        session_state: The LangGraph state dict to write to (in-place)
        keys: Tuple of keys to sync
    """
    for k in keys:
        if k in state_dict:
            session_state[k] = state_dict[k]


class StateMachineGraph:
    """Generic state machine graph base class for LangGraph.

    Implements the production pattern:
      Router → Guardrail → Handler → (loop or end)

    Subclasses must override:
      • _STATE_KEYS tuple (keys to persist from state dict)
      • _STATE_ENUM (enum type for state values)
      • _TERMINAL_STATES (set of terminal state enum values)
      • HANDLER_MAP (dict[StateEnum, Callable])
      • _build_routing_table() → dict[StateEnum, StateEnum]
      • _get_current_state(state) → StateEnum
      • _get_proposed_state(state) → StateEnum
      • _get_guardrails() → dict[StateEnum, GuardrailFn]
    """

    # Subclasses must override these
    _STATE_KEYS: tuple[str, ...] = ()
    _STATE_ENUM: type = None
    _TERMINAL_STATES: set = set()
    HANDLER_MAP: dict[Any, Callable] = {}

    def _build_routing_table(self) -> dict[Any, Any]:
        """Return {current_state: next_state} routing table. Override in subclass."""
        raise NotImplementedError

    def _get_current_state(self, state: dict[str, Any]) -> Any:
        """Extract current state from state dict. Override in subclass."""
        raise NotImplementedError

    def _get_proposed_state(self, state: dict[str, Any]) -> Any:
        """Extract proposed next state from state dict. Override in subclass."""
        raise NotImplementedError

    def _get_guardrails(self) -> dict[Any, Callable]:
        """Return guardrail registry. Override in subclass."""
        return {}

    # ─────────────────────────────────────────────────────────────────────────
    # GENERIC NODES (used by all subclasses)
    # ─────────────────────────────────────────────────────────────────────────

    def _router_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Pure code router: current_state → proposed_next via routing table.

        Args:
            state: State dict with current_state set

        Returns:
            Updated state with proposed_next set
        """
        routing_table = self._build_routing_table()
        current = self._get_current_state(state)

        if current not in routing_table:
            raise ValueError(f"Current state {current} not in routing table")

        proposal = routing_table[current]
        proposal_val = proposal.value if hasattr(proposal, "value") else proposal

        log.info("[ROUTER] %s → proposes %s", current, proposal)

        return {
            **state,
            "proposed_next": proposal_val,
            "audit_trail": state.get("audit_trail", []) + [f"router: {current} → {proposal}"],
        }

    def _guardrail_node(self, state: dict[str, Any]) -> dict[str, Any]:
        """Validate proposed_next; apply fallback if needed.

        Args:
            state: State dict with proposed_next set

        Returns:
            Updated state with guardrail result applied
        """
        proposed = self._get_proposed_state(state)
        guardrails = self._get_guardrails()
        guard = guardrails.get(proposed, lambda _: type("Result", (), {"passed": True})())
        result = guard(state)

        if result.passed:
            log.info("[GUARDRAIL] ✅  %s passed", proposed)
            return {
                **state,
                "audit_trail": state.get("audit_trail", []) + [f"guardrail PASS → {proposed}"],
            }

        fallback_val = (result.fallback or self._STATE_ENUM.ERROR).value
        log.warning(
            "[GUARDRAIL] ❌  %s failed (%s) → fallback: %s",
            proposed,
            result.reason,
            fallback_val,
        )
        return {
            **state,
            "proposed_next": fallback_val,
            "error_message": result.reason,
            "audit_trail": state.get("audit_trail", [])
            + [f"guardrail FAIL → {proposed} ({result.reason}) → fallback {fallback_val}"],
        }

    def _guardrail_router(self, state: dict[str, Any]) -> str:
        """Route to handler based on proposed_next.

        Args:
            state: State dict with proposed_next set

        Returns:
            State name (string) to route to
        """
        return state["proposed_next"]

    # ─────────────────────────────────────────────────────────────────────────
    # GRAPH BUILDING (generic, reusable for all subclasses)
    # ─────────────────────────────────────────────────────────────────────────

    def build_graph(self, state_schema: Any, checkpointer: Optional[Any] = None) -> Any:
        """Build and compile the LangGraph state machine.

        Pattern: Router → Guardrail → Handler → (loop or end)

        Args:
            state_schema: TypedDict or dict defining the state structure
            checkpointer: Optional BaseCheckpointSaver for checkpointing

        Returns:
            Compiled StateGraph ready for invocation
        """
        g = StateGraph(state_schema)

        # Add nodes with error handling
        g.add_node("router", safe_node(self._router_node))
        g.add_node("guardrail", safe_node(self._guardrail_node))

        # Add handler nodes with error handling
        for state_enum, handler_fn in self.HANDLER_MAP.items():
            g.add_node(state_enum.value, safe_node(handler_fn))

        # Entry point
        g.set_entry_point("router")

        # Edges: router → guardrail (always)
        g.add_edge("router", "guardrail")

        # Edges: guardrail → handler (conditional: based on proposed_next)
        g.add_conditional_edges(
            "guardrail",
            self._guardrail_router,
            {state.value: state.value for state in self.HANDLER_MAP.keys()},
        )

        # Edges: handlers → router (loop for non-terminal) or END (for terminal)
        for state_enum in self.HANDLER_MAP.keys():
            if state_enum in self._TERMINAL_STATES:
                g.add_edge(state_enum.value, END)
            else:
                g.add_edge(state_enum.value, "router")

        # Compile with optional checkpointer
        return g.compile(checkpointer=checkpointer) if checkpointer else g.compile()
