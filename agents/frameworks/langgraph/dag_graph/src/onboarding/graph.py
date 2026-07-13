"""Domain-specific LangGraph configuration for onboarding.

Graph inherits from EngineGraph and defines:
  • State machine (states + transitions)
  • Routing table (happy path)
  • Guardrails
  • Handlers

No semantic router: onboarding is code-routed only (design spec §2.3 — "No
LLM output transitions state"), matching the router-bypass principle this
whole design is built around.
"""

from __future__ import annotations

from typing import Any, Callable

from src.engine.engine_graph import EngineGraph
from src.engine.json_checkpointer import JsonCheckpointer

from .guardrails import guardrails
from .handlers import handler_map
from .session_state import OnboardingState, new_onboarding_session_state
from .state_transitions import (
    State,
    happy_path,
    terminal_states,
)

# ─────────────────────────────────────────────────────────────────────────────
# ONBOARDING GRAPH (inherits from EngineGraph)
# ─────────────────────────────────────────────────────────────────────────────

class Graph(EngineGraph):
    """New-hire onboarding pipeline using the generic EngineGraph.

    Implements the production pattern:
      Router → Guardrail → Handler → (loop or end)

    All generic logic (router, guardrail, graph building, ainvoke,
    aemit_event, arun_to_completion) is inherited from EngineGraph. This
    class only defines domain-specific configuration.
    """

    # Domain-specific configuration
    state_enum = State
    terminal_states = terminal_states
    handler_map = handler_map

    def _build_routing_table(self) -> dict[Any, Any]:
        """Return happy path routing table."""
        return happy_path

    def _get_current_state(self, state: dict[str, Any]) -> State:
        """Extract current state from state dict."""
        return State(state.get("current_state", State.INIT.value))

    def _get_proposed_state(self, state: dict[str, Any]) -> State:
        """Extract proposed next state from state dict."""
        return State(state.get("proposed_next", State.COLLECT.value))

    def _get_guardrails(self) -> dict[Any, Callable]:
        """Return guardrail registry."""
        return guardrails

    def _get_allowed_states(self, current_state: State) -> list[str]:
        """Get allowed next states for current state from allowed_transitions."""
        from .state_transitions import allowed_transitions

        allowed = allowed_transitions.get(current_state, set())
        return [s.value if hasattr(s, "value") else s for s in allowed]

    def _new_session_state(self) -> dict[str, Any]:
        """Create fresh session state."""
        return new_onboarding_session_state()


def build_graph(sessions_dir: str = ".onboarding_sessions") -> Graph:
    """Build and compile the onboarding state machine graph.

    Args:
        sessions_dir: Directory for checkpoint storage

    Returns:
        Graph with compiled_graph set and ready for invoke()/ainvoke()/
        aemit_event()/arun_to_completion()

    Note: no semantic_router option here — onboarding is code-routed only.
    EventLedger is colocated with sessions_dir so one-shot CLI processes
    sharing --sessions-dir also share dedupe state (not CWD .event_ledger).
    """
    from src.engine.event_ledger import EventLedger

    checkpointer = JsonCheckpointer(sessions_dir=sessions_dir)

    graph = Graph()
    graph.compiled_graph = graph.build_graph(OnboardingState, checkpointer=checkpointer)
    graph._ledger = EventLedger(ledger_dir=f"{sessions_dir}_ledger")
    return graph
