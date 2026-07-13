"""Sentry-error triage-and-fix harness domain.

Exercises EngineGraph event gates (aemit_event / wait_kind=either) instead of
LangGraph interrupt(), following the onboarding package layout.
"""

from src.triageprocessing.graph import Graph, build_graph
from src.triageprocessing.session_state import TriageState, new_triage_session_state
from src.triageprocessing.state_transitions import (
    State,
    allowed_transitions,
    happy_path,
    is_transition_allowed,
    terminal_states,
)

__all__ = [
    "State",
    "TriageState",
    "new_triage_session_state",
    "happy_path",
    "terminal_states",
    "allowed_transitions",
    "is_transition_allowed",
    "Graph",
    "build_graph",
]
