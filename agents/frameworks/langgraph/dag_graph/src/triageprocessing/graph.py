"""Triage-and-fix Graph — EngineGraph domain (harness steps 1–8).

No semantic router. No LangGraph interrupt(); the review gate is
AWAIT_APPROVAL with wait_kind=either.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from src.engine.engine_graph import EngineGraph
from src.engine.event_ledger import EventLedger
from src.engine.json_checkpointer import JsonCheckpointer

from .guardrails import guardrails
from .handlers import handler_map, set_worker
from .session_state import TriageState, new_triage_session_state
from .state_transitions import State, happy_path, terminal_states
from .worker import FakeWorker, Worker


class Graph(EngineGraph):
    state_enum = State
    terminal_states = terminal_states
    handler_map = handler_map

    def __init__(self, artifacts_dir: str = ".triage_artifacts") -> None:
        super().__init__()
        self.artifacts_dir = artifacts_dir

    def _build_routing_table(self) -> dict[Any, Any]:
        return happy_path

    def _get_current_state(self, state: dict[str, Any]) -> State:
        return State(state.get("current_state", State.INIT.value))

    def _get_proposed_state(self, state: dict[str, Any]) -> State:
        return State(state.get("proposed_next", State.INTAKE.value))

    def _get_guardrails(self) -> dict[Any, Callable]:
        return guardrails

    def _get_allowed_states(self, current_state: State) -> list[str]:
        from .state_transitions import allowed_transitions

        allowed = allowed_transitions.get(current_state, set())
        return [s.value if hasattr(s, "value") else s for s in allowed]

    def _new_session_state(self) -> dict[str, Any]:
        return new_triage_session_state(artifacts_dir=self.artifacts_dir)


def build_graph(
    sessions_dir: str = ".triage_sessions",
    artifacts_dir: str = ".triage_artifacts",
    worker: Optional[Worker] = None,
) -> Graph:
    set_worker(worker or FakeWorker())
    checkpointer = JsonCheckpointer(sessions_dir=sessions_dir)
    graph = Graph(artifacts_dir=artifacts_dir)
    graph.compiled_graph = graph.build_graph(TriageState, checkpointer=checkpointer)
    graph._ledger = EventLedger(ledger_dir=f"{sessions_dir}_ledger")
    return graph
