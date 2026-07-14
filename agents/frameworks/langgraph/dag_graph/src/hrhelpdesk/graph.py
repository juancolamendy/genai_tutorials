"""Domain-specific LangGraph configuration for the HR helpdesk chatbot."""

from __future__ import annotations

from typing import Any, Callable

from src.engine.engine_graph import EngineGraph
from src.engine.handler_registry import get_handler_metadata
from src.engine.json_checkpointer import JsonCheckpointer

from .guardrails import guardrails
from .handlers import handler_map, set_ledger
from .session_state import HelpdeskState, new_helpdesk_session_state
from .state_transitions import State, happy_path, terminal_states


class Graph(EngineGraph):
    """HR helpdesk hub + sticky-topic workflow."""

    state_enum = State
    terminal_states = terminal_states
    handler_map = handler_map

    def _build_routing_table(self) -> dict[Any, Any]:
        return happy_path

    def _get_current_state(self, state: dict[str, Any]) -> State:
        return State(state.get("current_state", State.IDLE.value))

    def _get_proposed_state(self, state: dict[str, Any]) -> State:
        return State(state.get("proposed_next", State.IDLE.value))

    def _get_guardrails(self) -> dict[Any, Callable]:
        return guardrails

    def _get_allowed_states(self, current_state: State) -> list[str]:
        from .state_transitions import allowed_transitions

        allowed = allowed_transitions.get(current_state, set())
        return [s.value for s in allowed]

    def _new_session_state(self) -> dict[str, Any]:
        return new_helpdesk_session_state()

    def _resolve_proposed_next(
        self, state: dict[str, Any], config: Any = None
    ) -> dict[str, Any]:
        """Hub fan-out from typed fields — never from message text."""
        current = self._get_current_state(state)

        if state.get("current_event_source") == "system":
            if state.get("current_state") != State.NOTIFY_USER.value:
                proposal = State.NOTIFY_USER
                return {
                    "proposed_next": proposal.value,
                    "audit_trail": [f"router: system → {proposal.value}"],
                }
            proposal = State.IDLE
            return {
                "proposed_next": proposal.value,
                "audit_trail": [f"router: post-notify → {proposal.value}"],
            }

        if state.get("pending_clarify"):
            proposal = State.HUB_CLARIFY
            return {
                "proposed_next": proposal.value,
                "audit_trail": [f"router: clarify → {proposal.value}"],
            }

        active = state.get("active_topic")
        topic_map = {
            "faq": State.TOPIC_FAQ,
            "escalate": State.TOPIC_ESCALATE,
            "booking": State.TOPIC_BOOKING,
        }
        if active in topic_map:
            proposal = topic_map[active]
            return {
                "proposed_next": proposal.value,
                "audit_trail": [f"router: topic {active} → {proposal.value}"],
            }

        proposal = State.IDLE
        return {
            "proposed_next": proposal.value,
            "audit_trail": [f"router: hub park {current} → {proposal.value}"],
        }

    def _is_system_event_legal(
        self,
        state: dict[str, Any],
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> bool:
        payload = payload or {}

        if event_type == "ticket_resolved":
            ticket_id = payload.get("ticket_id")
            return bool(ticket_id and ticket_id in (state.get("open_tickets") or []))

        if event_type == "booking_cancelled_by_system":
            booking_id = payload.get("booking_id")
            bookings = state.get("bookings") or []
            return any(
                b.get("booking_id") == booking_id and b.get("status") != "cancelled"
                for b in bookings
            )

        if event_type == "topic_timeout":
            return state.get("active_topic") == "booking"

        current = state.get("current_state", State.IDLE.value)
        meta = get_handler_metadata(current)
        expected = (meta.expected_events or []) if meta else []
        if meta and meta.wait_kind in ("either", "system_event") and event_type in expected:
            return True

        return False


def build_graph(sessions_dir: str = ".hrhelpdesk_sessions") -> Graph:
    """Build and compile the helpdesk graph with colocated ledger."""
    from src.engine.event_ledger import EventLedger

    checkpointer = JsonCheckpointer(sessions_dir=sessions_dir)
    graph = Graph()
    graph.compiled_graph = graph.build_graph(HelpdeskState, checkpointer=checkpointer)
    graph._ledger = EventLedger(ledger_dir=f"{sessions_dir}_ledger")
    set_ledger(graph._ledger)
    return graph


__all__ = ["Graph", "build_graph"]
