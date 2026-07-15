from typing import Any, Dict, List, Optional

from src.engine.chat_engine_session_state import (
    ChatEngineSessionState,
    new_chat_session_state,
)

from .state_transitions import State


class HelpdeskState(ChatEngineSessionState):
    """HR helpdesk durable state — hub + sticky topics.

    ``ticket_id`` / ``booking_id`` are ephemeral emit-payload channels: system
    events merge them via ``aemit_event(..., payload=...)``, and LangGraph only
    keeps keys declared on this TypedDict through ``ainvoke``.
    """

    open_tickets: List[str]
    bookings: List[Dict[str, Any]]
    last_event: Optional[str]
    last_event_at: Optional[str]
    pending_clarify: bool
    ticket_id: Optional[str]
    booking_id: Optional[str]


def new_helpdesk_session_state() -> HelpdeskState:
    """Return a fresh HelpdeskState parked at IDLE with hub routing queued."""
    chat_state = new_chat_session_state(
        current_state=State.IDLE.value,
        proposed_next=State.IDLE.value,
    )
    helpdesk_state: HelpdeskState = {
        **chat_state,
        "open_tickets": [],
        "bookings": [],
        "last_event": None,
        "last_event_at": None,
        "pending_clarify": False,
        "ticket_id": None,
        "booking_id": None,
    }
    return helpdesk_state


__all__ = ["HelpdeskState", "new_helpdesk_session_state"]
