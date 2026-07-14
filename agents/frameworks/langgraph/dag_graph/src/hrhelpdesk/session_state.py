from typing import Any, Dict, List, Optional

from src.engine.engine_session_state import ChatEngineSessionState, new_chat_session_state

from .state_transitions import State


class HelpdeskState(ChatEngineSessionState):
    """HR helpdesk durable state — hub + sticky topics."""

    open_tickets: List[str]
    bookings: List[Dict[str, Any]]
    last_event: Optional[str]
    last_event_at: Optional[str]
    pending_clarify: bool


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
    }
    return helpdesk_state


__all__ = ["HelpdeskState", "new_helpdesk_session_state"]
