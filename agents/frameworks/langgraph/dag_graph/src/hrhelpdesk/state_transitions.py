"""Domain-specific state machine for the HR helpdesk chatbot."""

from __future__ import annotations

from enum import Enum
from typing import Dict, Set

# ─────────────────────────────────────────────────────────────────────────────
# STATE ENUMERATION
# ─────────────────────────────────────────────────────────────────────────────


class State(str, Enum):
    """HR helpdesk hub + topic state enumeration."""

    ROUTE = "route"
    HUB_CLARIFY = "hub_clarify"
    TOPIC_FAQ = "topic_faq"
    TOPIC_ESCALATE = "topic_escalate"
    TOPIC_BOOKING = "topic_booking"
    NOTIFY_USER = "notify_user"
    IDLE = "idle"
    ERROR = "error"


# ─────────────────────────────────────────────────────────────────────────────
# STATE TRANSITIONS
# ─────────────────────────────────────────────────────────────────────────────

happy_path: dict[State, State] = {
    State.ROUTE: State.IDLE,
    State.HUB_CLARIFY: State.IDLE,
    State.TOPIC_FAQ: State.IDLE,
    State.TOPIC_ESCALATE: State.IDLE,
    State.TOPIC_BOOKING: State.IDLE,
    State.NOTIFY_USER: State.IDLE,
    State.IDLE: State.ROUTE,
}

terminal_states = {State.ERROR}

allowed_transitions: Dict[State, Set[State]] = {
    State.ROUTE: {
        State.HUB_CLARIFY,
        State.TOPIC_FAQ,
        State.TOPIC_ESCALATE,
        State.TOPIC_BOOKING,
        State.NOTIFY_USER,
        State.IDLE,
        State.ERROR,
    },
    State.HUB_CLARIFY: {State.IDLE, State.ERROR},
    State.TOPIC_FAQ: {State.IDLE, State.TOPIC_BOOKING, State.ROUTE, State.ERROR},
    State.TOPIC_ESCALATE: {State.IDLE, State.TOPIC_BOOKING, State.ROUTE, State.ERROR},
    State.TOPIC_BOOKING: {
        State.IDLE,
        State.TOPIC_BOOKING,
        State.TOPIC_FAQ,
        State.TOPIC_ESCALATE,
        State.ROUTE,
        State.ERROR,
    },
    State.NOTIFY_USER: {State.IDLE, State.ERROR},
    State.IDLE: {
        State.ROUTE,
        State.NOTIFY_USER,
        State.HUB_CLARIFY,
        State.TOPIC_FAQ,
        State.TOPIC_ESCALATE,
        State.TOPIC_BOOKING,
        State.ERROR,
    },
    State.ERROR: set(),
}


def is_transition_allowed(current: State, proposed: State) -> bool:
    """Check if transition from current to proposed state is allowed."""
    return proposed in allowed_transitions.get(current, set())


__all__ = [
    "State",
    "allowed_transitions",
    "happy_path",
    "terminal_states",
    "is_transition_allowed",
]
