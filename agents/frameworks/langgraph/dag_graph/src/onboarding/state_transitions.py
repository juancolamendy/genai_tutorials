"""Domain-specific state machine for the onboarding pipeline.

Defines states, allowed transitions, and state types — mirrors
docprocessing/state_transitions.py's shape.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Set

# ─────────────────────────────────────────────────────────────────────────────
# STATE ENUMERATION
# ─────────────────────────────────────────────────────────────────────────────

class State(str, Enum):
    """Onboarding pipeline state enumeration (design spec §3)."""

    INIT = "init"
    COLLECT = "collect"
    WELCOME_SENT = "welcome_sent"
    AWAIT_DOCUMENTS_SIGNED = "await_documents_signed"
    IT_PROVISIONED = "it_provisioned"
    AWAIT_HARDWARE_DELIVERED = "await_hardware_delivered"
    SCHEDULE_SENT = "schedule_sent"
    COMPLETE = "complete"
    ESCALATED = "escalated"
    ERROR = "error"


# ─────────────────────────────────────────────────────────────────────────────
# STATE TRANSITIONS
# ─────────────────────────────────────────────────────────────────────────────

happy_path: dict[State, State] = {
    State.INIT: State.COLLECT,
    State.COLLECT: State.WELCOME_SENT,
    State.WELCOME_SENT: State.AWAIT_DOCUMENTS_SIGNED,
    State.AWAIT_DOCUMENTS_SIGNED: State.IT_PROVISIONED,
    State.IT_PROVISIONED: State.AWAIT_HARDWARE_DELIVERED,
    State.AWAIT_HARDWARE_DELIVERED: State.SCHEDULE_SENT,
    State.SCHEDULE_SENT: State.COMPLETE,
}

terminal_states = set({State.COMPLETE, State.ESCALATED, State.ERROR})

# Adjacency list: which states can follow each state
allowed_transitions: Dict[State, Set[State]] = {
    State.INIT: {State.COLLECT},
    State.COLLECT: {State.COLLECT, State.WELCOME_SENT, State.ERROR},
    State.WELCOME_SENT: {State.AWAIT_DOCUMENTS_SIGNED, State.ERROR},
    State.AWAIT_DOCUMENTS_SIGNED: {
        State.IT_PROVISIONED,
        State.ESCALATED,
        State.ERROR,
    },
    State.IT_PROVISIONED: {State.AWAIT_HARDWARE_DELIVERED, State.ERROR},
    State.AWAIT_HARDWARE_DELIVERED: {
        State.SCHEDULE_SENT,
        State.ESCALATED,
        State.ERROR,
    },
    State.SCHEDULE_SENT: {State.COMPLETE, State.ERROR},
    State.COMPLETE: set(),
    State.ESCALATED: set(),
    State.ERROR: set(),
}


def is_transition_allowed(current: State, proposed: State) -> bool:
    """Check if transition from current to proposed state is allowed.

    Args:
        current: Current state
        proposed: Proposed next state

    Returns:
        True if transition is in allowed_transitions, False otherwise
    """
    return proposed in allowed_transitions.get(current, set())


__all__ = [
    "State",
    "allowed_transitions",
    "happy_path",
    "terminal_states",
    "is_transition_allowed",
]
