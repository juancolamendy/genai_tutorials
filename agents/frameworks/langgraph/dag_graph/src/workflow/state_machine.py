"""Domain-specific state machine for document processing pipeline.

Defines states, allowed transitions, and state types for the workflow.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Set

# ─────────────────────────────────────────────────────────────────────────────
# STATE ENUMERATION
# ─────────────────────────────────────────────────────────────────────────────

class State(str, Enum):
    """Pipeline state enumeration."""

    INIT = "init"
    FETCH = "fetch"
    VALIDATE = "validate"
    ENRICH = "enrich"
    STORE = "store"
    COMPLETE = "complete"
    RETRY = "retry"
    ERROR = "error"
    HUMAN_REVIEW = "human_review"


# ─────────────────────────────────────────────────────────────────────────────
# STATE TRANSITIONS
# ─────────────────────────────────────────────────────────────────────────────

# Adjacency list: which states can follow each state
ALLOWED_TRANSITIONS: Dict[State, Set[State]] = {
    State.INIT: {State.FETCH},
    State.FETCH: {State.VALIDATE, State.RETRY, State.ERROR},
    State.VALIDATE: {State.ENRICH, State.HUMAN_REVIEW, State.ERROR},
    State.ENRICH: {State.STORE, State.RETRY, State.ERROR},
    State.STORE: {State.COMPLETE, State.RETRY, State.ERROR},
    State.RETRY: {State.FETCH, State.ERROR},
    State.HUMAN_REVIEW: {State.ENRICH, State.ERROR},
    State.COMPLETE: set(),
    State.ERROR: set(),
}


def is_transition_allowed(current: State, proposed: State) -> bool:
    """Check if transition from current to proposed state is allowed.

    Args:
        current: Current state
        proposed: Proposed next state

    Returns:
        True if transition is in ALLOWED_TRANSITIONS, False otherwise
    """
    return proposed in ALLOWED_TRANSITIONS.get(current, set())


__all__ = [
    "State",
    "ALLOWED_TRANSITIONS",
    "is_transition_allowed",
]
