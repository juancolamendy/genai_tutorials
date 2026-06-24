"""
state_machine.py
────────────────────────────────────────────────────────────────────────────
State machine definition: states, allowed transitions, and the transition
validator that every guardrail uses.
"""

from enum import Enum


# structures
class State(str, Enum):
    INIT                = "init"
    FETCH               = "fetch"
    VALIDATE            = "validate"
    UPLOAD_SUPPORT_DOCS = "upload_support_docs"
    ENRICH              = "enrich"
    STORE               = "store"
    COMPLETE            = "complete"
    RETRY               = "retry"
    ERROR               = "error"
    HUMAN_REVIEW        = "human_review"


# variables
# Adjacency list: which states may follow each state.
# This is the single source of truth for legal transitions.
ALLOWED_TRANSITIONS: dict[State, set[State]] = {
    State.INIT:                {State.FETCH},
    State.FETCH:               {State.VALIDATE,          State.RETRY,        State.ERROR},
    State.VALIDATE:            {State.UPLOAD_SUPPORT_DOCS, State.HUMAN_REVIEW, State.ERROR},
    State.UPLOAD_SUPPORT_DOCS: {State.ENRICH,            State.RETRY,        State.ERROR},
    State.ENRICH:              {State.STORE,             State.RETRY,        State.ERROR},
    State.STORE:               {State.COMPLETE,          State.RETRY,        State.ERROR},
    State.RETRY:               {State.FETCH,             State.ERROR},
    State.HUMAN_REVIEW:        {State.ENRICH,            State.ERROR},
    State.COMPLETE:            set(),
    State.ERROR:               set(),
}


# States where the pipeline stops looping.
TERMINAL_STATES: frozenset[State] = frozenset({State.COMPLETE, State.ERROR})


# functions
def is_transition_allowed(current: State, proposed: State) -> bool:
    """Return True iff `proposed` appears in the adjacency list for `current`."""
    return proposed in ALLOWED_TRANSITIONS.get(current, set())

def get_allowed_transitions() -> dict:
    return ALLOWED_TRANSITIONS
