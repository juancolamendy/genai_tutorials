"""Router node for state machine routing."""

from typing import Dict
from .state_machine import State, PipelineState

HAPPY_PATH: Dict[State, State] = {
    State.INIT: State.FETCH,
    State.FETCH: State.VALIDATE,
    State.VALIDATE: State.ENRICH,
    State.ENRICH: State.STORE,
    State.STORE: State.COMPLETE,
    State.RETRY: State.FETCH,
    State.HUMAN_REVIEW: State.ENRICH,
}


def router(state: PipelineState) -> PipelineState:
    """Route to next state based on current state.

    Pure function that reads current_state from HAPPY_PATH routing table
    and proposes the next state. Appends routing decision to audit trail.

    Args:
        state: PipelineState with current_state set

    Returns:
        Updated state with proposed_next and audit_trail modified
    """
    current = State(state["current_state"])
    proposed = HAPPY_PATH.get(current, State.ERROR)

    audit_entry = f"router: {current.value} → {proposed.value}"

    return {
        **state,
        "proposed_next": proposed.value,
        "audit_trail": state["audit_trail"] + [audit_entry],
    }
