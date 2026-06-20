"""State machine definitions, data models, and validation."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Set
from typing_extensions import TypedDict


class State(str, Enum):
    """Pipeline state enumeration."""

    INIT = "init"
    FETCH = "fetch"
    VALIDATE = "validate"
    ENRICH = "enrich"
    STORE = "store"
    COMPLETE = "complete"
    RETRY = "retry"
    HUMAN_REVIEW = "human_review"
    ERROR = "error"


class PipelineState(TypedDict):
    """Central state dict for pipeline execution.

    Fields organized into three categories:
    - Control Plane: routing and error tracking
    - Execution Tracking: timing and limits
    - Business Payload: document data
    """

    # ─ Control Plane ─────────────────────────────────────────────────────
    current_state: str
    proposed_next: str
    retry_count: int
    error_message: Optional[str]
    error_type: Optional[str]
    audit_trail: list[str]
    fallback_depth: int

    # ─ Execution Tracking ────────────────────────────────────────────────
    started_at: float
    node_timeout_seconds: int

    # ─ Business Payload ──────────────────────────────────────────────────
    document_id: str
    raw_data: Optional[Dict[str, Any]]
    validated_data: Optional[Dict[str, Any]]
    enriched_data: Optional[Dict[str, Any]]


@dataclass
class GuardrailResult:
    """Result of guardrail check."""

    passed: bool
    reason: str = ""
    fallback: Optional[State] = None


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
    if current not in ALLOWED_TRANSITIONS:
        return False
    return proposed in ALLOWED_TRANSITIONS[current]
