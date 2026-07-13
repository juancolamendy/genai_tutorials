"""Domain state machine for Sentry-error triage-and-fix harness.

Mirrors onboarding/state_transitions.py. The review gate is AWAIT_APPROVAL
(wait_kind=either) — not LangGraph interrupt().
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Set


class State(str, Enum):
    """Triage-and-fix pipeline states (harness steps 1–8)."""

    INIT = "init"
    INTAKE = "intake"
    GATHER_CONTEXT = "gather_context"
    EXECUTE_PATCH = "execute_patch"
    COLLECT_DIFF = "collect_diff"
    AWAIT_APPROVAL = "await_approval"
    PUBLISH = "publish"
    REJECT = "reject"
    WRITE_REPORT = "write_report"
    COMPLETE = "complete"
    REJECTED = "rejected"
    ERROR = "error"


happy_path: dict[State, State] = {
    State.INIT: State.INTAKE,
    State.INTAKE: State.GATHER_CONTEXT,
    State.GATHER_CONTEXT: State.EXECUTE_PATCH,
    State.EXECUTE_PATCH: State.COLLECT_DIFF,
    State.COLLECT_DIFF: State.AWAIT_APPROVAL,
    State.AWAIT_APPROVAL: State.PUBLISH,
    State.PUBLISH: State.WRITE_REPORT,
    State.REJECT: State.WRITE_REPORT,
    State.WRITE_REPORT: State.COMPLETE,
}

terminal_states = {State.COMPLETE, State.REJECTED, State.ERROR}

allowed_transitions: Dict[State, Set[State]] = {
    State.INIT: {State.INTAKE},
    State.INTAKE: {State.GATHER_CONTEXT, State.ERROR},
    State.GATHER_CONTEXT: {State.EXECUTE_PATCH, State.ERROR},
    State.EXECUTE_PATCH: {State.COLLECT_DIFF, State.ERROR},
    State.COLLECT_DIFF: {State.AWAIT_APPROVAL, State.ERROR},
    State.AWAIT_APPROVAL: {State.PUBLISH, State.REJECT, State.ERROR},
    State.PUBLISH: {State.WRITE_REPORT, State.ERROR},
    State.REJECT: {State.WRITE_REPORT, State.ERROR},
    State.WRITE_REPORT: {State.COMPLETE, State.REJECTED, State.ERROR},
    State.COMPLETE: set(),
    State.REJECTED: set(),
    State.ERROR: set(),
}


def is_transition_allowed(current: State, proposed: State) -> bool:
    return proposed in allowed_transitions.get(current, set())


__all__ = [
    "State",
    "allowed_transitions",
    "happy_path",
    "terminal_states",
    "is_transition_allowed",
]
