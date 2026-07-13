"""Tests for EngineSessionState's event-extension fields and engine/errors.py.

Covers the additive fields from the engine-event-extension design (§4):
current_event_source, current_event_type, output_messages — and the new
exception types (§2.2) used by ainvoke/aemit_event/arun_to_completion in
later phases.
"""

import pytest

from src.engine.engine_session_state import new_engine_session_state
from src.engine.errors import (
    GraphAlreadyCompleteError,
    GraphAlreadyInteractiveError,
    GraphIncompleteError,
    GraphRunError,
)

# ─────────────────────────────────────────────────────────────────────────────
# EngineSessionState additions
# ─────────────────────────────────────────────────────────────────────────────

def test_new_session_state_initializes_output_messages_empty():
    """output_messages is reducer-backed like audit_trail/messages, so a fresh
    session must start with the reducer's identity value ([]), not absent."""
    state = new_engine_session_state()
    assert state["output_messages"] == []


def test_new_session_state_initializes_handler_status_ok():
    state = new_engine_session_state()
    assert state["handler_status"] == "ok"


def test_new_session_state_does_not_set_current_event_source():
    """current_event_source is stamped fresh by aemit_event's branches on
    every turn (never carried over from a prior turn), so a freshly
    constructed session should leave it unset rather than pre-seeding a
    value that could go stale."""
    state = new_engine_session_state()
    assert "current_event_source" not in state


def test_new_session_state_does_not_set_current_event_type():
    state = new_engine_session_state()
    assert "current_event_type" not in state


def test_current_event_source_defaults_to_human_when_absent():
    """Documents the read-side contract: callers read this field via
    state.get("current_event_source", "human") so pre-existing docprocessing
    sessions (which never set it) behave as human-sourced."""
    state = new_engine_session_state()
    assert state.get("current_event_source", "human") == "human"


def test_current_event_type_defaults_to_message_when_absent():
    state = new_engine_session_state()
    assert state.get("current_event_type", "message") == "message"


def test_existing_fields_unaffected_by_new_additions():
    """Regression guard: the new fields must be purely additive — every
    field new_engine_session_state() already set must be unchanged."""
    state = new_engine_session_state()
    assert state["current_state"] == "init"
    assert state["proposed_next"] == "init"
    assert state["status"] == "ok"
    assert state["audit_trail"] == ["init session state"]
    assert state["messages"] == []
    assert state["turn_number"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# engine/errors.py
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "error_cls",
    [GraphIncompleteError, GraphAlreadyInteractiveError, GraphAlreadyCompleteError, GraphRunError],
)
def test_error_class_is_exception_subclass(error_cls):
    assert issubclass(error_cls, Exception)


@pytest.mark.parametrize(
    "error_cls",
    [GraphIncompleteError, GraphAlreadyInteractiveError, GraphAlreadyCompleteError, GraphRunError],
)
def test_error_class_raisable_with_message(error_cls):
    with pytest.raises(error_cls) as exc_info:
        raise error_cls("thread 'x' is not in a valid state for this operation")
    assert "thread 'x'" in str(exc_info.value)


def test_error_classes_are_distinct_types():
    """Each error type must be independently catchable — a caller catching
    GraphAlreadyCompleteError must not also catch GraphAlreadyInteractiveError
    (the round-2 design fix distinguishes 'already finished' from 'still
    mid-flow' specifically so callers can branch on which one occurred)."""
    errors = [
        GraphIncompleteError,
        GraphAlreadyInteractiveError,
        GraphAlreadyCompleteError,
        GraphRunError,
    ]
    assert len(set(errors)) == len(errors)

    with pytest.raises(GraphAlreadyCompleteError):
        try:
            raise GraphAlreadyCompleteError("already done")
        except GraphAlreadyInteractiveError:
            pytest.fail("complete-error must not be catchable as interactive-error")
