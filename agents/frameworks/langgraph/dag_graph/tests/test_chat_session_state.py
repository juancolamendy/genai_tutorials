"""Tests for ChatEngineSessionState (chatbot sticky-lane fields)."""

from src.engine.engine_session_state import (
    new_chat_session_state,
    new_engine_session_state,
)


def test_new_chat_session_state_includes_lane_fields():
    state = new_chat_session_state()
    assert state["schema_version"] == 1
    assert state["active_topic"] is None
    assert state["topic_started_at"] is None
    assert state["topic_data"] == {}


def test_new_chat_session_state_defaults_to_idle_hub():
    state = new_chat_session_state()
    assert state["current_state"] == "idle"
    assert state["proposed_next"] == "route"


def test_new_engine_session_state_does_not_include_chat_fields():
    """Linear pipelines must not be forced into the chat schema."""
    state = new_engine_session_state()
    assert "active_topic" not in state
    assert "topic_data" not in state
    assert "schema_version" not in state
