"""Tests for multi-turn workflow support."""

import pytest

from src.engine.input_validation import InputValidationError
from src.workflow.pipeline_state import new_pipeline


def test_new_pipeline_includes_multi_turn_fields():
    """Test that new_pipeline includes all multi-turn fields."""
    state = new_pipeline("doc-001", timeout_seconds=30.0)

    # Multi-turn fields should be present
    assert "turn_input" in state
    assert "turn_number" in state
    assert "conversation_history" in state
    assert "max_history_turns" in state
    assert "router_timeout_sec" in state
    assert "user_id" in state
    assert "session_id" in state
    assert "semantic_context" in state
    assert "router_confidence" in state

    # Check initial values
    assert state["turn_input"] is None
    assert state["turn_number"] == 0
    assert state["conversation_history"] == []
    assert state["max_history_turns"] == 10
    assert state["router_timeout_sec"] == 10.0
    assert state["user_id"] is None
    assert state["session_id"] is None
    assert state["semantic_context"] == {}
    assert state["router_confidence"] == 0.0


def test_new_pipeline_existing_fields_preserved():
    """Test that existing fields are still present in new_pipeline."""
    state = new_pipeline("doc-001")

    # Existing fields should be present
    assert state["current_state"] == "init"
    assert state["proposed_next"] == "fetch"  # Initialized to FETCH state
    assert state["guardrail_ok"] is True
    assert state["retry_count"] == 0
    assert state["error_message"] is None
    assert isinstance(state["audit_trail"], list)
    assert state["document_id"] == "doc-001"
    assert state["raw_data"] is None
    assert state["validated_data"] is None
    assert state["enriched_data"] is None


def test_pipeline_state_conversation_history_is_list():
    """Test that conversation_history is initialized as list."""
    state = new_pipeline("doc-001")
    assert isinstance(state["conversation_history"], list)
    assert len(state["conversation_history"]) == 0


def test_pipeline_state_semantic_context_is_dict():
    """Test that semantic_context is initialized as dict."""
    state = new_pipeline("doc-001")
    assert isinstance(state["semantic_context"], dict)
    assert len(state["semantic_context"]) == 0


def test_pipeline_state_timeout_parameter():
    """Test that timeout_seconds parameter is properly handled."""
    state1 = new_pipeline("doc-001", timeout_seconds=60.0)
    state2 = new_pipeline("doc-002", timeout_seconds=300.0)

    # Both should have same multi-turn defaults regardless of timeout
    assert state1["max_history_turns"] == 10
    assert state2["max_history_turns"] == 10
    assert state1["router_timeout_sec"] == 10.0
    assert state2["router_timeout_sec"] == 10.0


def test_pipeline_state_document_id_is_set():
    """Test that document_id is properly set from entity_id."""
    state = new_pipeline("invoice-12345")
    assert state["document_id"] == "invoice-12345"


def test_conversation_history_structure():
    """Test the expected structure of conversation history entries."""
    state = new_pipeline("doc-001")

    # Manually add a turn entry (simulating what invoke_turn would do)
    turn_entry = {
        "role": "assistant",
        "content": "Processed to validate state",
        "semantic_context": {
            "entities": {"doc_type": "invoice"},
            "intents": ["submit"],
        },
        "state": "validate",
        "turn_number": 1,
    }

    state["conversation_history"].append(turn_entry)

    # Verify structure
    assert len(state["conversation_history"]) == 1
    assert state["conversation_history"][0]["role"] == "assistant"
    assert state["conversation_history"][0]["content"] == "Processed to validate state"
    assert state["conversation_history"][0]["state"] == "validate"


def test_multi_turn_state_mutable():
    """Test that multi-turn state fields can be modified."""
    state = new_pipeline("doc-001")

    # Modify fields
    state["turn_number"] = 1
    state["turn_input"] = "Process this document"
    state["user_id"] = "user-123"
    state["session_id"] = "session-456"
    state["router_confidence"] = 0.95

    # Verify modifications
    assert state["turn_number"] == 1
    assert state["turn_input"] == "Process this document"
    assert state["user_id"] == "user-123"
    assert state["session_id"] == "session-456"
    assert state["router_confidence"] == 0.95


def test_semantic_context_modification():
    """Test that semantic_context can be populated."""
    state = new_pipeline("doc-001")

    state["semantic_context"]["entities"] = {"doc_id": "ABC123"}
    state["semantic_context"]["intents"] = ["review", "approve"]

    assert state["semantic_context"]["entities"]["doc_id"] == "ABC123"
    assert "review" in state["semantic_context"]["intents"]


def test_history_trimming_simulation():
    """Test that conversation history can be trimmed to max_history_turns."""
    state = new_pipeline("doc-001")
    max_turns = state["max_history_turns"]  # 10

    # Add more than max_turns
    for i in range(15):
        state["conversation_history"].append({
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"Turn {i}",
            "state": "validate",
        })

    # Trim to max
    state["conversation_history"] = state["conversation_history"][-max_turns:]

    assert len(state["conversation_history"]) == max_turns
    assert state["conversation_history"][0]["content"] == "Turn 5"  # First 5 dropped
    assert state["conversation_history"][-1]["content"] == "Turn 14"  # Last kept
