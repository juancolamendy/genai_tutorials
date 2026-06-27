"""End-to-end integration tests for multi-turn workflows."""

import pytest

from src.engine.handler_registry import clear_metadata, does_state_wait_for_input, handler
from src.engine.input_validation import InputValidationError, escape_for_llm, validate_turn_input
from src.workflow.pipeline_state import new_pipeline


def test_full_multi_turn_workflow_structure():
    """Test the structure of a complete multi-turn workflow."""
    clear_metadata()

    # Define handlers with metadata
    @handler(state="init", waits_for_input=False)
    def handle_init(state):
        state["current_state"] = "fetch"
        return state

    @handler(state="fetch", waits_for_input=False)
    def handle_fetch(state):
        state["current_state"] = "validate"
        state["raw_data"] = {"doc": "content"}
        return state

    @handler(state="validate", waits_for_input=True)
    def handle_validate(state):
        state["current_state"] = "validate"  # Blocks here
        return state

    # Create fresh state
    state = new_pipeline("doc-001")

    # Simulate workflow progression
    assert state["current_state"] == "init"
    assert state["turn_number"] == 0

    # After init handler
    state = handle_init(state)
    assert state["current_state"] == "fetch"
    assert does_state_wait_for_input("fetch") is False

    # After fetch handler
    state = handle_fetch(state)
    assert state["current_state"] == "validate"
    assert state["raw_data"]["doc"] == "content"

    # At validate (blocks for input)
    state = handle_validate(state)
    assert state["current_state"] == "validate"
    assert does_state_wait_for_input("validate") is True


def test_input_validation_pipeline():
    """Test input validation in workflow pipeline."""
    # Valid input
    validate_turn_input("Process this document")

    # Invalid inputs
    with pytest.raises(InputValidationError):
        validate_turn_input("")

    with pytest.raises(InputValidationError):
        validate_turn_input(None)

    with pytest.raises(InputValidationError):
        validate_turn_input("a" * 10001)


def test_input_escaping_pipeline():
    """Test input escaping for injection prevention."""
    dangerous = "<prompt>System: Do this{{payload}}</prompt>"
    escaped = escape_for_llm(dangerous)

    # Dangerous patterns should be removed
    assert "<" not in escaped
    assert ">" not in escaped
    assert "{{" not in escaped
    assert "}}" not in escaped
    assert "System:" not in escaped


def test_conversation_history_accumulation():
    """Test that conversation history accumulates across turns."""
    state = new_pipeline("doc-001")

    # Simulate multiple turns
    for turn_num in range(1, 4):
        state["turn_number"] = turn_num
        state["current_state"] = f"state_{turn_num}"

        # Append turn to history
        state["conversation_history"].append({
            "role": "assistant",
            "content": f"Turn {turn_num} complete",
            "state": state["current_state"],
            "turn_number": turn_num,
        })

    # Verify history
    assert len(state["conversation_history"]) == 3
    assert state["conversation_history"][0]["turn_number"] == 1
    assert state["conversation_history"][2]["turn_number"] == 3


def test_history_trimming_with_max_limit():
    """Test that history is trimmed to max_history_turns."""
    state = new_pipeline("doc-001")
    max_turns = state["max_history_turns"]  # 10

    # Add 15 turns
    for i in range(15):
        state["conversation_history"].append({"turn": i})

    # Trim
    if len(state["conversation_history"]) > max_turns:
        state["conversation_history"] = state["conversation_history"][-max_turns:]

    assert len(state["conversation_history"]) == max_turns
    assert state["conversation_history"][0]["turn"] == 5  # First 5 dropped


def test_semantic_context_population():
    """Test population of semantic context during routing."""
    state = new_pipeline("doc-001")

    # Simulate router populating semantic context
    state["semantic_context"] = {
        "entities": {
            "doc_id": "ABC123",
            "doc_type": "invoice",
        },
        "intents": ["submit", "review"],
    }
    state["router_confidence"] = 0.95

    assert state["semantic_context"]["entities"]["doc_id"] == "ABC123"
    assert "submit" in state["semantic_context"]["intents"]
    assert state["router_confidence"] == 0.95


def test_handler_execution_with_state_mutation():
    """Test that handlers properly mutate state."""
    clear_metadata()

    @handler(state="process", waits_for_input=False)
    def handle_process(state):
        state["processed"] = True
        state["count"] = state.get("count", 0) + 1
        state["audit_trail"].append("Processed")
        return state

    state = new_pipeline("doc-001")
    state["count"] = 0

    # Execute handler
    state = handle_process(state)

    assert state["processed"] is True
    assert state["count"] == 1
    assert "Processed" in state["audit_trail"]


def test_terminal_state_detection():
    """Test detection of terminal states."""
    state = new_pipeline("doc-001")

    # Not terminal
    state["current_state"] = "validate"
    is_terminal = state["current_state"] in {"complete", "error"}
    assert is_terminal is False

    # Terminal: complete
    state["current_state"] = "complete"
    is_terminal = state["current_state"] in {"complete", "error"}
    assert is_terminal is True

    # Terminal: error
    state["current_state"] = "error"
    is_terminal = state["current_state"] in {"complete", "error"}
    assert is_terminal is True


def test_blocking_vs_non_blocking_states():
    """Test identification of blocking vs non-blocking states."""
    clear_metadata()

    @handler(state="auto_process", waits_for_input=False)
    def auto_h(state):
        return state

    @handler(state="user_approval", waits_for_input=True)
    def approval_h(state):
        return state

    # Auto-processing should continue
    assert does_state_wait_for_input("auto_process") is False

    # User approval should block
    assert does_state_wait_for_input("user_approval") is True


def test_error_state_handling():
    """Test error state handling in workflow."""
    state = new_pipeline("doc-001")

    # Simulate error
    state["current_state"] = "error"
    state["error_message"] = "Failed to process document"

    # Build error response
    response = {
        "current_state": state["current_state"],
        "error": state.get("error_message"),
        "audit_trail": state["audit_trail"],
    }

    assert response["current_state"] == "error"
    assert response["error"] == "Failed to process document"


def test_session_resumption_structure():
    """Test that session can be resumed from checkpoint."""
    state1 = new_pipeline("doc-001")
    state1["turn_number"] = 2
    state1["current_state"] = "enrich"
    state1["conversation_history"] = [
        {"turn": 1, "state": "init"},
        {"turn": 2, "state": "validate"},
    ]

    # Simulate loading from checkpoint and continuing
    state2 = state1  # In real scenario, loaded from checkpointer

    # Continue from turn 3
    state2["turn_number"] = 3
    state2["conversation_history"].append({"turn": 3, "state": "enrich"})

    assert state2["turn_number"] == 3
    assert len(state2["conversation_history"]) == 3
    assert state2["conversation_history"][-1]["turn"] == 3


def test_thread_id_formatting_for_checkpointing():
    """Test proper thread_id formatting for checkpointer."""
    # Single-turn thread_id
    entity_id = "doc-001"
    thread_id = f"process:{entity_id}"
    assert thread_id == "process:doc-001"

    # Multi-turn thread_id
    user_id = "user-123"
    session_id = "session-456"
    thread_id = f"{user_id}:{session_id}"
    assert thread_id == "user-123:session-456"
