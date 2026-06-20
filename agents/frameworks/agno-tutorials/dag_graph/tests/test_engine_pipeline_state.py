"""
Tests for engine/pipeline_state.py — EngineState control plane.
"""

import pytest
from engine.pipeline_state import EngineState, init_engine_state, audit


class TestEngineStateInitialization:
    """Tests for init_engine_state() factory function."""

    def test_init_engine_state_returns_dict(self):
        """init_engine_state() returns a dict-like EngineState."""
        state = init_engine_state()
        assert isinstance(state, dict)

    def test_init_engine_state_has_turn_input_field(self):
        """init_engine_state() includes turn_input field (None by default)."""
        state = init_engine_state()
        assert "turn_input" in state
        assert state["turn_input"] is None

    def test_init_engine_state_has_turn_number_field(self):
        """init_engine_state() includes turn_number field (0 by default)."""
        state = init_engine_state()
        assert "turn_number" in state
        assert state["turn_number"] == 0

    def test_init_engine_state_has_turns_field(self):
        """init_engine_state() includes turns field (empty list by default)."""
        state = init_engine_state()
        assert "turns" in state
        assert state["turns"] == []
        assert isinstance(state["turns"], list)

    def test_init_engine_state_has_semantic_context_field(self):
        """init_engine_state() includes semantic_context field."""
        state = init_engine_state()
        assert "semantic_context" in state
        assert isinstance(state["semantic_context"], dict)
        assert "entities" in state["semantic_context"]
        assert "intents" in state["semantic_context"]

    def test_init_engine_state_has_conversation_id_field(self):
        """init_engine_state() includes conversation_id field (empty string by default)."""
        state = init_engine_state()
        assert "conversation_id" in state
        assert state["conversation_id"] == ""

    def test_init_engine_state_has_max_history_turns_field(self):
        """init_engine_state() includes max_history_turns field (10 by default)."""
        state = init_engine_state()
        assert "max_history_turns" in state
        assert state["max_history_turns"] == 10

    def test_init_engine_state_has_current_state_field(self):
        """init_engine_state() includes current_state field (init by default)."""
        state = init_engine_state()
        assert "current_state" in state
        assert state["current_state"] == "init"

    def test_init_engine_state_has_proposed_next_field(self):
        """init_engine_state() includes proposed_next field (init by default)."""
        state = init_engine_state()
        assert "proposed_next" in state
        assert state["proposed_next"] == "init"

    def test_init_engine_state_has_retry_count_field(self):
        """init_engine_state() includes retry_count field (0 by default)."""
        state = init_engine_state()
        assert "retry_count" in state
        assert state["retry_count"] == 0

    def test_init_engine_state_has_error_message_field(self):
        """init_engine_state() includes error_message field (None by default)."""
        state = init_engine_state()
        assert "error_message" in state
        assert state["error_message"] is None

    def test_init_engine_state_has_guardrail_ok_field(self):
        """init_engine_state() includes guardrail_ok field (True by default)."""
        state = init_engine_state()
        assert "guardrail_ok" in state
        assert state["guardrail_ok"] is True

    def test_init_engine_state_has_audit_trail_field(self):
        """init_engine_state() includes audit_trail field (empty list by default)."""
        state = init_engine_state()
        assert "audit_trail" in state
        assert state["audit_trail"] == []
        assert isinstance(state["audit_trail"], list)

    def test_init_engine_state_returns_fresh_state(self):
        """Each call to init_engine_state() returns a new instance."""
        state1 = init_engine_state()
        state2 = init_engine_state()
        # Verify they are different objects
        assert state1 is not state2
        # But have same values
        assert state1 == state2


class TestAuditFunction:
    """Tests for audit() helper function."""

    def test_audit_appends_to_audit_trail(self):
        """audit() appends entry to audit_trail."""
        state = init_engine_state()
        new_state = audit(state, "test entry")

        assert len(new_state["audit_trail"]) == 1
        assert new_state["audit_trail"][0] == "test entry"

    def test_audit_preserves_original_state(self):
        """audit() returns new state without modifying original (immutable-style)."""
        state = init_engine_state()
        original_len = len(state["audit_trail"])

        new_state = audit(state, "new entry")

        # Original unchanged
        assert len(state["audit_trail"]) == original_len
        # New state has entry
        assert len(new_state["audit_trail"]) == original_len + 1

    def test_audit_returns_dict(self):
        """audit() returns a dict."""
        state = init_engine_state()
        new_state = audit(state, "entry")
        assert isinstance(new_state, dict)

    def test_audit_appends_multiple_entries(self):
        """audit() can be chained to append multiple entries."""
        state = init_engine_state()
        state1 = audit(state, "entry 1")
        state2 = audit(state1, "entry 2")
        state3 = audit(state2, "entry 3")

        assert len(state3["audit_trail"]) == 3
        assert state3["audit_trail"][0] == "entry 1"
        assert state3["audit_trail"][1] == "entry 2"
        assert state3["audit_trail"][2] == "entry 3"

    def test_audit_with_empty_entry(self):
        """audit() handles empty string entries."""
        state = init_engine_state()
        new_state = audit(state, "")
        assert len(new_state["audit_trail"]) == 1
        assert new_state["audit_trail"][0] == ""

    def test_audit_with_special_characters(self):
        """audit() handles entries with special characters."""
        state = init_engine_state()
        entry = "error: validation failed [reason: 'invalid']"
        new_state = audit(state, entry)

        assert new_state["audit_trail"][0] == entry

    def test_audit_preserves_other_fields(self):
        """audit() preserves all other fields in state."""
        state = init_engine_state()
        state["current_state"] = "validate"
        state["turn_number"] = 5
        state["retry_count"] = 2

        new_state = audit(state, "entry")

        assert new_state["current_state"] == "validate"
        assert new_state["turn_number"] == 5
        assert new_state["retry_count"] == 2
        assert new_state["audit_trail"][0] == "entry"


class TestEngineStateTypeStructure:
    """Tests for EngineState TypedDict structure."""

    def test_engine_state_is_dict_like(self):
        """EngineState behaves like a dict."""
        state = init_engine_state()
        # Should support dict operations
        assert list(state.keys())
        assert list(state.values())
        assert "current_state" in state

    def test_all_required_fields_present(self):
        """init_engine_state() includes all required control plane fields."""
        state = init_engine_state()
        required_fields = [
            "turn_input", "turn_number", "turns", "semantic_context",
            "conversation_id", "max_history_turns", "current_state",
            "proposed_next", "retry_count", "error_message", "guardrail_ok",
            "audit_trail"
        ]
        for field in required_fields:
            assert field in state, f"Missing required field: {field}"
