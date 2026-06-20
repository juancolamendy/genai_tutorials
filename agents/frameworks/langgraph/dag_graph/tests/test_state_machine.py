"""Tests for state machine, PipelineState, and GuardrailResult."""

import pytest
from src.engine.state_machine import (
    State,
    PipelineState,
    GuardrailResult,
    ALLOWED_TRANSITIONS,
    is_transition_allowed,
)


class TestStateEnum:
    """Tests for State enum."""

    def test_state_enum_has_nine_values(self):
        """Test that State enum has 9 values."""
        states = [s.value for s in State]
        expected = [
            "init",
            "fetch",
            "validate",
            "enrich",
            "store",
            "complete",
            "retry",
            "human_review",
            "error",
        ]
        assert len(states) == 9
        assert states == expected

    def test_state_init(self):
        """Test State.INIT value."""
        assert State.INIT.value == "init"

    def test_state_complete(self):
        """Test State.COMPLETE value."""
        assert State.COMPLETE.value == "complete"

    def test_state_error(self):
        """Test State.ERROR value."""
        assert State.ERROR.value == "error"


class TestAllowedTransitions:
    """Tests for ALLOWED_TRANSITIONS mapping."""

    def test_allowed_transitions_has_all_states(self):
        """Test that ALLOWED_TRANSITIONS covers all 9 states."""
        assert len(ALLOWED_TRANSITIONS) == 9
        for state in State:
            assert state in ALLOWED_TRANSITIONS

    def test_init_transitions(self):
        """Test INIT transitions."""
        transitions = ALLOWED_TRANSITIONS[State.INIT]
        assert transitions == {State.FETCH}

    def test_fetch_transitions(self):
        """Test FETCH transitions."""
        transitions = ALLOWED_TRANSITIONS[State.FETCH]
        assert transitions == {State.VALIDATE, State.RETRY, State.ERROR}

    def test_validate_transitions(self):
        """Test VALIDATE transitions."""
        transitions = ALLOWED_TRANSITIONS[State.VALIDATE]
        assert transitions == {State.ENRICH, State.HUMAN_REVIEW, State.ERROR}

    def test_enrich_transitions(self):
        """Test ENRICH transitions."""
        transitions = ALLOWED_TRANSITIONS[State.ENRICH]
        assert transitions == {State.STORE, State.RETRY, State.ERROR}

    def test_store_transitions(self):
        """Test STORE transitions."""
        transitions = ALLOWED_TRANSITIONS[State.STORE]
        assert transitions == {State.COMPLETE, State.RETRY, State.ERROR}

    def test_retry_transitions(self):
        """Test RETRY transitions."""
        transitions = ALLOWED_TRANSITIONS[State.RETRY]
        assert transitions == {State.FETCH, State.ERROR}

    def test_human_review_transitions(self):
        """Test HUMAN_REVIEW transitions."""
        transitions = ALLOWED_TRANSITIONS[State.HUMAN_REVIEW]
        assert transitions == {State.ENRICH, State.ERROR}

    def test_complete_transitions(self):
        """Test COMPLETE terminal state has no transitions."""
        transitions = ALLOWED_TRANSITIONS[State.COMPLETE]
        assert transitions == set()

    def test_error_transitions(self):
        """Test ERROR terminal state has no transitions."""
        transitions = ALLOWED_TRANSITIONS[State.ERROR]
        assert transitions == set()


class TestGuardrailResult:
    """Tests for GuardrailResult dataclass."""

    def test_guardrail_result_passed_only(self):
        """Test GuardrailResult with passed=True."""
        result = GuardrailResult(passed=True)
        assert result.passed is True
        assert result.reason == ""
        assert result.fallback is None

    def test_guardrail_result_failed_with_reason(self):
        """Test GuardrailResult with passed=False and reason."""
        result = GuardrailResult(
            passed=False, reason="test failure", fallback=State.RETRY
        )
        assert result.passed is False
        assert result.reason == "test failure"
        assert result.fallback == State.RETRY

    def test_guardrail_result_default_values(self):
        """Test GuardrailResult default field values."""
        result = GuardrailResult(passed=True)
        assert result.reason == ""
        assert result.fallback is None


class TestPipelineState:
    """Tests for PipelineState TypedDict."""

    def test_pipeline_state_has_all_fields(self):
        """Test that PipelineState can be instantiated with all 17 fields."""
        state: PipelineState = {
            "current_state": "init",
            "proposed_next": "fetch",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": ["init"],
            "fallback_depth": 0,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "TEST-001",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        assert state["current_state"] == "init"
        assert state["document_id"] == "TEST-001"
        assert len(state) == 13  # All fields present

    def test_pipeline_state_control_plane_fields(self):
        """Test control plane fields in PipelineState."""
        state: PipelineState = {
            "current_state": "fetch",
            "proposed_next": "validate",
            "retry_count": 1,
            "error_message": "Test error",
            "error_type": "transient",
            "audit_trail": ["init", "fetch"],
            "fallback_depth": 0,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "TEST-001",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        assert state["current_state"] == "fetch"
        assert state["proposed_next"] == "validate"
        assert state["retry_count"] == 1
        assert state["error_type"] == "transient"
        assert state["fallback_depth"] == 0

    def test_pipeline_state_business_payload_fields(self):
        """Test business payload fields in PipelineState."""
        state: PipelineState = {
            "current_state": "validate",
            "proposed_next": "enrich",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "DOC-123",
            "raw_data": {"content": "test"},
            "validated_data": {"schema": "valid"},
            "enriched_data": {"tags": ["test"]},
        }
        assert state["document_id"] == "DOC-123"
        assert state["raw_data"]["content"] == "test"
        assert state["validated_data"]["schema"] == "valid"
        assert state["enriched_data"]["tags"] == ["test"]


class TestIsTransitionAllowed:
    """Tests for is_transition_allowed function."""

    def test_init_to_fetch_allowed(self):
        """Test that INIT → FETCH is allowed."""
        assert is_transition_allowed(State.INIT, State.FETCH) is True

    def test_fetch_to_validate_allowed(self):
        """Test that FETCH → VALIDATE is allowed."""
        assert is_transition_allowed(State.FETCH, State.VALIDATE) is True

    def test_fetch_to_error_allowed(self):
        """Test that FETCH → ERROR is allowed."""
        assert is_transition_allowed(State.FETCH, State.ERROR) is True

    def test_fetch_to_complete_not_allowed(self):
        """Test that FETCH → COMPLETE is NOT allowed."""
        assert is_transition_allowed(State.FETCH, State.COMPLETE) is False

    def test_complete_to_any_not_allowed(self):
        """Test that COMPLETE → * is NOT allowed (terminal)."""
        assert is_transition_allowed(State.COMPLETE, State.FETCH) is False
        assert is_transition_allowed(State.COMPLETE, State.ERROR) is False

    def test_error_to_any_not_allowed(self):
        """Test that ERROR → * is NOT allowed (terminal)."""
        assert is_transition_allowed(State.ERROR, State.FETCH) is False
        assert is_transition_allowed(State.ERROR, State.COMPLETE) is False

    def test_retry_to_fetch_allowed(self):
        """Test that RETRY → FETCH is allowed."""
        assert is_transition_allowed(State.RETRY, State.FETCH) is True

    def test_human_review_to_enrich_allowed(self):
        """Test that HUMAN_REVIEW → ENRICH is allowed."""
        assert is_transition_allowed(State.HUMAN_REVIEW, State.ENRICH) is True

    def test_human_review_to_fetch_not_allowed(self):
        """Test that HUMAN_REVIEW → FETCH is NOT allowed."""
        assert is_transition_allowed(State.HUMAN_REVIEW, State.FETCH) is False

    def test_all_states_in_allowed_transitions(self):
        """Test that all State enum values are in ALLOWED_TRANSITIONS."""
        for state in State:
            assert (
                state in ALLOWED_TRANSITIONS
            ), f"State {state} not in ALLOWED_TRANSITIONS"
