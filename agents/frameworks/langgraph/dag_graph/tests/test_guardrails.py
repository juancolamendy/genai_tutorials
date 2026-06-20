"""Tests for guardrail framework and composable checks."""

import pytest
import time
from src.engine.state_machine import (
    State,
    PipelineState,
    GuardrailResult,
    ALLOWED_TRANSITIONS,
)
from src.engine.guardrail import (
    GuardrailFn,
    make_guardrail,
    check_transition_allowed,
    check_retry_budget_with_error_type,
    check_raw_data_present,
    check_validated_data_present,
    check_enriched_data_present,
    check_document_size,
    check_fallback_depth,
    check_pipeline_timeout,
    GUARDRAILS,
)


class TestMakeGuardrail:
    """Tests for guardrail composition."""

    def test_make_guardrail_returns_callable(self):
        """Test that make_guardrail returns a callable."""
        def dummy_check(state: PipelineState) -> GuardrailResult:
            return GuardrailResult(passed=True)

        result = make_guardrail(dummy_check)
        assert callable(result)

    def test_make_guardrail_single_check_passes(self):
        """Test composed guardrail with single passing check."""
        def check_pass(state: PipelineState) -> GuardrailResult:
            return GuardrailResult(passed=True)

        composed = make_guardrail(check_pass)
        empty_state: PipelineState = {
            "current_state": "init",
            "proposed_next": "fetch",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = composed(empty_state)
        assert result.passed is True

    def test_make_guardrail_single_check_fails(self):
        """Test composed guardrail with single failing check."""
        def check_fail(state: PipelineState) -> GuardrailResult:
            return GuardrailResult(
                passed=False, reason="test failure", fallback=State.ERROR
            )

        composed = make_guardrail(check_fail)
        empty_state: PipelineState = {
            "current_state": "init",
            "proposed_next": "fetch",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = composed(empty_state)
        assert result.passed is False
        assert result.reason == "test failure"

    def test_make_guardrail_short_circuits_on_first_failure(self):
        """Test that guardrail composition short-circuits on first failure."""
        call_count = {"check1": 0, "check2": 0, "check3": 0}

        def check1(state: PipelineState) -> GuardrailResult:
            call_count["check1"] += 1
            return GuardrailResult(passed=True)

        def check2(state: PipelineState) -> GuardrailResult:
            call_count["check2"] += 1
            return GuardrailResult(
                passed=False, reason="check2 failed", fallback=State.ERROR
            )

        def check3(state: PipelineState) -> GuardrailResult:
            call_count["check3"] += 1
            return GuardrailResult(passed=True)

        composed = make_guardrail(check1, check2, check3)
        empty_state: PipelineState = {
            "current_state": "init",
            "proposed_next": "fetch",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = composed(empty_state)

        assert result.passed is False
        assert call_count["check1"] == 1
        assert call_count["check2"] == 1
        assert call_count["check3"] == 0  # Never called due to short-circuit

    def test_make_guardrail_all_pass(self):
        """Test that all-passing checks return passed=True."""
        def check1(state: PipelineState) -> GuardrailResult:
            return GuardrailResult(passed=True)

        def check2(state: PipelineState) -> GuardrailResult:
            return GuardrailResult(passed=True)

        composed = make_guardrail(check1, check2)
        empty_state: PipelineState = {
            "current_state": "init",
            "proposed_next": "fetch",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = composed(empty_state)
        assert result.passed is True


class TestCheckTransitionAllowed:
    """Tests for check_transition_allowed guardrail."""

    def test_valid_transition_passes(self):
        """Test that valid transition passes."""
        state: PipelineState = {
            "current_state": "init",
            "proposed_next": "fetch",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = check_transition_allowed(state)
        assert result.passed is True

    def test_invalid_transition_fails(self):
        """Test that invalid transition fails."""
        state: PipelineState = {
            "current_state": "fetch",
            "proposed_next": "store",  # Invalid: fetch -> store not allowed
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = check_transition_allowed(state)
        assert result.passed is False
        assert result.fallback == State.ERROR


class TestCheckRetryBudgetWithErrorType:
    """Tests for check_retry_budget_with_error_type guardrail."""

    def test_no_error_passes(self):
        """Test that no error passes."""
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
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = check_retry_budget_with_error_type(state)
        assert result.passed is True

    def test_transient_error_within_budget_passes(self):
        """Test that transient error within budget passes."""
        state: PipelineState = {
            "current_state": "validate",
            "proposed_next": "enrich",
            "retry_count": 1,
            "error_message": "Timeout",
            "error_type": "transient",
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = check_retry_budget_with_error_type(state)
        assert result.passed is True

    def test_transient_error_exceeds_budget_fails(self):
        """Test that transient error exceeding budget fails."""
        state: PipelineState = {
            "current_state": "validate",
            "proposed_next": "enrich",
            "retry_count": 4,  # > MAX_RETRIES (3)
            "error_message": "Timeout",
            "error_type": "transient",
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = check_retry_budget_with_error_type(state)
        assert result.passed is False
        assert result.fallback == State.ERROR

    def test_permanent_error_immediately_fails(self):
        """Test that permanent error immediately fails (no retry)."""
        state: PipelineState = {
            "current_state": "fetch",
            "proposed_next": "validate",
            "retry_count": 0,
            "error_message": "Document not found",
            "error_type": "permanent",
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = check_retry_budget_with_error_type(state)
        assert result.passed is False
        assert result.fallback == State.ERROR


class TestCheckDataPresent:
    """Tests for data presence checks."""

    def test_raw_data_present_passes(self):
        """Test that raw_data present passes."""
        state: PipelineState = {
            "current_state": "fetch",
            "proposed_next": "validate",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": {"content": "test"},
            "validated_data": None,
            "enriched_data": None,
        }
        result = check_raw_data_present(state)
        assert result.passed is True

    def test_raw_data_missing_fails(self):
        """Test that raw_data missing fails."""
        state: PipelineState = {
            "current_state": "fetch",
            "proposed_next": "validate",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = check_raw_data_present(state)
        assert result.passed is False

    def test_validated_data_present_passes(self):
        """Test that validated_data present passes."""
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
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": {"schema": "valid"},
            "enriched_data": None,
        }
        result = check_validated_data_present(state)
        assert result.passed is True


class TestCheckDocumentSize:
    """Tests for document size check."""

    def test_small_document_passes(self):
        """Test that small document passes."""
        import json

        state: PipelineState = {
            "current_state": "fetch",
            "proposed_next": "validate",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": {"content": "small"},
            "validated_data": None,
            "enriched_data": None,
        }
        result = check_document_size(state)
        assert result.passed is True

    def test_large_document_fails(self):
        """Test that large document (>10MB) fails."""
        state: PipelineState = {
            "current_state": "fetch",
            "proposed_next": "validate",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": {"content": "x" * (11 * 1024 * 1024)},  # 11MB
            "validated_data": None,
            "enriched_data": None,
        }
        result = check_document_size(state)
        assert result.passed is False
        assert result.fallback == State.ERROR


class TestCheckFallbackDepth:
    """Tests for fallback cascade detection."""

    def test_no_fallback_passes(self):
        """Test that no fallback (depth=0) passes."""
        state: PipelineState = {
            "current_state": "fetch",
            "proposed_next": "validate",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = check_fallback_depth(state)
        assert result.passed is True

    def test_one_fallback_passes(self):
        """Test that one fallback (depth=1) passes."""
        state: PipelineState = {
            "current_state": "validate",
            "proposed_next": "human_review",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 1,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = check_fallback_depth(state)
        assert result.passed is True

    def test_two_fallbacks_pass(self):
        """Test that two fallbacks (depth=2) pass."""
        state: PipelineState = {
            "current_state": "validate",
            "proposed_next": "human_review",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 2,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = check_fallback_depth(state)
        assert result.passed is True

    def test_three_fallbacks_fails(self):
        """Test that cascade >2 fails."""
        state: PipelineState = {
            "current_state": "validate",
            "proposed_next": "human_review",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 3,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = check_fallback_depth(state)
        assert result.passed is False
        assert result.fallback == State.ERROR


class TestCheckPipelineTimeout:
    """Tests for timeout check."""

    def test_within_timeout_passes(self):
        """Test that execution within timeout passes."""
        current_time = time.time()
        state: PipelineState = {
            "current_state": "fetch",
            "proposed_next": "validate",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": current_time - 30,  # 30 seconds ago
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = check_pipeline_timeout(state)
        assert result.passed is True

    def test_exceeds_timeout_fails(self):
        """Test that execution exceeding timeout fails."""
        current_time = time.time()
        state: PipelineState = {
            "current_state": "fetch",
            "proposed_next": "validate",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": current_time - 70,  # 70 seconds ago
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = check_pipeline_timeout(state)
        assert result.passed is False
        assert result.fallback == State.ERROR


class TestGuardrailsRegistry:
    """Tests for GUARDRAILS registry."""

    def test_guardrails_has_all_states(self):
        """Test that GUARDRAILS covers all 8 states."""
        assert len(GUARDRAILS) == 8
        for state in [
            State.FETCH,
            State.VALIDATE,
            State.ENRICH,
            State.STORE,
            State.COMPLETE,
            State.RETRY,
            State.HUMAN_REVIEW,
            State.ERROR,
        ]:
            assert state in GUARDRAILS

    def test_guardrail_function_callable(self):
        """Test that each guardrail is callable."""
        for state, guardrail_fn in GUARDRAILS.items():
            assert callable(guardrail_fn)

    def test_error_guardrail_always_passes(self):
        """Test that ERROR state guardrail always passes."""
        state: PipelineState = {
            "current_state": "error",
            "proposed_next": "error",
            "retry_count": 10,
            "error_message": "Critical error",
            "error_type": "permanent",
            "audit_trail": [],
            "fallback_depth": 10,
            "started_at": 0.0,
            "node_timeout_seconds": 1,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = GUARDRAILS[State.ERROR](state)
        assert result.passed is True
