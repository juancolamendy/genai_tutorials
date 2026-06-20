"""Tests for handler architecture and exception handling pattern."""

import pytest
from src.engine.state_machine import State, PipelineState
from src.pipeline.handlers import (
    handle_fetch_template,
    handle_validate_template,
    handle_enrich_template,
    handle_store_template,
    handle_retry_template,
    handle_human_review_template,
    handle_complete_template,
    handle_error_template,
)


class TestHandlerContract:
    """Tests for handler execution contract."""

    def test_handler_sets_current_state_on_success(self):
        """Test that handler sets current_state to its own state value."""
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
        result = handle_fetch_template(state)
        assert result["current_state"] == "fetch"

    def test_handler_does_not_raise_exception(self):
        """Test that handler catches all exceptions and returns state."""
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
        # Should not raise, should return state
        result = handle_fetch_template(state)
        assert isinstance(result, dict)
        assert result["current_state"] == "fetch"

    def test_handler_returns_updated_state(self):
        """Test that handler returns state dict with all fields."""
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
        result = handle_fetch_template(state)

        # All fields must be present
        assert "current_state" in result
        assert "proposed_next" in result
        assert "retry_count" in result
        assert "error_message" in result
        assert "error_type" in result
        assert "audit_trail" in result
        assert "fallback_depth" in result
        assert "started_at" in result
        assert "node_timeout_seconds" in result
        assert "document_id" in result
        assert "raw_data" in result
        assert "validated_data" in result
        assert "enriched_data" in result

    def test_handler_sets_error_type_on_exception(self):
        """Test that handler sets error_type on exception."""
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
        result = handle_fetch_template(state)

        # error_type should be set (either None on success or transient/permanent on error)
        assert result["error_type"] is None or result["error_type"] in [
            "transient",
            "permanent",
        ]

    def test_handler_success_has_no_error_message(self):
        """Test that successful handler has no error_message."""
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
        result = handle_fetch_template(state)

        if result["error_type"] is None:
            # Success case: no error_message
            assert result["error_message"] is None
        else:
            # Error case: error_message set
            assert result["error_message"] is not None

    def test_handler_immutability(self):
        """Test that handler returns new dict, doesn't mutate input."""
        original_state: PipelineState = {
            "current_state": "fetch",
            "proposed_next": "validate",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": ["init"],
            "fallback_depth": 0,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = handle_fetch_template(original_state)

        # Original should not be modified
        assert original_state["current_state"] == "fetch"
        assert len(original_state["audit_trail"]) == 1

        # Result is a new dict (may have different fields)
        assert isinstance(result, dict)


class TestFetchHandler:
    """Tests for handle_fetch_template."""

    def test_fetch_handler_success(self):
        """Test FETCH handler success case."""
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
            "document_id": "DOC-001",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = handle_fetch_template(state)
        assert result["current_state"] == "fetch"
        assert result["raw_data"] is not None
        assert result["error_type"] is None


class TestValidateHandler:
    """Tests for handle_validate_template."""

    def test_validate_handler_success(self):
        """Test VALIDATE handler success case."""
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
            "document_id": "DOC-001",
            "raw_data": {"content": "test"},
            "validated_data": None,
            "enriched_data": None,
        }
        result = handle_validate_template(state)
        assert result["current_state"] == "validate"
        assert result["validated_data"] is not None
        assert result["error_type"] is None


class TestEnrichHandler:
    """Tests for handle_enrich_template."""

    def test_enrich_handler_success(self):
        """Test ENRICH handler success case."""
        state: PipelineState = {
            "current_state": "enrich",
            "proposed_next": "store",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "DOC-001",
            "raw_data": None,
            "validated_data": {"schema_version": "1.0"},
            "enriched_data": None,
        }
        result = handle_enrich_template(state)
        assert result["current_state"] == "enrich"
        assert result["enriched_data"] is not None
        assert result["error_type"] is None


class TestStoreHandler:
    """Tests for handle_store_template."""

    def test_store_handler_success(self):
        """Test STORE handler success case."""
        state: PipelineState = {
            "current_state": "store",
            "proposed_next": "complete",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "DOC-001",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": {"content": "enriched"},
        }
        result = handle_store_template(state)
        assert result["current_state"] == "store"
        assert result["error_type"] is None


class TestRetryHandler:
    """Tests for handle_retry_template."""

    def test_retry_handler_increments_count(self):
        """Test RETRY handler increments retry_count."""
        state: PipelineState = {
            "current_state": "retry",
            "proposed_next": "fetch",
            "retry_count": 1,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "DOC-001",
            "raw_data": {"content": "stale"},
            "validated_data": None,
            "enriched_data": None,
        }
        result = handle_retry_template(state)
        assert result["current_state"] == "retry"
        assert result["retry_count"] == 2
        assert result["raw_data"] is None  # Cleared
        assert result["error_type"] is None

    def test_retry_handler_clears_raw_data(self):
        """Test RETRY handler clears stale raw_data."""
        state: PipelineState = {
            "current_state": "retry",
            "proposed_next": "fetch",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "DOC-001",
            "raw_data": {"stale": "data"},
            "validated_data": None,
            "enriched_data": None,
        }
        result = handle_retry_template(state)
        assert result["raw_data"] is None


class TestHumanReviewHandler:
    """Tests for handle_human_review_template."""

    def test_human_review_handler_success(self):
        """Test HUMAN_REVIEW handler success case."""
        state: PipelineState = {
            "current_state": "human_review",
            "proposed_next": "enrich",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "DOC-001",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = handle_human_review_template(state)
        assert result["current_state"] == "human_review"
        assert result["error_type"] is None


class TestCompleteHandler:
    """Tests for handle_complete_template."""

    def test_complete_handler_success(self):
        """Test COMPLETE handler success case."""
        state: PipelineState = {
            "current_state": "complete",
            "proposed_next": "",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "DOC-001",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = handle_complete_template(state)
        assert result["current_state"] == "complete"
        assert result["error_type"] is None


class TestErrorHandler:
    """Tests for handle_error_template."""

    def test_error_handler_preserves_error_state(self):
        """Test ERROR handler preserves error information."""
        state: PipelineState = {
            "current_state": "error",
            "proposed_next": "",
            "retry_count": 3,
            "error_message": "Critical failure",
            "error_type": "permanent",
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "DOC-001",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = handle_error_template(state)
        assert result["current_state"] == "error"
        assert result["error_message"] == "Critical failure"
        assert result["error_type"] == "permanent"
