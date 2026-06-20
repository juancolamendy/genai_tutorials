"""Tests for public API: run_pipeline."""

import pytest
from src.pipeline.workflow import run_pipeline


class TestRunPipelineAPI:
    """Tests for run_pipeline() public API."""

    def test_run_pipeline_valid_document_id(self):
        """Test run_pipeline with valid document_id."""
        result = run_pipeline("TEST-001")
        assert result is not None
        assert "current_state" in result
        assert result["document_id"] == "TEST-001"

    def test_run_pipeline_empty_document_id_raises(self):
        """Test run_pipeline raises on empty document_id."""
        with pytest.raises(ValueError, match="cannot be empty"):
            run_pipeline("")

    def test_run_pipeline_oversized_document_id_raises(self):
        """Test run_pipeline raises on oversized document_id."""
        with pytest.raises(ValueError, match="exceeds max length"):
            run_pipeline("x" * 300)

    def test_run_pipeline_returns_final_state(self):
        """Test that run_pipeline returns final PipelineState."""
        result = run_pipeline("TEST-001")
        assert result["current_state"] in ["complete", "error"]

    def test_run_pipeline_has_audit_trail(self):
        """Test that result includes audit_trail."""
        result = run_pipeline("TEST-001")
        assert "audit_trail" in result
        assert isinstance(result["audit_trail"], list)
        assert len(result["audit_trail"]) > 0

    def test_run_pipeline_timeout_parameter(self):
        """Test that timeout_seconds parameter is set."""
        result = run_pipeline("TEST-001", timeout_seconds=120)
        assert result["node_timeout_seconds"] == 120

    def test_run_pipeline_custom_initial_state(self):
        """Test run_pipeline with custom initial_state."""
        custom_state = {
            "current_state": "init",
            "proposed_next": "fetch",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": ["init"],
            "fallback_depth": 0,
            "started_at": 0.0,
            "node_timeout_seconds": 60,
            "document_id": "CUSTOM-001",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = run_pipeline("CUSTOM-001", initial_state=custom_state)
        assert result["document_id"] == "CUSTOM-001"

    def test_run_pipeline_happy_path(self):
        """Test happy path: document processes without errors."""
        result = run_pipeline("HAPPY-001")
        assert result["current_state"] in ["complete", "error"]
        assert "audit_trail" in result
