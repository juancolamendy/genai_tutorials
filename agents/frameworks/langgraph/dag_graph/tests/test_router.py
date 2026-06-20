"""Tests for router node and timeout guards."""

import pytest
import time
from src.engine.state_machine import State, PipelineState
from src.engine.router import router, HAPPY_PATH


class TestHappyPath:
    """Tests for HAPPY_PATH routing table."""

    def test_happy_path_has_all_states(self):
        """Test that HAPPY_PATH covers main processing states."""
        assert State.INIT in HAPPY_PATH
        assert State.FETCH in HAPPY_PATH
        assert State.VALIDATE in HAPPY_PATH
        assert State.ENRICH in HAPPY_PATH
        assert State.STORE in HAPPY_PATH
        assert State.RETRY in HAPPY_PATH
        assert State.HUMAN_REVIEW in HAPPY_PATH

    def test_init_to_fetch(self):
        """Test INIT → FETCH routing."""
        assert HAPPY_PATH[State.INIT] == State.FETCH

    def test_fetch_to_validate(self):
        """Test FETCH → VALIDATE routing."""
        assert HAPPY_PATH[State.FETCH] == State.VALIDATE

    def test_validate_to_enrich(self):
        """Test VALIDATE → ENRICH routing."""
        assert HAPPY_PATH[State.VALIDATE] == State.ENRICH

    def test_enrich_to_store(self):
        """Test ENRICH → STORE routing."""
        assert HAPPY_PATH[State.ENRICH] == State.STORE

    def test_store_to_complete(self):
        """Test STORE → COMPLETE routing."""
        assert HAPPY_PATH[State.STORE] == State.COMPLETE

    def test_retry_to_fetch(self):
        """Test RETRY → FETCH routing (loop back)."""
        assert HAPPY_PATH[State.RETRY] == State.FETCH

    def test_human_review_to_enrich(self):
        """Test HUMAN_REVIEW → ENRICH routing."""
        assert HAPPY_PATH[State.HUMAN_REVIEW] == State.ENRICH


class TestRouterNode:
    """Tests for router() function."""

    def test_router_sets_proposed_next_from_routing_table(self):
        """Test that router looks up and sets proposed_next."""
        state: PipelineState = {
            "current_state": "init",
            "proposed_next": "",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": time.time(),
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = router(state)
        assert result["proposed_next"] == "fetch"

    def test_router_appends_to_audit_trail(self):
        """Test that router appends routing decision to audit trail."""
        state: PipelineState = {
            "current_state": "init",
            "proposed_next": "",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": ["init"],
            "fallback_depth": 0,
            "started_at": time.time(),
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = router(state)
        assert len(result["audit_trail"]) == 2
        assert "router:" in result["audit_trail"][-1]
        assert "init" in result["audit_trail"][-1]
        assert "fetch" in result["audit_trail"][-1]

    def test_router_returns_unchanged_state_except_proposed_next(self):
        """Test that router only modifies proposed_next and audit_trail."""
        state: PipelineState = {
            "current_state": "fetch",
            "proposed_next": "unknown",
            "retry_count": 5,
            "error_message": "Old error",
            "error_type": "transient",
            "audit_trail": ["init"],
            "fallback_depth": 2,
            "started_at": 123.0,
            "node_timeout_seconds": 60,
            "document_id": "DOC-123",
            "raw_data": {"key": "value"},
            "validated_data": None,
            "enriched_data": None,
        }
        result = router(state)

        # Check unchanged fields
        assert result["current_state"] == "fetch"
        assert result["retry_count"] == 5
        assert result["error_message"] == "Old error"
        assert result["error_type"] == "transient"
        assert result["fallback_depth"] == 2
        assert result["started_at"] == 123.0
        assert result["node_timeout_seconds"] == 60
        assert result["document_id"] == "DOC-123"
        assert result["raw_data"] == {"key": "value"}
        assert result["validated_data"] is None
        assert result["enriched_data"] is None

        # Check modified fields
        assert result["proposed_next"] == "validate"
        assert len(result["audit_trail"]) == 2

    def test_router_fetch_to_validate(self):
        """Test FETCH → VALIDATE routing."""
        state: PipelineState = {
            "current_state": "fetch",
            "proposed_next": "",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": time.time(),
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = router(state)
        assert result["proposed_next"] == "validate"

    def test_router_validate_to_enrich(self):
        """Test VALIDATE → ENRICH routing."""
        state: PipelineState = {
            "current_state": "validate",
            "proposed_next": "",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": time.time(),
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = router(state)
        assert result["proposed_next"] == "enrich"

    def test_router_enrich_to_store(self):
        """Test ENRICH → STORE routing."""
        state: PipelineState = {
            "current_state": "enrich",
            "proposed_next": "",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": time.time(),
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = router(state)
        assert result["proposed_next"] == "store"

    def test_router_store_to_complete(self):
        """Test STORE → COMPLETE routing."""
        state: PipelineState = {
            "current_state": "store",
            "proposed_next": "",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": time.time(),
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = router(state)
        assert result["proposed_next"] == "complete"

    def test_router_retry_to_fetch(self):
        """Test RETRY → FETCH routing (loop back)."""
        state: PipelineState = {
            "current_state": "retry",
            "proposed_next": "",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": time.time(),
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = router(state)
        assert result["proposed_next"] == "fetch"

    def test_router_human_review_to_enrich(self):
        """Test HUMAN_REVIEW → ENRICH routing."""
        state: PipelineState = {
            "current_state": "human_review",
            "proposed_next": "",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": time.time(),
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = router(state)
        assert result["proposed_next"] == "enrich"

    def test_router_with_complete_state(self):
        """Test router behavior when in COMPLETE state (terminal)."""
        state: PipelineState = {
            "current_state": "complete",
            "proposed_next": "",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": time.time(),
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = router(state)
        # COMPLETE is terminal, router should default to ERROR
        assert result["proposed_next"] == "error"

    def test_router_with_error_state(self):
        """Test router behavior when in ERROR state (terminal)."""
        state: PipelineState = {
            "current_state": "error",
            "proposed_next": "",
            "retry_count": 0,
            "error_message": "Fatal error",
            "error_type": "permanent",
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": time.time(),
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = router(state)
        # ERROR is terminal, router should default to ERROR
        assert result["proposed_next"] == "error"

    def test_router_preserves_state_immutability(self):
        """Test that router returns new dict, doesn't mutate input."""
        original_state: PipelineState = {
            "current_state": "init",
            "proposed_next": "original",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": ["original"],
            "fallback_depth": 0,
            "started_at": time.time(),
            "node_timeout_seconds": 60,
            "document_id": "TEST",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
        result = router(original_state)

        # Original should not be modified
        assert original_state["proposed_next"] == "original"
        assert len(original_state["audit_trail"]) == 1

        # Result should be different
        assert result["proposed_next"] == "fetch"
        assert len(result["audit_trail"]) == 2
