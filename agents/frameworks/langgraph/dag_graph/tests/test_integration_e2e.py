"""End-to-end integration tests for complete LangGraph state machine."""

import pytest
import time
from src.engine.state_machine import State, PipelineState
from src.engine.router import router
from src.engine.guardrail import GUARDRAILS
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


HANDLER_MAP = {
    State.FETCH: handle_fetch_template,
    State.VALIDATE: handle_validate_template,
    State.ENRICH: handle_enrich_template,
    State.STORE: handle_store_template,
    State.RETRY: handle_retry_template,
    State.HUMAN_REVIEW: handle_human_review_template,
    State.COMPLETE: handle_complete_template,
    State.ERROR: handle_error_template,
}

TERMINAL_STATES = {State.COMPLETE, State.ERROR}


def simulate_one_turn(state: PipelineState) -> PipelineState:
    """Simulate one turn of router → guardrail → handler → router."""
    current = State(state["current_state"])

    # 1. Router: propose next state
    state = router(state)

    # 2. Guardrail: validate and potentially redirect
    # Only run guardrail for non-terminal, non-init states
    if current not in {State.INIT} and current in GUARDRAILS:
        guardrail_fn = GUARDRAILS[current]
        result = guardrail_fn(state)
        if not result.passed:
            state = {
                **state,
                "proposed_next": result.fallback.value if result.fallback else "error",
                "error_message": result.reason,
                "audit_trail": state["audit_trail"]
                + [f"guardrail FAIL → {result.fallback.value if result.fallback else 'error'} ({result.reason})"],
                "fallback_depth": state.get("fallback_depth", 0) + 1,
            }
        else:
            state["audit_trail"].append(f"guardrail PASS → {state['proposed_next']}")
            state["fallback_depth"] = 0
    elif current == State.INIT:
        # INIT skips guardrail, just proceed
        state["audit_trail"].append(f"guardrail SKIP (init) → {state['proposed_next']}")

    # 3. Handler: execute business logic
    next_state = State(state["proposed_next"])
    if next_state in HANDLER_MAP:
        state = HANDLER_MAP[next_state](state)

    return state


class TestHappyPath:
    """Test: document successfully processes INIT → COMPLETE."""

    def test_happy_path_init_to_complete(self):
        """Test happy path: INIT → FETCH → VALIDATE → ENRICH → STORE → COMPLETE."""
        state: PipelineState = {
            "current_state": "init",
            "proposed_next": "fetch",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": ["init"],
            "fallback_depth": 0,
            "started_at": time.time(),
            "node_timeout_seconds": 60,
            "document_id": "TEST-001",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }

        steps = 0
        max_steps = 20
        while state["current_state"] not in ["complete", "error"] and steps < max_steps:
            state = simulate_one_turn(state)
            steps += 1

        assert state["current_state"] == "complete"
        assert steps < max_steps  # Should complete quickly
        assert len(state["audit_trail"]) > 5  # Multiple steps recorded

    def test_audit_trail_completeness(self):
        """Test that audit trail captures all routing and guardrail decisions."""
        state: PipelineState = {
            "current_state": "init",
            "proposed_next": "fetch",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": ["init"],
            "fallback_depth": 0,
            "started_at": time.time(),
            "node_timeout_seconds": 60,
            "document_id": "TEST-001",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }

        steps = 0
        max_steps = 20
        while state["current_state"] not in ["complete", "error"] and steps < max_steps:
            state = simulate_one_turn(state)
            steps += 1

        # Verify audit trail has router and guardrail entries
        audit_str = " ".join(state["audit_trail"])
        assert "router:" in audit_str
        assert "guardrail" in audit_str


class TestErrorRecovery:
    """Test: transient errors trigger RETRY recovery."""

    def test_transient_error_recovery(self):
        """Test that transient errors allow retries."""
        state: PipelineState = {
            "current_state": "validate",
            "proposed_next": "enrich",
            "retry_count": 0,
            "error_message": "Timeout",
            "error_type": "transient",
            "audit_trail": ["init"],
            "fallback_depth": 0,
            "started_at": time.time(),
            "node_timeout_seconds": 60,
            "document_id": "TEST-001",
            "raw_data": {"content": "test"},
            "validated_data": {"schema": "valid"},
            "enriched_data": None,
        }

        # Guardrail should allow transient error
        guardrail_fn = GUARDRAILS[State.VALIDATE]
        result = guardrail_fn(state)

        # Transient error within budget should pass
        assert result.passed is True


class TestRetryBudget:
    """Test: retry budget enforcement prevents infinite loops."""

    def test_retry_budget_exhaustion(self):
        """Test that retry_count > MAX_RETRIES causes ERROR."""
        state: PipelineState = {
            "current_state": "fetch",
            "proposed_next": "validate",
            "retry_count": 4,  # > MAX_RETRIES (3)
            "error_message": "Timeout",
            "error_type": "transient",
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": time.time(),
            "node_timeout_seconds": 60,
            "document_id": "TEST-001",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }

        guardrail_fn = GUARDRAILS[State.FETCH]
        result = guardrail_fn(state)

        assert result.passed is False
        assert result.fallback == State.ERROR


class TestPermanentErrors:
    """Test: permanent errors skip retries and go to ERROR."""

    def test_permanent_error_immediate_failure(self):
        """Test that permanent errors immediately route to ERROR."""
        state: PipelineState = {
            "current_state": "fetch",
            "proposed_next": "validate",
            "retry_count": 0,
            "error_message": "Document not found",
            "error_type": "permanent",
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": time.time(),
            "node_timeout_seconds": 60,
            "document_id": "MISSING",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }

        guardrail_fn = GUARDRAILS[State.FETCH]
        result = guardrail_fn(state)

        assert result.passed is False
        assert result.fallback == State.ERROR


class TestTimeoutEnforcement:
    """Test: pipeline timeout guard prevents indefinite execution."""

    def test_timeout_detection(self):
        """Test that elapsed > timeout triggers ERROR."""
        state: PipelineState = {
            "current_state": "fetch",
            "proposed_next": "validate",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": time.time() - 70,  # 70 seconds ago
            "node_timeout_seconds": 60,
            "document_id": "TEST-001",
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }

        guardrail_fn = GUARDRAILS[State.FETCH]
        result = guardrail_fn(state)

        assert result.passed is False
        assert result.fallback == State.ERROR


class TestCascadeDetection:
    """Test: fallback cascade detection prevents loops."""

    def test_cascade_loop_prevention(self):
        """Test that fallback_depth > 2 routes to ERROR."""
        state: PipelineState = {
            "current_state": "validate",
            "proposed_next": "enrich",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 3,  # > MAX_DEPTH (2)
            "started_at": time.time(),
            "node_timeout_seconds": 60,
            "document_id": "TEST-001",
            "raw_data": {"content": "test"},
            "validated_data": None,
            "enriched_data": None,
        }

        guardrail_fn = GUARDRAILS[State.VALIDATE]
        result = guardrail_fn(state)

        assert result.passed is False
        assert result.fallback == State.ERROR


class TestDocumentSizeValidation:
    """Test: oversized documents are rejected."""

    def test_document_size_rejection(self):
        """Test that documents > 10MB are rejected."""
        state: PipelineState = {
            "current_state": "fetch",
            "proposed_next": "validate",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": time.time(),
            "node_timeout_seconds": 60,
            "document_id": "TEST-001",
            "raw_data": {"huge": "x" * (11 * 1024 * 1024)},  # 11MB
            "validated_data": None,
            "enriched_data": None,
        }

        guardrail_fn = GUARDRAILS[State.VALIDATE]
        result = guardrail_fn(state)

        assert result.passed is False
        assert result.fallback == State.ERROR


class TestRetryIncrement:
    """Test: RETRY handler increments exactly by 1."""

    def test_retry_count_increment_safety(self):
        """Test that RETRY increments retry_count by exactly 1."""
        state: PipelineState = {
            "current_state": "retry",
            "proposed_next": "fetch",
            "retry_count": 2,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": time.time(),
            "node_timeout_seconds": 60,
            "document_id": "TEST-001",
            "raw_data": {"stale": "data"},
            "validated_data": None,
            "enriched_data": None,
        }

        result = handle_retry_template(state)

        assert result["retry_count"] == 3
        assert result["raw_data"] is None  # Stale data cleared


class TestDataPresenceValidation:
    """Test: data presence checks prevent downstream failures."""

    def test_raw_data_required_for_validate(self):
        """Test that VALIDATE requires raw_data."""
        state: PipelineState = {
            "current_state": "fetch",
            "proposed_next": "validate",
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": [],
            "fallback_depth": 0,
            "started_at": time.time(),
            "node_timeout_seconds": 60,
            "document_id": "TEST-001",
            "raw_data": None,  # Missing!
            "validated_data": None,
            "enriched_data": None,
        }

        guardrail_fn = GUARDRAILS[State.VALIDATE]
        result = guardrail_fn(state)

        assert result.passed is False
        assert result.fallback == State.RETRY
