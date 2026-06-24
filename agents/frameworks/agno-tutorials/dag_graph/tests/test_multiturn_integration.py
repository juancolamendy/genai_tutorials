"""
test_multiturn_integration.py
────────────────────────────────────────────────────────────────────────────
Integration tests for multi-turn conversation workflows.

Tests the complete flow:
  Turn 1: validate_turn_input → escape_for_llm → _prepare_turn_metadata
          → self.run() [semantic router + guardrail + handlers]
          → _trim_history → _build_turn_response

  Turn 2-N: Resume from DB, same flow

Tests semantic routing, entity extraction, intent detection, pause/resume.
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch

from src.workflow.workflow import DocPipelineWorkflow
from src.workflow.pipeline_state import PipelineState


class TestMultiTurnBasicFlow:
    """Test basic multi-turn conversation flow."""

    def test_process_turn_single_turn(self):
        """Test: Single turn processes without error."""
        wf = DocPipelineWorkflow(name="test", session_id="test_session")

        # Mock the LLM router to return a decision
        with patch.object(wf.router, 'route') as mock_route:
            mock_route.return_value = Mock(
                proposed_next="fetch",
                confidence=0.9,
                semantic_entities={"document_ids": ["doc_001"]},
                semantic_intents=["confirm"],
                reasoning="User confirmed document"
            )

            response = wf.process_turn(
                user_id="user_1",
                session_id="session_abc",
                turn_input="Process document DOC-001"
            )

        # Verify response structure
        assert "current_state" in response
        assert "waits_for_input" in response
        assert "turn_number" in response
        assert "semantic_context" in response
        assert "router_confidence" in response
        assert response["error"] is None

    def test_process_turn_extracts_entities(self):
        """Test: Router extracts entities from user input."""
        wf = DocPipelineWorkflow(name="test", session_id="test_session")

        with patch.object(wf.router, 'route') as mock_route:
            mock_route.return_value = Mock(
                proposed_next="validate",
                confidence=0.85,
                semantic_entities={"amounts": ["$1,200.00"], "document_ids": ["INV-001"]},
                semantic_intents=["confirm", "escalate"],
                reasoning="High-value invoice detected"
            )

            response = wf.process_turn(
                user_id="user_1",
                session_id="session_xyz",
                turn_input="Process invoice INV-001 for $1,200.00"
            )

        # Verify entities were extracted
        assert response["semantic_context"]["entities"]["amounts"] == ["$1,200.00"]
        assert response["semantic_context"]["entities"]["document_ids"] == ["INV-001"]
        assert response["semantic_context"]["intents"] == ["confirm", "escalate"]
        assert response["router_confidence"] == 0.85

    def test_process_turn_invalid_input_rejected(self):
        """Test: Invalid input is rejected before routing."""
        wf = DocPipelineWorkflow(name="test", session_id="test_session")

        # Input exceeds token limit (mock: > 2000 tokens ~ > 8000 chars)
        long_input = "x" * 10_001

        response = wf.process_turn(
            user_id="user_1",
            session_id="session_abc",
            turn_input=long_input
        )

        assert response["error"] is not None
        assert "exceeds" in response["error"].lower()
        assert response["current_state"] is None
        assert response["waits_for_input"] is False

    def test_process_turn_increment_turn_number(self):
        """Test: Turn number increments on each call."""
        wf = DocPipelineWorkflow(name="test", session_id="test_session")

        with patch.object(wf.router, 'route') as mock_route:
            mock_route.return_value = Mock(
                proposed_next="fetch",
                confidence=0.9,
                semantic_entities={},
                semantic_intents=[],
            )

            # Turn 1
            response1 = wf.process_turn(
                user_id="user_1",
                session_id="session_abc",
                turn_input="Turn 1 input"
            )
            assert response1["turn_number"] == 1

            # Turn 2 (would need to reload session from DB in real scenario)
            # For this test, we simulate by manually incrementing
            wf.session_state["turn_number"] = 1

            response2 = wf.process_turn(
                user_id="user_1",
                session_id="session_abc",
                turn_input="Turn 2 input"
            )
            assert response2["turn_number"] == 2

    def test_handler_error_routes_to_error_state(self):
        """Test: Handler exceptions route to ERROR state."""
        wf = DocPipelineWorkflow(name="test", session_id="test_session")

        # Mock router to propose fetch
        with patch.object(wf.router, 'route') as mock_route:
            mock_route.return_value = Mock(
                proposed_next="fetch",
                confidence=0.9,
                semantic_entities={},
                semantic_intents=[],
            )

            # Mock handler to raise exception
            with patch.dict(wf.HANDLER_MAP):

                def failing_fetch(p: PipelineState) -> PipelineState:
                    raise RuntimeError("Simulated fetch error")

                wf.HANDLER_MAP[wf._STATE_ENUM.FETCH] = failing_fetch

                response = wf.process_turn(
                    user_id="user_1",
                    session_id="session_abc",
                    turn_input="Process document"
                )

                # Should route to ERROR state on exception
                assert response["current_state"] == "error"
                assert response["error"] is not None


class TestMultiTurnHistoryManagement:
    """Test conversation history trimming and management."""

    def test_conversation_history_initialized(self):
        """Test: conversation_history field is initialized."""
        wf = DocPipelineWorkflow(name="test", session_id="test_session")

        with patch.object(wf.router, 'route') as mock_route:
            mock_route.return_value = Mock(
                proposed_next="fetch",
                confidence=0.9,
                semantic_entities={},
                semantic_intents=[],
            )

            wf.process_turn(
                user_id="user_1",
                session_id="session_abc",
                turn_input="Turn 1"
            )

            # Verify conversation_history exists
            assert "conversation_history" in wf.session_state
            assert isinstance(wf.session_state["conversation_history"], list)

    def test_conversation_history_trimmed_to_max(self):
        """Test: conversation_history is trimmed to max_history_turns."""
        wf = DocPipelineWorkflow(name="test", session_id="test_session")
        wf.session_state["max_history_turns"] = 3

        with patch.object(wf.router, 'route') as mock_route:
            mock_route.return_value = Mock(
                proposed_next="fetch",
                confidence=0.9,
                semantic_entities={},
                semantic_intents=[],
            )

            # Simulate 5 turns
            for i in range(5):
                wf.session_state["conversation_history"].append({
                    "role": "user",
                    "content": f"Turn {i+1} input",
                })
                wf._trim_history()

            # Should keep only last 3
            assert len(wf.session_state["conversation_history"]) <= 3


class TestSemanticRouting:
    """Test semantic router decision making."""

    def test_router_respects_allowed_states(self):
        """Test: Router proposed_next is validated against allowed_states."""
        wf = DocPipelineWorkflow(name="test", session_id="test_session")

        with patch.object(wf.router, 'route') as mock_route:
            # Mock router proposing an invalid state
            mock_route.return_value = Mock(
                proposed_next="invalid_state",
                confidence=0.9,
                semantic_entities={},
                semantic_intents=[],
            )

            # The _semantic_router_step should validate and fall back
            response = wf.process_turn(
                user_id="user_1",
                session_id="session_abc",
                turn_input="Some input"
            )

            # Should not crash, should handle gracefully
            assert response is not None

    def test_router_confidence_stored(self):
        """Test: Router confidence is stored in session_state."""
        wf = DocPipelineWorkflow(name="test", session_id="test_session")

        with patch.object(wf.router, 'route') as mock_route:
            mock_route.return_value = Mock(
                proposed_next="fetch",
                confidence=0.75,
                semantic_entities={},
                semantic_intents=[],
            )

            response = wf.process_turn(
                user_id="user_1",
                session_id="session_abc",
                turn_input="Some input"
            )

            assert response["router_confidence"] == 0.75


class TestHandlerMetadata:
    """Test handler metadata and waits_for_input."""

    def test_waits_for_input_metadata(self):
        """Test: Handler metadata waits_for_input is accessible."""
        from engine.handler_registry import does_state_wait_for_input

        # handle_human_review should have waits_for_input=True
        assert does_state_wait_for_input("human_review") is True

        # handle_fetch should have waits_for_input=False
        assert does_state_wait_for_input("fetch") is False

    def test_response_includes_waits_for_input(self):
        """Test: process_turn response includes waits_for_input flag."""
        wf = DocPipelineWorkflow(name="test", session_id="test_session")

        with patch.object(wf.router, 'route') as mock_route:
            mock_route.return_value = Mock(
                proposed_next="fetch",
                confidence=0.9,
                semantic_entities={},
                semantic_intents=[],
            )

            response = wf.process_turn(
                user_id="user_1",
                session_id="session_abc",
                turn_input="Process document"
            )

            assert "waits_for_input" in response
            assert isinstance(response["waits_for_input"], bool)


class TestBackwardCompatibility:
    """Test that one-turn process() still works."""

    def test_process_one_turn_unchanged(self):
        """Test: process() method still works for one-turn workflows."""
        wf = DocPipelineWorkflow(name="test", session_id="test_session")

        result = wf.process(document_id="DOC-001")

        # Should return PipelineState
        assert isinstance(result, dict)
        assert "current_state" in result
        assert "document_id" in result
        assert result["document_id"] == "DOC-001"

    def test_process_creates_pipeline_run_record(self):
        """Test: process() records completed runs."""
        wf = DocPipelineWorkflow(name="test", session_id="test_session")

        result = wf.process(document_id="DOC-001")

        # Should return a valid PipelineState
        assert result is not None
        assert "current_state" in result
        # Should have output field with run record
        assert "output" in wf.session_state or "pipeline_runs" in wf.session_state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
