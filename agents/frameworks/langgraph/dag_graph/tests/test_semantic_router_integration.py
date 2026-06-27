"""Tests for semantic router integration in graph."""

import pytest

from src.engine.router import DefaultSemanticRouter
from src.workflow.graph import DocumentPipelineGraph
from src.workflow.pipeline_state import new_pipeline
from src.workflow.state_machine import State


class MockSemanticRouter:
    """Mock semantic router for testing."""

    def route(self, state):
        """Return a mock router decision."""
        from src.workflow.router import DocRouterOutput

        current = state.get("current_state", State.INIT.value)

        # Simple mock: INIT → FETCH
        if current == State.INIT.value:
            next_state = State.FETCH
        else:
            next_state = State.VALIDATE

        return DocRouterOutput(
            proposed_next=next_state,
            confidence=0.95,
            semantic_entities={"doc_type": "invoice"},
            semantic_intents=["process", "validate"],
            reasoning="Mock routing decision",
        )


def test_semantic_router_integration():
    """Test that semantic router is used when available."""
    mock_router = MockSemanticRouter()
    graph = DocumentPipelineGraph(semantic_router=mock_router)

    state = new_pipeline("doc-001")
    state["current_state"] = State.INIT.value

    # Router node should use semantic router
    result = graph._router_node(state)

    assert result["proposed_next"] == State.FETCH.value
    assert result["router_confidence"] == 0.95
    assert result["semantic_context"]["entities"]["doc_type"] == "invoice"
    assert "process" in result["semantic_context"]["intents"]


def test_semantic_router_fallback_on_error():
    """Test that code router is used if semantic router fails."""

    class FailingSemanticRouter:
        def route(self, state):
            raise RuntimeError("Router failed")

    failing_router = FailingSemanticRouter()
    graph = DocumentPipelineGraph(semantic_router=failing_router)

    state = new_pipeline("doc-001")
    state["current_state"] = State.INIT.value

    # Should fall back to code routing
    result = graph._router_node(state)

    # Code router should route INIT → FETCH
    assert result["proposed_next"] == State.FETCH.value


def test_code_router_used_without_semantic_router():
    """Test that code router is used when no semantic router is set."""
    graph = DocumentPipelineGraph()  # No semantic router
    assert graph.semantic_router is None

    state = new_pipeline("doc-001")
    state["current_state"] = State.INIT.value

    # Should use code router
    result = graph._router_node(state)

    # Code router should route INIT → FETCH
    assert result["proposed_next"] == State.FETCH.value
    # Code router doesn't set semantic_context or router_confidence
    # (they come from state initialization, not from router)
    assert "router_confidence" not in result or result.get("router_confidence") == 0.0


def test_set_semantic_router_after_init():
    """Test that semantic router can be set after initialization."""
    graph = DocumentPipelineGraph()
    assert graph.semantic_router is None

    # Set router after initialization
    mock_router = MockSemanticRouter()
    graph.set_semantic_router(mock_router)

    assert graph.semantic_router is not None

    state = new_pipeline("doc-001")
    state["current_state"] = State.INIT.value

    result = graph._router_node(state)

    assert result["proposed_next"] == State.FETCH.value
    assert result["router_confidence"] == 0.95


def test_semantic_router_reasoning_stored():
    """Test that router reasoning is stored in state."""

    class ReasoningRouter:
        def route(self, state):
            from src.workflow.router import DocRouterOutput

            return DocRouterOutput(
                proposed_next=State.FETCH,
                confidence=0.9,
                semantic_entities={},
                semantic_intents=[],
                reasoning="Document ready for processing",
            )

    router = ReasoningRouter()
    graph = DocumentPipelineGraph(semantic_router=router)

    state = new_pipeline("doc-001")
    state["current_state"] = State.INIT.value

    result = graph._router_node(state)

    assert "router_reasoning" in result
    assert result["router_reasoning"] == "Document ready for processing"


def test_semantic_router_audit_trail():
    """Test that semantic routing decisions are recorded in audit trail."""
    mock_router = MockSemanticRouter()
    graph = DocumentPipelineGraph(semantic_router=mock_router)

    state = new_pipeline("doc-001")
    state["current_state"] = State.INIT.value

    result = graph._router_node(state)

    # Check audit trail mentions semantic routing
    audit = result["audit_trail"]
    assert any("semantic" in entry for entry in audit)


def test_code_router_audit_trail():
    """Test that code routing decisions are recorded in audit trail."""
    graph = DocumentPipelineGraph()

    state = new_pipeline("doc-001")
    state["current_state"] = State.INIT.value

    result = graph._router_node(state)

    # Check audit trail mentions code routing
    audit = result["audit_trail"]
    assert any("code" in entry for entry in audit)
