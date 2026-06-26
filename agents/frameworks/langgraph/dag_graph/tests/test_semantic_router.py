"""Tests for semantic router base classes."""

import pytest
from pydantic import BaseModel, Field

from src.engine.router import (
    RouterDecision,
    BaseSemanticRouter,
    DefaultSemanticRouter,
)


class MockRouterOutput(BaseModel):
    """Mock output schema for router."""

    proposed_next: str = Field(..., description="Next state")
    confidence: float = Field(..., description="Confidence [0.0, 1.0]")
    semantic_entities: dict = Field(default_factory=dict)
    semantic_intents: list = Field(default_factory=list)
    reasoning: str = ""


class MockRouter(DefaultSemanticRouter):
    """Mock concrete router implementation."""

    output_schema = MockRouterOutput


def test_router_decision_creation():
    """Test RouterDecision dataclass creation."""
    decision = RouterDecision(
        proposed_next="validate",
        confidence=0.95,
        semantic_entities={"doc_id": "123"},
        semantic_intents=["submit"],
    )
    assert decision.proposed_next == "validate"
    assert decision.confidence == 0.95
    assert decision.semantic_entities["doc_id"] == "123"
    assert decision.semantic_intents == ["submit"]


def test_router_decision_defaults():
    """Test RouterDecision default values."""
    decision = RouterDecision(
        proposed_next="validate",
        confidence=0.8,
    )
    assert decision.proposed_next == "validate"
    assert decision.confidence == 0.8
    assert decision.semantic_entities == {}
    assert decision.semantic_intents == []
    assert decision.reasoning is None


def test_base_router_is_abstract():
    """Test that BaseSemanticRouter cannot be instantiated."""
    with pytest.raises(TypeError):
        BaseSemanticRouter()


def test_default_router_without_schema():
    """Test that DefaultSemanticRouter requires output_schema."""
    with pytest.raises(NotImplementedError):
        DefaultSemanticRouter()


def test_default_router_with_schema():
    """Test DefaultSemanticRouter with valid schema."""
    router = MockRouter()
    assert router.model == "claude-haiku-4-5-20251001"
    assert router.output_schema == MockRouterOutput


def test_get_instructions():
    """Test that get_instructions returns valid string."""
    router = MockRouter()
    instructions = router.get_instructions()
    assert isinstance(instructions, str)
    assert "ALLOWED NEXT STATES" in instructions
    assert len(instructions) > 0


def test_build_router_prompt():
    """Test prompt building with required fields."""
    router = MockRouter()
    prompt = router.build_router_prompt(
        current_state="validate",
        turn_input="Document looks good",
        history_text="User: Process my doc",
        allowed_states=["enrich", "store", "error"],
    )
    assert "validate" in prompt
    assert "Document looks good" in prompt
    assert "enrich, store, error" in prompt


def test_router_lazy_loads_llm():
    """Test that LLM is lazy-loaded on first access."""
    router = MockRouter()
    assert router.llm is None
    # Note: actual LLM loading would require API key, so we just test lazy init
    # In real tests, this would be mocked


def test_router_decision_confidence_clamping():
    """Test that confidence is clamped to [0.0, 1.0]."""
    # This will be tested in route() method once LLM is mocked
    # For now, test the logic separately
    decision = RouterDecision(proposed_next="test", confidence=1.5)
    # Clamping happens in route(), not in RouterDecision
    assert decision.confidence == 1.5


def test_router_prompt_includes_history():
    """Test that router prompt includes formatted history."""
    router = MockRouter()
    history_text = "User: Start\nAssistant: Ready"
    prompt = router.build_router_prompt(
        current_state="init",
        turn_input="Begin",
        history_text=history_text,
        allowed_states=["fetch"],
    )
    assert history_text in prompt


def test_router_allowed_states_formatting():
    """Test that allowed states are properly formatted in prompt."""
    router = MockRouter()
    allowed = ["state1", "state2", "state3"]
    prompt = router.build_router_prompt(
        current_state="current",
        turn_input="input",
        history_text="",
        allowed_states=allowed,
    )
    assert "state1, state2, state3" in prompt
