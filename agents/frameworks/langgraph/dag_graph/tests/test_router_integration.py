"""Tests for semantic router integration with graph routing."""


from src.workflow.router import DocPipelineRouter, DocRouterOutput


def test_router_decision_structure():
    """Test that router decision has all required fields."""
    # Test the output schema
    output = DocRouterOutput(
        proposed_next="validate",
        confidence=0.95,
        semantic_entities={"doc_id": "123"},
        semantic_intents=["submit"],
        reasoning="User provided valid input",
    )

    assert output.proposed_next == "validate"
    assert output.confidence == 0.95
    assert output.semantic_entities["doc_id"] == "123"


def test_router_can_be_initialized():
    """Test that router initializes correctly."""
    router = DocPipelineRouter()
    assert router.model == "claude-haiku-4-5-20251001"


def test_router_with_custom_model():
    """Test router initialization with custom model."""
    router = DocPipelineRouter(model="claude-opus-4-8")
    assert router.model == "claude-opus-4-8"


def test_router_get_instructions_not_empty():
    """Test that router provides instructions."""
    router = DocPipelineRouter()
    instructions = router.get_instructions()
    assert len(instructions) > 0
    assert "state" in instructions.lower()


def test_router_build_prompt_structure():
    """Test that router builds valid prompts."""
    router = DocPipelineRouter()
    prompt = router.build_router_prompt(
        current_state="validate",
        turn_input="Document looks good",
        history_text="Prior: Started processing",
        allowed_states=["enrich", "error"],
    )

    assert "validate" in prompt
    assert "Document looks good" in prompt
    assert "enrich, error" in prompt


def test_router_prompt_with_empty_history():
    """Test router prompt with no history."""
    router = DocPipelineRouter()
    prompt = router.build_router_prompt(
        current_state="init",
        turn_input="Start processing",
        history_text="",
        allowed_states=["fetch"],
    )

    assert "init" in prompt
    assert "Start processing" in prompt


def test_router_prompt_with_long_history():
    """Test router prompt with conversation history."""
    router = DocPipelineRouter()
    history = "\n".join([
        "User: Request 1",
        "Assistant: Response 1",
        "User: Request 2",
        "Assistant: Response 2",
    ])
    prompt = router.build_router_prompt(
        current_state="validate",
        turn_input="Continue",
        history_text=history,
        allowed_states=["enrich"],
    )

    assert history in prompt


def test_state_routing_with_allowed_states():
    """Test that routing respects allowed states."""
    router = DocPipelineRouter()

    # Verify allowed states are properly formatted
    allowed = ["validate", "enrich", "error"]
    allowed_str = ", ".join(allowed)

    prompt = router.build_router_prompt(
        current_state="init",
        turn_input="test",
        history_text="",
        allowed_states=allowed,
    )

    assert allowed_str in prompt


def test_document_router_specific_instructions():
    """Test that DocPipelineRouter has document-specific instructions."""
    router = DocPipelineRouter()
    instructions = router.get_instructions()

    # Should mention document states
    assert "validate" in instructions.lower()
    assert "enrich" in instructions.lower()
    assert "store" in instructions.lower()


def test_router_instructions_include_constraints():
    """Test that router instructions include important rules."""
    router = DocPipelineRouter()
    instructions = router.get_instructions()

    assert "IMPORTANT RULES" in instructions
    assert "ALLOWED NEXT STATES" in instructions


def test_multiple_routers_independent():
    """Test that multiple router instances are independent."""
    router1 = DocPipelineRouter()
    router2 = DocPipelineRouter(model="claude-opus-4-8")

    assert router1.model == "claude-haiku-4-5-20251001"
    assert router2.model == "claude-opus-4-8"
