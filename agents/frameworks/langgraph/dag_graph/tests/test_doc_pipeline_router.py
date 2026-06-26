"""Tests for document pipeline router."""

from src.workflow.router import DocPipelineRouter, DocRouterOutput


def test_doc_router_output_schema():
    """Test that DocRouterOutput is properly defined."""
    output = DocRouterOutput(
        proposed_next="validate",
        confidence=0.95,
        semantic_entities={"doc_id": "123"},
        semantic_intents=["submit"],
        reasoning="User provided valid document",
    )
    assert output.proposed_next == "validate"
    assert output.confidence == 0.95
    assert output.semantic_entities["doc_id"] == "123"
    assert "submit" in output.semantic_intents
    assert output.reasoning == "User provided valid document"


def test_doc_router_output_defaults():
    """Test DocRouterOutput default values."""
    output = DocRouterOutput(
        proposed_next="validate",
        confidence=0.8,
    )
    assert output.proposed_next == "validate"
    assert output.confidence == 0.8
    assert output.semantic_entities == {}
    assert output.semantic_intents == []
    assert output.reasoning == ""


def test_doc_router_initialization():
    """Test that DocPipelineRouter initializes correctly."""
    router = DocPipelineRouter()
    assert router.model == "claude-haiku-4-5-20251001"
    assert router.output_schema == DocRouterOutput


def test_doc_router_custom_model():
    """Test DocPipelineRouter with custom model."""
    router = DocPipelineRouter(model="claude-opus-4-8")
    assert router.model == "claude-opus-4-8"


def test_doc_router_get_instructions():
    """Test that instructions are document-specific."""
    router = DocPipelineRouter()
    instructions = router.get_instructions()
    assert "document" in instructions.lower()
    assert "validate" in instructions.lower()
    assert "enrich" in instructions.lower()
    assert "ALLOWED NEXT STATES" in instructions


def test_doc_router_build_prompt():
    """Test prompt building with document-specific content."""
    router = DocPipelineRouter()
    prompt = router.build_router_prompt(
        current_state="fetch",
        turn_input="Document received, ready to validate",
        history_text="User: Start processing",
        allowed_states=["validate", "error"],
    )
    assert "fetch" in prompt
    assert "validate, error" in prompt
    assert "Document received" in prompt
    assert "DOCUMENT PIPELINE ROUTING DECISION" in prompt


def test_doc_router_prompt_structure():
    """Test that routing prompt has proper structure."""
    router = DocPipelineRouter()
    prompt = router.build_router_prompt(
        current_state="validate",
        turn_input="Looks good",
        history_text="Prior: Initial setup",
        allowed_states=["enrich", "human_review"],
    )
    # Verify prompt contains all key sections
    assert "Current State:" in prompt
    assert "Allowed Next States:" in prompt
    assert "Conversation History" in prompt
    assert "User Input" in prompt


def test_doc_router_all_valid_states():
    """Test that router knows all valid document states."""
    router = DocPipelineRouter()
    # States should be mentioned in instructions
    instructions = router.get_instructions()
    valid_states = [
        "init",
        "fetch",
        "validate",
        "enrich",
        "store",
        "complete",
        "retry",
        "error",
        "human_review",
    ]
    for state in valid_states:
        assert state in instructions.lower()


def test_doc_router_instructions_include_rules():
    """Test that instructions include important rules."""
    router = DocPipelineRouter()
    instructions = router.get_instructions()
    # Should include important rules
    assert "IMPORTANT RULES:" in instructions
    assert "ALLOWED NEXT STATES" in instructions
    assert "confidence" in instructions.lower()
    assert "entities" in instructions.lower()
    assert "intents" in instructions.lower()
