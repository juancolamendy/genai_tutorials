"""
Document pipeline semantic router for LLM-powered state transitions.

Inherits from DefaultSemanticRouter and provides domain-specific:
- Output schema (Pydantic model)
- System instructions
- Prompt template
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from engine.router import DefaultSemanticRouter


class DocRouterOutput(BaseModel):
    """Output schema for document pipeline router."""

    proposed_next: str = Field(
        ...,
        description=(
            "Next state (init, fetch, validate, enrich, store, "
            "complete, retry, error, human_review)"
        ),
    )
    confidence: float = Field(
        ...,
        description="Confidence score [0.0, 1.0]",
    )
    semantic_entities: dict = Field(
        default_factory=dict,
        description="Extracted entities from user input (e.g., doc_id, keywords)",
    )
    semantic_intents: list = Field(
        default_factory=list,
        description="Extracted intents (e.g., submit, review, escalate)",
    )
    reasoning: str = Field(
        default="",
        description="Brief reasoning for the decision",
    )


class DocPipelineRouter(DefaultSemanticRouter):
    """
    Semantic router for document processing workflow.

    Uses Claude LLM to determine next state based on:
    - Current processing state
    - User input / document content
    - Conversation history
    - Valid allowed transitions

    Example:
        router = DocPipelineRouter()
        decision = router.route(
            current_state="validate",
            turn_input="Document looks good, proceed",
            history=[...],
            allowed_states=["enrich", "human_review", "error"],
        )
        # RouterDecision(proposed_next="enrich", confidence=0.95, ...)
    """

    output_schema = DocRouterOutput

    def get_instructions(self) -> str:
        """Return document pipeline-specific routing instructions."""
        return """You are a document processing workflow router.

Your job is to determine which state the document workflow should transition to next,
based on the current state, user input/feedback, conversation history, and document content.

STATES:
- init: Starting state
- fetch: Retrieve document from source
- validate: Check document schema and content validity
- enrich: Add metadata, tags, summary
- store: Persist enriched document
- complete: Pipeline finished successfully
- retry: Retry last operation (transient failure)
- error: Pipeline encountered permanent error
- human_review: Wait for human expert review

IMPORTANT RULES:
1. Always choose from ALLOWED NEXT STATES
2. Only transition if the operation would succeed
3. Return confidence [0.0, 1.0] based on clarity
4. Extract entities (doc_id, keywords, amounts) from input
5. Extract intents (submit, review, escalate, confirm)
6. If unsure, default to human_review rather than proceeding blindly

Be concise and confident in your decision."""

    def build_router_prompt(
        self,
        current_state: str,
        turn_input: str,
        history_text: str,
        allowed_states: list,
    ) -> str:
        """Build document-specific routing prompt."""
        allowed_str = ", ".join(allowed_states)
        return f"""DOCUMENT PIPELINE ROUTING DECISION

Current State: {current_state}
Allowed Next States: {allowed_str}

Conversation History (last 3 turns):
{history_text}

User Input / Feedback:
{turn_input}

Based on the current state, user feedback, and allowed transitions,
determine the next state. Extract any entities and intents from the user input."""
