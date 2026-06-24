"""
workflow/router.py
────────────────────────────────────────────────────────────────────────────
DocPipelineRouter — Domain-specific semantic router for document processing.

Inherits from DefaultSemanticRouter and overrides:
  • get_instructions() — Domain-specific LLM instructions
  • build_router_prompt() — Domain-specific prompt format
  • output_schema — Domain-specific Pydantic model

Reuses from DefaultSemanticRouter:
  • Agent initialization and LLM calls
  • State validation and fallback logic
  • Confidence clamping and error handling
  • History formatting
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from pydantic import BaseModel

from engine.router import DefaultSemanticRouter

log = logging.getLogger(__name__)


class RouterOutput(BaseModel):
    """Structure for router LLM output."""
    proposed_next: str
    confidence: float
    semantic_intents: list[str] = []
    semantic_entities: dict[str, Any] = {}
    reasoning: str = ""


class DocPipelineRouter(DefaultSemanticRouter):
    """
    Document pipeline semantic router using Claude LLM via Agno.

    Inherits common routing logic from DefaultSemanticRouter and provides
    domain-specific instructions and prompt formatting for document processing.

    Extracts:
      • Entities: amounts (e.g., "$99.99"), items (e.g., "document.pdf")
      • Intents: confirm, clarify, escalate, upload, cancel
    """

    output_schema = RouterOutput

    def get_instructions(self) -> list[str]:
        """Return domain-specific instructions for document routing."""
        return [
            "You are a state machine router for document processing workflows.",
            "Given the current state, user input, conversation history, and allowed next states,",
            "determine which state the workflow should transition to next.",
            "",
            "Classify the user's intent: confirm, clarify, escalate, upload, cancel, proceed, etc.",
            "Extract entities: amounts (e.g. '$99.99'), document IDs, item names, keywords.",
            "Always propose one of the ALLOWED NEXT STATES.",
            "Return confidence 0.0-1.0 based on how clear the user's intent is.",
            "Provide brief reasoning for your decision.",
        ]

    def build_router_prompt(self,
                           current_state: str,
                           turn_input: str,
                           history_text: str,
                           allowed_states: list) -> str:
        """Build domain-specific LLM prompt for state classification."""
        allowed_str = ", ".join(allowed_states)
        return f"""
WORKFLOW STATE MACHINE ROUTING

Current State: {current_state}
Allowed Next States: {allowed_str}

Conversation History:
{history_text}

User Input: {repr(turn_input)}

---

Your task:
1. Analyze the user's input to identify their intent (confirm, clarify, escalate, upload, cancel, etc.)
2. Extract relevant entities (amounts like "$99.99", document IDs, item names, keywords)
3. Determine the best next state from the ALLOWED NEXT STATES list
4. Rate your confidence 0.0-1.0 (how clear was the user's intent?)
5. Provide brief reasoning for your decision

IMPORTANT: You MUST choose one of the allowed next states. Never propose a state outside that list.
"""
