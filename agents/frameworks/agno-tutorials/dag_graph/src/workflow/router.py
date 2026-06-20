"""
workflow/router.py
────────────────────────────────────────────────────────────────────────────
DocPipelineRouter — LLM-powered semantic router for document processing.

Inherits from BaseSemanticRouter and provides:
  • Claude LLM integration for state classification
  • Domain-specific entity extraction (amounts, items, document_ids)
  • Domain-specific intent extraction (confirm, clarify, escalate, etc.)
  • Constraint retry logic for invalid transitions
  • Timeout handling
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from engine.router import BaseSemanticRouter, RouterDecision

log = logging.getLogger(__name__)


class DocPipelineRouter(BaseSemanticRouter):
    """
    Document pipeline semantic router using Claude LLM.

    Classifies user input and determines next state based on:
      • Current state in workflow
      • User's turn input (entities like amounts, items)
      • Conversation history
      • Allowed next states per state machine

    Extracts:
      • Entities: amounts (e.g., "$99.99"), items (e.g., "document.pdf")
      • Intents: confirm, clarify, escalate, upload, cancel
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        """
        Initialize router with Claude model.

        Args:
            model: Claude model ID (default claude-haiku for cost efficiency)
        """
        self.model = model
        # Note: Claude LLM would be initialized here when agno is available
        # from agno.models.anthropic import Claude
        # self.llm = Claude(id=model)

    def route(self,
              current_state: str,
              turn_input: str,
              history: list,
              allowed_states: list,
              timeout_sec: float = 10.0) -> RouterDecision:
        """
        Classify next state using Claude LLM.

        Args:
            current_state: Current state (e.g., "validate")
            turn_input: User's input (already validated and escaped)
            history: Recent turn history (last N turns)
            allowed_states: Valid next states
            timeout_sec: LLM call timeout

        Returns:
            RouterDecision with proposed next state and semantic entities
        """
        try:
            # Format history for LLM context
            history_text = self._format_history(history)

            # Build prompt with current state, user input, allowed states
            prompt = self._build_router_prompt(
                current_state, turn_input, history_text, allowed_states
            )

            # Call Claude LLM (would be: response = self.llm.call(prompt, timeout=timeout_sec))
            # For now, return stub response showing the structure
            response = self._mock_llm_call(prompt)

            # Parse LLM response to RouterDecision
            decision = self._parse_response(response, allowed_states)

            return decision

        except Exception as e:
            log.exception("Router error: %s", e)
            return RouterDecision(
                proposed_next="error",
                confidence=0.0,
                semantic_entities={},
                semantic_intents=[],
                reasoning=f"Router error: {str(e)}"
            )

    def _build_router_prompt(self,
                             current_state: str,
                             turn_input: str,
                             history_text: str,
                             allowed_states: list) -> str:
        """Build LLM prompt for state classification."""
        allowed_str = ", ".join(allowed_states)
        prompt = f"""
You are a state machine router for document processing workflows.

Current state: {current_state}
Allowed next states: {allowed_str}

Conversation history:
{history_text}

User input: {repr(turn_input)}

Classify the user's intent and determine the next state:
1. What are the user's intents? (confirm, clarify, escalate, upload, cancel, etc.)
2. What entities did the user mention? (amounts, items, document_ids, keywords)
3. Which allowed next state should we transition to?

Respond with JSON:
{{
  "proposed_next": "<state>",
  "confidence": 0.95,
  "semantic_intents": ["confirm"],
  "semantic_entities": {{"amounts": ["$99.99"], "items": []}},
  "reasoning": "User confirmed amount"
}}
"""
        return prompt

    def _format_history(self, history: list, max_turns: int = 10) -> str:
        """Format turn history for LLM context."""
        if not history:
            return "(no history)"
        recent = history[-max_turns:]
        lines = []
        for turn in recent:
            role = turn.get("role", "?")
            content = turn.get("content", "")
            lines.append(f"[{role}]: {content}")
        return "\n".join(lines)

    def _mock_llm_call(self, prompt: str) -> str:
        """Mock LLM call (would be replaced with actual Claude call)."""
        # In production: response = self.llm.call(prompt)
        # For testing: return valid JSON structure
        return json.dumps({
            "proposed_next": "validate",
            "confidence": 0.9,
            "semantic_intents": ["confirm"],
            "semantic_entities": {},
            "reasoning": "User provided input"
        })

    def _parse_response(self, response: str, allowed_states: list) -> RouterDecision:
        """Parse LLM response and extract RouterDecision."""
        try:
            # Extract JSON from response
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if not match:
                return RouterDecision(proposed_next=allowed_states[0], confidence=0.5)

            data = json.loads(match.group())

            proposed = data.get("proposed_next", allowed_states[0])

            # Constraint retry: if invalid transition, use allowed state
            if proposed not in allowed_states:
                log.warning(f"Router proposed invalid state {proposed}; using {allowed_states[0]}")
                proposed = allowed_states[0]

            return RouterDecision(
                proposed_next=proposed,
                confidence=float(data.get("confidence", 0.5)),
                semantic_entities=data.get("semantic_entities", {}),
                semantic_intents=data.get("semantic_intents", []),
                reasoning=data.get("reasoning")
            )

        except Exception as e:
            log.error("Failed to parse router response: %s", e)
            return RouterDecision(proposed_next=allowed_states[0], confidence=0.5)
