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

from pydantic import BaseModel

from engine.router import BaseSemanticRouter, RouterDecision
from engine.agent import make_agent

log = logging.getLogger(__name__)


# Output schema for router LLM calls
class RouterOutput(BaseModel):
    """Structure for router LLM output."""
    proposed_next: str
    confidence: float
    semantic_intents: list[str] = []
    semantic_entities: dict[str, Any] = {}
    reasoning: str = ""


class DocPipelineRouter(BaseSemanticRouter):
    """
    Document pipeline semantic router using Claude LLM via Agno.

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
        Initialize router with Claude LLM via Agno.

        Args:
            model: Claude model ID (default claude-haiku for cost efficiency)
        """
        self.model = model

        # Create Agno agent for routing decisions
        self.router_agent = make_agent(
            name="DocPipelineRouter",
            description="Routes document processing workflow based on user intent and context.",
            output_schema=RouterOutput,
            instructions=[
                "You are a state machine router for document processing workflows.",
                "Given the current state, user input, conversation history, and allowed next states,",
                "determine which state the workflow should transition to next.",
                "",
                "Classify the user's intent: confirm, clarify, escalate, upload, cancel, proceed, etc.",
                "Extract entities: amounts (e.g. '$99.99'), document IDs, item names, keywords.",
                "Always propose one of the ALLOWED NEXT STATES.",
                "Return confidence 0.0-1.0 based on how clear the user's intent is.",
                "Provide brief reasoning for your decision.",
            ],
        )

    def route(self,
              current_state: str,
              turn_input: str,
              history: list,
              allowed_states: list,
              timeout_sec: float = 10.0) -> RouterDecision:
        """
        Classify next state using Claude LLM via Agno.

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

            # Call Claude LLM via Agno agent
            result = self.router_agent.run(prompt)

            # Extract RouterOutput from agent result
            router_output: RouterOutput = result.content

            # Validate proposed next state is in allowed_states
            proposed = router_output.proposed_next
            if proposed not in allowed_states:
                log.warning(
                    f"Router proposed invalid state '{proposed}'. "
                    f"Allowed: {allowed_states}. Falling back to first allowed state."
                )
                proposed = allowed_states[0] if allowed_states else "error"

            # Convert RouterOutput to RouterDecision
            decision = RouterDecision(
                proposed_next=proposed,
                confidence=max(0.0, min(1.0, router_output.confidence)),  # Clamp to [0, 1]
                semantic_entities=router_output.semantic_entities or {},
                semantic_intents=router_output.semantic_intents or [],
                reasoning=router_output.reasoning
            )

            log.info(
                "[Router] %s + '%s...' → %s (confidence: %.2f)",
                current_state,
                turn_input[:30] if turn_input else "(empty)",
                proposed,
                decision.confidence
            )

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
