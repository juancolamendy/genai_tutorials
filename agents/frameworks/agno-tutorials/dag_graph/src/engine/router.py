"""
engine/router.py
────────────────────────────────────────────────────────────────────────────
Base semantic router abstract class and RouterDecision output structure.

Defines the interface that all semantic routers must implement:
  • route() method takes current state, user input, history, and returns decision
  • RouterDecision includes proposed_next, confidence, semantic entities/intents

Workflow-specific routers (e.g., DocPipelineRouter) inherit and override
route() to provide domain-specific LLM prompts and entity extraction.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class RouterDecision:
    """Output from semantic router classification."""

    proposed_next: str                      # Next state (e.g., "validate")
    confidence: float                       # [0.0, 1.0] confidence score
    semantic_entities: dict = field(default_factory=dict)  # Extracted entities
    semantic_intents: list = field(default_factory=list)   # Extracted intents
    reasoning: Optional[str] = None                         # Optional explanation


class BaseSemanticRouter(ABC):
    """
    Abstract base class for LLM-powered state routers.

    Subclasses must implement route() to classify user input and determine
    next state in a multi-turn conversation.

    Extracted semantic information (entities, intents) helps guardrails
    and handlers make domain-specific decisions.
    """

    @abstractmethod
    def route(self,
              current_state: str,
              turn_input: str,
              history: list,
              allowed_states: list,
              timeout_sec: float = 10.0) -> RouterDecision:
        """
        Classify user input and determine next state.

        Args:
            current_state: Current state (e.g., "validate")
            turn_input: User's input text (already validated and escaped)
            history: List of prior turns [{role, content, ...}]
            allowed_states: Valid next states per state machine transitions
            timeout_sec: LLM call timeout (default 10s)

        Returns:
            RouterDecision with proposed_next, confidence, entities, intents

        Implementation notes:
            • Must respect allowed_states; propose invalid → retry with constraints
            • Extract semantic entities specific to domain (amounts, items, keywords)
            • Extract user intents (confirm, clarify, escalate, etc.)
            • Set confidence [0.0, 1.0] reflecting decision quality
            • Handle timeouts gracefully; return ERROR state on timeout
        """
        raise NotImplementedError


class DefaultSemanticRouter(BaseSemanticRouter):
    """
    Concrete base router with common LLM-powered routing logic.

    Subclasses override:
      • get_instructions() — LLM system instructions (domain-specific)
      • build_router_prompt() — LLM prompt template (domain-specific)
      • output_schema — Pydantic model for LLM response (domain-specific)

    Common logic handles:
      • Agent initialization via make_agent()
      • Route classification workflow
      • State validation and fallback
      • Confidence clamping
      • Logging and error handling
    """

    output_schema: type = None  # Subclasses must set this to a Pydantic model

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        """
        Initialize router with Claude LLM via Agno.

        Args:
            model: Claude model ID (default haiku for cost efficiency)
        """
        if self.output_schema is None:
            raise NotImplementedError(
                "Subclass must set output_schema to a Pydantic model"
            )

        self.model = model
        from engine.agent import make_agent

        self.router_agent = make_agent(
            name=self.__class__.__name__,
            description="Routes workflow based on user intent and context.",
            output_schema=self.output_schema,
            instructions=self.get_instructions(),
        )

    def get_instructions(self) -> list[str]:
        """
        Return LLM system instructions for routing.

        Subclasses override to provide domain-specific instructions.
        Default provides generic routing guidance.
        """
        return [
            "You are a state machine router for workflows.",
            "Given the current state, user input, conversation history, and allowed next states,",
            "determine which state the workflow should transition to next.",
            "Always propose one of the ALLOWED NEXT STATES.",
            "Extract relevant entities and intents from the user's input.",
            "Return confidence 0.0-1.0 based on how clear the user's intent is.",
            "Provide brief reasoning for your decision.",
        ]

    def build_router_prompt(self,
                           current_state: str,
                           turn_input: str,
                           history_text: str,
                           allowed_states: list) -> str:
        """
        Build LLM prompt for state classification.

        Subclasses override to provide domain-specific prompt format.
        Default provides generic structure.
        """
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
1. Analyze the user's input to identify their intent
2. Extract relevant entities from the input
3. Determine the best next state from the ALLOWED NEXT STATES list
4. Rate your confidence 0.0-1.0 (how clear was the user's intent?)
5. Provide brief reasoning for your decision

IMPORTANT: You MUST choose one of the allowed next states. Never propose a state outside that list.
"""

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
            prompt = self.build_router_prompt(
                current_state, turn_input, history_text, allowed_states
            )

            # Call Claude LLM via Agno agent
            result = self.router_agent.run(prompt)

            # Extract output from agent result (assumes output_schema structure)
            router_output = result.content

            # Validate proposed next state is in allowed_states
            proposed = router_output.proposed_next
            if proposed not in allowed_states:
                log.warning(
                    f"Router proposed invalid state '{proposed}'. "
                    f"Allowed: {allowed_states}. Falling back to first allowed state."
                )
                proposed = allowed_states[0] if allowed_states else "error"

            # Build RouterDecision from output
            decision = RouterDecision(
                proposed_next=proposed,
                confidence=max(0.0, min(1.0, router_output.confidence)),
                semantic_entities=getattr(router_output, 'semantic_entities', {}),
                semantic_intents=getattr(router_output, 'semantic_intents', []),
                reasoning=getattr(router_output, 'reasoning', ''),
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
