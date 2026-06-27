"""
Semantic router base classes for LLM-powered state transitions.

Provides:
- RouterDecision: Output structure with decision metadata
- BaseSemanticRouter: Abstract interface
- DefaultSemanticRouter: Concrete implementation with common LLM logic
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

    proposed_next: str  # Next state (e.g., "validate")
    confidence: float  # [0.0, 1.0] confidence score
    semantic_entities: dict = field(default_factory=dict)  # Extracted entities
    semantic_intents: list = field(default_factory=list)  # Extracted intents
    reasoning: Optional[str] = None  # Optional explanation


class BaseSemanticRouter(ABC):
    """
    Abstract base class for LLM-powered state routers.

    Subclasses must implement route() to classify user input and determine
    the next state in a multi-turn conversation.

    Example:
        class MyRouter(BaseSemanticRouter):
            def route(self, current_state, turn_input, history, allowed_states, timeout_sec):
                # Use LLM to decide next state
                return RouterDecision(proposed_next="validate", confidence=0.95)
    """

    @abstractmethod
    def route(
        self,
        current_state: str,
        turn_input: str,
        history: list,
        allowed_states: list,
        timeout_sec: float = 10.0,
    ) -> RouterDecision:
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
            • Must respect allowed_states; invalid proposal → retry with constraints
            • Extract semantic entities specific to domain (amounts, items, keywords)
            • Extract user intents (confirm, clarify, escalate, etc.)
            • Set confidence [0.0, 1.0] reflecting decision quality
            • Handle timeouts gracefully; return ERROR state on timeout
        """
        raise NotImplementedError


class DefaultSemanticRouter(BaseSemanticRouter):
    """
    Concrete router with common LLM-powered routing logic.

    Subclasses override:
    - output_schema: Pydantic model for LLM response (domain-specific)
    - get_instructions(): LLM system instructions
    - build_router_prompt(): LLM prompt template

    Common logic handles:
    - LLM client initialization
    - Route classification workflow
    - State validation and fallback
    - Confidence clamping
    - Logging and error handling

    Example:
        class DocRouter(DefaultSemanticRouter):
            output_schema = DocRouterOutput  # Pydantic model

            def get_instructions(self):
                return "You are a document router. Choose next state: ..."

            def build_router_prompt(self, current_state, turn_input, ...):
                return f"State: {current_state}\\nInput: {turn_input}\\n..."
    """

    output_schema: type = None  # Subclasses MUST set this to a Pydantic model

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        """
        Initialize router with Claude LLM via LangChain.

        Args:
            model: Claude model ID (default haiku for cost efficiency)

        Raises:
            NotImplementedError: If output_schema not set by subclass
        """
        if self.output_schema is None:
            raise NotImplementedError(
                f"Subclass {self.__class__.__name__} must set output_schema to a Pydantic model"
            )

        self.model = model
        self.llm = None  # Lazy-loaded in route()

    def _get_llm(self):
        """Lazy-load LLM client (singleton per router instance)."""
        if self.llm is None:
            from langchain_anthropic import ChatAnthropic

            self.llm = ChatAnthropic(
                model=self.model,
                temperature=0,
                timeout=15.0,
            )
        return self.llm

    def get_instructions(self) -> str:
        """
        Return LLM system instructions for routing.

        Subclasses override to provide domain-specific instructions.
        Default provides generic routing guidance.
        """
        return """You are a state machine router for workflows.
Given the current state, user input, conversation history, and allowed next states,
determine which state the workflow should transition to next.

IMPORTANT: Always propose one of the ALLOWED NEXT STATES.

Extract relevant entities and intents from the user's input.
Return confidence 0.0-1.0 based on how clear the user's intent is.
Provide brief reasoning for your decision."""

    def build_router_prompt(
        self,
        current_state: str,
        turn_input: str,
        history_text: str,
        allowed_states: list,
    ) -> str:
        """
        Build LLM prompt for state classification.

        Subclasses override to provide domain-specific prompt format.
        Default provides generic structure.

        Args:
            current_state: Current state (e.g., "validate")
            turn_input: User's input text
            history_text: Formatted conversation history
            allowed_states: List of valid next states

        Returns:
            Prompt string for LLM
        """
        allowed_str = ", ".join(allowed_states)
        return f"""WORKFLOW STATE MACHINE ROUTING

Current State: {current_state}
Allowed Next States: {allowed_str}

Conversation History (last 5 turns):
{history_text}

User Input: {turn_input}

Determine the next state based on the user's intent and the current state.
Always choose from the ALLOWED NEXT STATES."""

    def route(
        self,
        current_state: str,
        turn_input: str,
        history: list,
        allowed_states: list,
        timeout_sec: float = 10.0,
    ) -> RouterDecision:
        """
        Route using LLM with structure constraints.

        Args:
            current_state: Current state
            turn_input: User input (already escaped)
            history: List of prior turns
            allowed_states: Valid next states
            timeout_sec: Timeout for LLM call

        Returns:
            RouterDecision with validated proposed_next
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            # Build history text from last 5 turns only
            history_text = "\n".join(
                [
                    f"{turn.get('role', 'user').title()}: {turn.get('content', '')[:100]}"
                    for turn in history[-5:]
                ]
            ) or "(No prior turns)"

            # Build prompt
            prompt = self.build_router_prompt(
                current_state,
                turn_input,
                history_text,
                allowed_states,
            )

            # Call LLM with structured output
            llm = self._get_llm()
            with_structure = llm.with_structured_output(self.output_schema)

            response = with_structure.invoke(
                [
                    SystemMessage(content=self.get_instructions()),
                    HumanMessage(content=prompt),
                ]
            )

            # Validate proposed_next is in allowed_states
            if response.proposed_next not in allowed_states:
                log.warning(
                    "[SemanticRouter] Invalid proposal '%s' not in %s; fallback to '%s'",
                    response.proposed_next,
                    allowed_states,
                    allowed_states[0],
                )
                response.proposed_next = allowed_states[0]

            # Clamp confidence to [0.0, 1.0]
            confidence = max(0.0, min(1.0, response.confidence))

            log.info(
                "[SemanticRouter] %s + '%s...' → %s (conf: %.2f)",
                current_state,
                turn_input[:30] if turn_input else "(empty)",
                response.proposed_next,
                confidence,
            )

            return RouterDecision(
                proposed_next=response.proposed_next,
                confidence=confidence,
                semantic_entities=getattr(response, "semantic_entities", {}),
                semantic_intents=getattr(response, "semantic_intents", []),
                reasoning=getattr(response, "reasoning", None),
            )

        except Exception as e:
            log.exception("[SemanticRouter] Error in route(): %s", e)
            # Fallback: return first allowed state with low confidence
            return RouterDecision(
                proposed_next=allowed_states[0] if allowed_states else "error",
                confidence=0.0,
                reasoning=f"Error during routing: {str(e)}",
            )
