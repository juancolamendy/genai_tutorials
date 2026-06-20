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

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


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
