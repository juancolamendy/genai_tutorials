"""
Semantic router base classes for LLM-powered state transitions.

Provides:
- RouterDecision: Output structure with decision metadata
- BaseSemanticRouter: Abstract interface
- DefaultSemanticRouter: Concrete implementation with common LLM logic
- DefaultTopicRouter: Concrete implementation for topic+confidence
  classification (chatbot hub routing, as opposed to state routing)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from src.engine.chains import get_model, make_chain
from src.engine.chat_engine_graph import TopicDecision

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
            def route(self, current_state, input_message, history, allowed_states, timeout_sec):
                # Use LLM to decide next state
                return RouterDecision(proposed_next="validate", confidence=0.95)
    """

    @abstractmethod
    def route(
        self,
        current_state: str,
        input_message: str,
        history: list,
        allowed_states: list,
        timeout_sec: float = 10.0,
    ) -> RouterDecision:
        """
        Classify user input and determine next state.

        Args:
            current_state: Current state (e.g., "validate")
            input_message: User's input text (already validated and escaped)
            history: List of prior BaseMessage turns
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

            def build_router_prompt(self, current_state, input_message, ...):
                return f"State: {current_state}\\nInput: {input_message}\\n..."
    """

    output_schema: type = None  # Subclasses MUST set this to a Pydantic model

    def __init__(self, model: str = "anthropic:claude-haiku-4-5-20251001"):
        """
        Initialize router with a chat model via LangChain (provider-agnostic).

        Args:
            model: Chat model ID in "provider:model" format (e.g.
                "anthropic:claude-haiku-4-5-20251001", "openai:gpt-4o-mini").
                Defaults to Claude Haiku for cost efficiency.

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
            from langchain.chat_models import init_chat_model

            self.llm = init_chat_model(
                self.model,
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
Given the current state, turn input, conversation history, and allowed next states,
determine which state the workflow should transition to next.

IMPORTANT: Always propose one of the ALLOWED NEXT STATES.

Extract relevant entities and intents from the user's input.
Return confidence 0.0-1.0 based on how clear the user's intent is.
Provide brief reasoning for your decision."""

    def build_router_prompt(
        self,
        current_state: str,
        input_message: str,
        history_text: str,
        allowed_states: list,
    ) -> str:
        """
        Build LLM prompt for state classification.

        Subclasses override to provide domain-specific prompt format.
        Default provides generic structure.

        Args:
            current_state: Current state (e.g., "validate")
            input_message: Turn input text
            history_text: Formatted conversation history
            allowed_states: List of valid next states

        Returns:
            Prompt string for LLM
        """
        allowed_str = ", ".join(allowed_states)
        return f"""ENGINE STATE MACHINE ROUTING

Current State: {current_state}
Allowed Next States: {allowed_str}

Conversation History (last 5 turns):
{history_text}

User Input: {input_message}

Determine the next state based on the user's intent and the current state.
Always choose from the ALLOWED NEXT STATES."""

    def route(
        self,
        current_state: str,
        input_message: str,
        history: list,
        allowed_states: list,
        timeout_sec: float = 10.0,
    ) -> RouterDecision:
        """
        Route using LLM with structure constraints.

        Args:
            current_state: Current state
            input_message: User input (already escaped)
            history: List of prior BaseMessage turns
            allowed_states: Valid next states
            timeout_sec: Timeout for LLM call

        Returns:
            RouterDecision with validated proposed_next
        """
        if not allowed_states:
            log.warning("[SemanticRouter] No allowed_states provided; routing to 'error'")
            return RouterDecision(
                proposed_next="error",
                confidence=0.0,
                reasoning="No allowed_states provided to route()",
            )

        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            # Build history text from last 5 turns only
            history_text = "\n".join(
                [
                    f"{msg.type.title()}: {str(msg.content)[:100]}"
                    for msg in history[-5:]
                ]
            ) or "(No prior turns)"

            # Build prompt
            prompt = self.build_router_prompt(
                current_state,
                input_message,
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
                input_message[:30] if input_message else "(empty)",
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


class DefaultTopicRouter:
    """Concrete ``TopicRouter`` (structural) for LLM topic+confidence classification.

    Unlike ``DefaultSemanticRouter`` (which proposes a graph *state* and rolls
    its own LLM call), this classifies an utterance into a *topic* and reuses
    ``engine.chains.make_chain`` — the same helper every domain's
    ``chains.py`` already uses — instead of a second hand-rolled
    LLM-invocation path.

    Subclasses set:
        output_schema: Pydantic model with ``topic`` (str|Enum) and
            ``confidence`` (float) fields. Required.
        system_prompt / get_instructions(): classification system prompt.
        model_role: passed to ``get_model(role)`` for model selection
            (default ``"router"``).
        unclear_topic: topic value used when classification fails or the
            model itself returns it (default ``"unclear"``).
        chain_name: name passed to ``make_chain`` for its process-level
            cache (default ``"TopicRouterChain"`` — override per subclass
            so multiple topic routers don't collide in the shared cache).

    Example:
        class MyTopicRouter(DefaultTopicRouter):
            output_schema = MyRouterOutput
            chain_name = "MyTopicRouterChain"

            def get_instructions(self):
                return "Classify the message into: a | b | unclear..."
    """

    output_schema: type = None  # Subclasses MUST set this to a Pydantic model
    system_prompt: str = ""  # or override get_instructions()
    model_role: str = "router"
    unclear_topic: str = "unclear"
    chain_name: str = "TopicRouterChain"

    def __init__(self) -> None:
        if self.output_schema is None:
            raise NotImplementedError(
                f"Subclass {self.__class__.__name__} must set output_schema to a Pydantic model"
            )
        self._chain = None  # Lazy-built in _get_chain()

    def get_instructions(self) -> str:
        """Return the LLM system prompt. Subclasses override or set ``system_prompt``."""
        return self.system_prompt

    def _get_chain(self):
        """Lazily build (and cache) the classification chain via ``make_chain``."""
        if self._chain is None:
            self._chain = make_chain(
                name=self.chain_name,
                system_prompt=self.get_instructions(),
                output_schema=self.output_schema,
                model_id=get_model(self.model_role),
            )
        return self._chain

    def classify(
        self,
        input_message: str,
        history: Optional[list] = None,
    ) -> TopicDecision:
        """Classify ``input_message`` into a ``TopicDecision``.

        ``history`` is accepted for ``TopicRouter`` protocol conformance but
        unused by this default implementation. On any failure (LLM error,
        malformed response), falls back to ``TopicDecision(unclear_topic,
        0.0)`` rather than raising — callers already treat unclear/low
        confidence uniformly via ``ChatEngineGraph.should_clarify``, so a
        swallowed error degrades to "ask the user to clarify" instead of
        surfacing as a handler error.
        """
        try:
            raw = self._get_chain().invoke({"input": input_message})
            response = raw if isinstance(raw, self.output_schema) else self.output_schema(**raw)
            topic = response.topic
            topic_val = topic.value if hasattr(topic, "value") else str(topic)
            confidence = max(0.0, min(1.0, float(response.confidence)))
            return TopicDecision(topic=topic_val, confidence=confidence)
        except Exception as exc:
            log.exception("[DefaultTopicRouter] classify() failed: %s", exc)
            return TopicDecision(topic=self.unclear_topic, confidence=0.0)
