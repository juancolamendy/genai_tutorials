"""Chatbot-oriented EngineGraph: topic + confidence fan-out over typed state.

Linear pipelines keep using ``EngineGraph`` (+ optional ``semantic_router`` that
proposes a next *state*). Chatbots need a different contract:

  utterance → {topic, confidence} → typed fields (active_topic / pending_clarify)
            → proposed_next graph state

``ChatEngineGraph`` owns that fan-out. Domains set ``topic_to_state``,
``idle_state`` / ``clarify_state`` / ``notify_state``, and optionally a
``topic_router`` that implements ``classify(input_message, history) -> TopicDecision``.

Classification may still run in handlers (today's helpdesk) via
``topic_decision_to_delta``; the graph router only reads typed fields and must
not re-read message text for transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from langchain_core.runnables import RunnableConfig

from src.engine.engine_graph import EngineGraph


@dataclass(frozen=True)
class TopicDecision:
    """Result of chatbot topic classification (not a graph state name)."""

    topic: str  # e.g. faq | escalate | booking | unclear
    confidence: float


class TopicRouter(Protocol):
    """Minimal protocol for topic + confidence classifiers."""

    def classify(
        self,
        input_message: str,
        history: Optional[list[Any]] = None,
    ) -> TopicDecision: ...


class ChatEngineGraph(EngineGraph):
    """EngineGraph specialized for hub + sticky-topic chatbots.

    Subclasses configure:
      • topic_to_state — map lane id → graph state value (str or Enum)
      • idle_state / clarify_state / notify_state — hub park / clarify / notify
      • confidence_threshold — below this (or topic==unclear) → clarify
      • topic_router — optional classifier (TopicRouter protocol)

    ``_resolve_proposed_next`` is typed-field fan-out only (system / sticky /
    clarify / idle). It does not call an LLM on message text — that keeps the
    "transitions don't read messages" invariant. Handlers (or a future hook)
    write ``active_topic`` / ``pending_clarify`` using ``topic_decision_to_delta``.
    """

    idle_state: str = "idle"
    clarify_state: str = "clarify"
    notify_state: str = "notify_user"
    topic_to_state: dict[str, Any] = {}
    confidence_threshold: float = 0.7
    unclear_topic: str = "unclear"
    topic_router: Optional[TopicRouter] = None

    def _state_value(self, state_ref: Any) -> str:
        return state_ref.value if hasattr(state_ref, "value") else str(state_ref)

    def _topic_state_value(self, topic: str) -> Optional[str]:
        mapped = self.topic_to_state.get(topic)
        if mapped is None:
            return None
        return self._state_value(mapped)

    def should_clarify(self, decision: TopicDecision) -> bool:
        """True when the classifier abstains or is below threshold."""
        if decision.topic == self.unclear_topic:
            return True
        return decision.confidence < self.confidence_threshold

    def topic_decision_to_delta(
        self,
        decision: TopicDecision,
        *,
        source: str = "chat",
        now: Optional[str] = None,
    ) -> dict[str, Any]:
        """Turn a TopicDecision into sticky-lane state fields (handler helper).

        Does not set ``proposed_next`` — the graph router derives that from
        ``active_topic`` / ``pending_clarify`` on the next resolve.
        """
        base: dict[str, Any] = {
            "router_confidence": decision.confidence,
            "semantic_context": {"router_topic": decision.topic},
            "handler_status": "ok",
        }
        if self.should_clarify(decision):
            base.update(
                {
                    "pending_clarify": True,
                    "active_topic": None,
                    "topic_data": {},
                    "audit_trail": [f"{source}: routed to clarify"],
                }
            )
            return base

        started = now or datetime.now(timezone.utc).isoformat()
        base.update(
            {
                "pending_clarify": False,
                "active_topic": decision.topic,
                "topic_started_at": started,
                "topic_data": {},
                "audit_trail": [f"{source}: routed to {decision.topic}"],
            }
        )
        return base

    def classify_utterance(
        self,
        input_message: str,
        history: Optional[list[Any]] = None,
    ) -> TopicDecision:
        """Classify via ``topic_router``; missing router → unclear @ 0.0."""
        if self.topic_router is None:
            return TopicDecision(topic=self.unclear_topic, confidence=0.0)
        return self.topic_router.classify(input_message, history)

    def _resolve_proposed_next(
        self, state: dict[str, Any], config: Optional[RunnableConfig] = None
    ) -> dict[str, Any]:
        """Hub fan-out from typed chat fields — never from message text."""
        current = self._get_current_state(state)
        current_val = self._state_value(current)
        idle = self._state_value(self.idle_state)
        clarify = self._state_value(self.clarify_state)
        notify = self._state_value(self.notify_state)

        if state.get("current_event_source") == "system":
            if current_val != notify:
                return {
                    "proposed_next": notify,
                    "audit_trail": [f"router: system → {notify}"],
                }
            return {
                "proposed_next": idle,
                "audit_trail": [f"router: post-notify → {idle}"],
            }

        if state.get("pending_clarify"):
            return {
                "proposed_next": clarify,
                "audit_trail": [f"router: clarify → {clarify}"],
            }

        active = state.get("active_topic")
        topic_state = self._topic_state_value(active) if active else None
        if topic_state is not None:
            return {
                "proposed_next": topic_state,
                "audit_trail": [f"router: topic {active} → {topic_state}"],
            }

        return {
            "proposed_next": idle,
            "audit_trail": [f"router: hub park {current_val} → {idle}"],
        }


__all__ = [
    "TopicDecision",
    "TopicRouter",
    "ChatEngineGraph",
]
