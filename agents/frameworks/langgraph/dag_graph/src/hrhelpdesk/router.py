"""Topic-classification router for the HR helpdesk chatbot."""

from __future__ import annotations

from src.engine.router import DefaultTopicRouter

from .chains import RouterOutput, RouterTopic


class HelpdeskSemanticRouter(DefaultTopicRouter):
    """Classifies a helpdesk utterance into faq | escalate | booking | unclear."""

    output_schema = RouterOutput
    unclear_topic = RouterTopic.UNCLEAR.value
    chain_name = "HelpdeskRouteChain"

    def get_instructions(self) -> str:
        return """You are a semantic router for an HR helpdesk assistant.

Classify the user's message into exactly one topic:
- faq: policy / benefits / PTO questions answerable from a knowledge base
- escalate: complaints, HR issues, or requests needing a human ticket
- booking: desk / office seat reservations
- unclear: ambiguous or off-topic

Respond ONLY with valid JSON:
{{
  "topic": "faq" | "escalate" | "booking" | "unclear",
  "confidence": <float 0.0-1.0>
}}"""


__all__ = ["HelpdeskSemanticRouter"]
