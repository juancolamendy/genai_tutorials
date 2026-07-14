"""Tests for HelpdeskSemanticRouter (hrhelpdesk's TopicRouter wiring)."""

from __future__ import annotations

from unittest.mock import MagicMock

from src.hrhelpdesk.chains import RouterOutput, RouterTopic
from src.hrhelpdesk.router import HelpdeskSemanticRouter


def test_helpdesk_router_config():
    router = HelpdeskSemanticRouter()
    assert router.output_schema is RouterOutput
    assert router.unclear_topic == RouterTopic.UNCLEAR.value
    assert "faq" in router.get_instructions()
    assert "booking" in router.get_instructions()


def test_helpdesk_router_classify_faq():
    router = HelpdeskSemanticRouter()
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = RouterOutput(topic=RouterTopic.FAQ, confidence=0.88)
    router._chain = fake_chain

    decision = router.classify("How much PTO do I get?")

    assert decision.topic == "faq"
    assert decision.confidence == 0.88


def test_helpdesk_router_classify_falls_back_to_unclear_on_error():
    router = HelpdeskSemanticRouter()
    fake_chain = MagicMock()
    fake_chain.invoke.side_effect = RuntimeError("boom")
    router._chain = fake_chain

    decision = router.classify("anything")

    assert decision.topic == "unclear"
    assert decision.confidence == 0.0
