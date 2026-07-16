"""Tests for DefaultTopicRouter (engine-level topic+confidence classifier base)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from src.engine.chat_engine_graph import TopicDecision
from src.engine.router import DefaultTopicRouter


class _MockTopicOutput(BaseModel):
    topic: str
    confidence: float


class _MockTopicRouter(DefaultTopicRouter):
    output_schema = _MockTopicOutput
    system_prompt = "Classify the topic."
    chain_name = "MockTopicRouterChain"


def _with_fake_chain(router: DefaultTopicRouter, fake_chain: MagicMock) -> DefaultTopicRouter:
    router._chain = fake_chain
    return router


def test_requires_output_schema():
    class _NoSchemaRouter(DefaultTopicRouter):
        pass

    with pytest.raises(NotImplementedError):
        _NoSchemaRouter()


def test_defaults():
    router = _MockTopicRouter()
    assert router.model_role == "router"
    assert router.unclear_topic == "unclear"


def test_classify_returns_topic_decision():
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = _MockTopicOutput(topic="faq", confidence=0.92)
    router = _with_fake_chain(_MockTopicRouter(), fake_chain)

    decision = router.classify("How much PTO do I get?")

    assert decision == TopicDecision(topic="faq", confidence=0.92)
    fake_chain.invoke.assert_called_once_with({"input": "How much PTO do I get?"})


@pytest.mark.asyncio
async def test_aclassify_uses_ainvoke():
    from unittest.mock import AsyncMock

    fake_chain = MagicMock()
    fake_chain.ainvoke = AsyncMock(
        return_value=_MockTopicOutput(topic="faq", confidence=0.91)
    )
    router = _with_fake_chain(_MockTopicRouter(), fake_chain)

    decision = await router.aclassify("How much PTO do I get?")

    assert decision == TopicDecision(topic="faq", confidence=0.91)
    fake_chain.ainvoke.assert_awaited_once_with({"input": "How much PTO do I get?"})


def test_classify_normalizes_dict_response():
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = {"topic": "booking", "confidence": 0.8}
    router = _with_fake_chain(_MockTopicRouter(), fake_chain)

    decision = router.classify("book a desk")

    assert decision.topic == "booking"
    assert decision.confidence == 0.8


def test_classify_clamps_confidence():
    fake_chain = MagicMock()
    fake_chain.invoke.return_value = _MockTopicOutput(topic="faq", confidence=1.5)
    router = _with_fake_chain(_MockTopicRouter(), fake_chain)

    decision = router.classify("anything")

    assert decision.confidence == 1.0


def test_classify_falls_back_to_unclear_on_exception():
    fake_chain = MagicMock()
    fake_chain.invoke.side_effect = RuntimeError("boom")
    router = _with_fake_chain(_MockTopicRouter(), fake_chain)

    decision = router.classify("anything")

    assert decision == TopicDecision(topic="unclear", confidence=0.0)


def test_get_chain_builds_via_make_chain(monkeypatch):
    captured = {}

    def fake_make_chain(*, name, system_prompt, output_schema, model_id):
        captured.update(
            name=name,
            system_prompt=system_prompt,
            output_schema=output_schema,
            model_id=model_id,
        )
        return MagicMock()

    monkeypatch.setattr("src.engine.router.make_chain", fake_make_chain)
    monkeypatch.setattr("src.engine.router.get_model", lambda role: f"model-for-{role}")

    router = _MockTopicRouter()
    router._get_chain()

    assert captured["name"] == "MockTopicRouterChain"
    assert captured["system_prompt"] == "Classify the topic."
    assert captured["output_schema"] is _MockTopicOutput
    assert captured["model_id"] == "model-for-router"
