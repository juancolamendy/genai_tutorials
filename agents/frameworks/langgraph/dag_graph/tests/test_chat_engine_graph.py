"""Tests for ChatEngineGraph topic fan-out helpers."""

from unittest.mock import MagicMock

import pytest

from src.engine.chat_engine_graph import ChatEngineGraph, TopicDecision
from src.engine.escape_checker import DefaultEscapeChecker, EscapeDecision, EscapeOutput


class _TinyChatGraph(ChatEngineGraph):
    idle_state = "idle"
    clarify_state = "clarify"
    notify_state = "notify_user"
    topic_to_state = {"faq": "topic_faq", "booking": "topic_booking"}
    confidence_threshold = 0.7


def test_should_clarify_on_unclear_or_low_confidence():
    g = _TinyChatGraph()
    assert g.should_clarify(TopicDecision("unclear", 0.99)) is True
    assert g.should_clarify(TopicDecision("faq", 0.5)) is True
    assert g.should_clarify(TopicDecision("faq", 0.9)) is False


def test_topic_decision_to_delta_sets_active_topic():
    g = _TinyChatGraph()
    delta = g.topic_decision_to_delta(
        TopicDecision("faq", 0.95), source="test", now="2026-07-14T00:00:00+00:00"
    )
    assert delta["pending_clarify"] is False
    assert delta["active_topic"] == "faq"
    assert delta["topic_started_at"] == "2026-07-14T00:00:00+00:00"


def test_resolve_proposed_next_from_typed_fields():
    g = _TinyChatGraph()
    g._get_current_state = lambda state: state.get("current_state", "idle")  # type: ignore

    sticky = g._resolve_proposed_next(
        {"current_state": "idle", "active_topic": "faq", "pending_clarify": False}
    )
    assert sticky["proposed_next"] == "topic_faq"

    clarify = g._resolve_proposed_next(
        {"current_state": "idle", "pending_clarify": True}
    )
    assert clarify["proposed_next"] == "clarify"

    system = g._resolve_proposed_next(
        {
            "current_state": "idle",
            "current_event_source": "system",
            "pending_clarify": False,
        }
    )
    assert system["proposed_next"] == "notify_user"


def test_run_escape_uses_escape_checker():
    g = _TinyChatGraph()
    mock = MagicMock()
    mock.check.return_value = EscapeDecision(escape=True)
    g.escape_checker = mock
    assert g.run_escape("switch to PTO").escape is True
    mock.check.assert_called_once_with("switch to PTO")


@pytest.mark.asyncio
async def test_arun_escape_uses_acheck():
    from unittest.mock import AsyncMock

    g = _TinyChatGraph()
    mock = MagicMock()
    mock.acheck = AsyncMock(return_value=EscapeDecision(escape=True))
    g.escape_checker = mock
    assert (await g.arun_escape("switch to PTO")).escape is True
    mock.acheck.assert_awaited_once_with("switch to PTO")


@pytest.mark.asyncio
async def test_default_escape_checker_acheck_uses_ainvoke():
    from unittest.mock import AsyncMock

    checker = DefaultEscapeChecker()
    chain = MagicMock()
    chain.ainvoke = AsyncMock(return_value=EscapeOutput(escape=True))
    checker._get_chain = MagicMock(return_value=chain)  # type: ignore[method-assign]
    assert (await checker.acheck("cancel this")).escape is True
    chain.ainvoke.assert_awaited_once_with({"input": "cancel this"})


def test_default_escape_checker_failure_stays_in_lane():
    checker = DefaultEscapeChecker()
    checker._get_chain = MagicMock(side_effect=RuntimeError("boom"))  # type: ignore[method-assign]
    assert checker.check("anything") == EscapeDecision(escape=False)


def test_default_escape_checker_parses_output():
    checker = DefaultEscapeChecker()
    chain = MagicMock()
    chain.invoke.return_value = EscapeOutput(escape=True)
    checker._get_chain = MagicMock(return_value=chain)  # type: ignore[method-assign]
    assert checker.check("cancel this").escape is True
