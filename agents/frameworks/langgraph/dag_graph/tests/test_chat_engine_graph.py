"""Tests for ChatEngineGraph topic fan-out helpers."""

from src.engine.chat_engine_graph import ChatEngineGraph, TopicDecision


class _TinyChatGraph(ChatEngineGraph):
    idle_state = "idle"
    clarify_state = "hub_clarify"
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
    assert clarify["proposed_next"] == "hub_clarify"

    system = g._resolve_proposed_next(
        {
            "current_state": "idle",
            "current_event_source": "system",
            "pending_clarify": False,
        }
    )
    assert system["proposed_next"] == "notify_user"
