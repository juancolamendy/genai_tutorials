"""Tests for conversational message hygiene helpers."""

from langchain_core.messages import AIMessage, HumanMessage

from src.engine.message_hygiene import (
    close_topic_delta,
    segment_reset_messages,
    trim_messages,
)


def test_trim_messages_keeps_last_n():
    msgs = [HumanMessage(content=str(i), id=f"h{i}") for i in range(15)]
    trimmed = trim_messages(msgs, max_n=12)
    assert len(trimmed) == 12
    assert trimmed[0].content == "3"
    assert trimmed[-1].content == "14"


def test_segment_reset_removes_prior_and_appends_summary():
    msgs = [
        HumanMessage(content="hi", id="h1"),
        AIMessage(content="hello", id="a1"),
    ]
    update = segment_reset_messages(msgs, summary="Topic closed: answered FAQ")
    assert len(update) == 3
    assert update[0].id == "h1"
    assert update[1].id == "a1"
    assert isinstance(update[2], AIMessage)
    assert "FAQ" in update[2].content


def test_close_topic_delta_clears_lane_fields():
    msgs = [HumanMessage(content="x", id="h1")]
    delta = close_topic_delta(msgs, summary="done")
    assert delta["active_topic"] is None
    assert delta["topic_data"] == {}
    assert delta["topic_started_at"] is None
    assert len(delta["messages"]) == 2
