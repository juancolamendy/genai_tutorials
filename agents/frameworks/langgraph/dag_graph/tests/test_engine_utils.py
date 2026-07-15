"""Tests for engine conversational utils (trim / segment reset / helpers)."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.engine.utils import (
    close_topic_delta,
    content_hash,
    find_tool_call,
    find_tool_message,
    last_ai_content,
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
    assert update[2].additional_kwargs.get("segment_reset") is True


def test_close_topic_delta_clears_lane_fields():
    msgs = [HumanMessage(content="x", id="h1")]
    delta = close_topic_delta(msgs, summary="done")
    assert delta["active_topic"] is None
    assert delta["topic_data"] == {}
    assert delta["topic_started_at"] is None
    assert len(delta["messages"]) == 2


def test_content_hash_is_stable_and_short():
    assert content_hash("a", "b") == content_hash("a", "b")
    assert len(content_hash("a")) == 16


def test_last_ai_content_and_tool_helpers():
    msgs = [
        HumanMessage(content="hi"),
        AIMessage(
            content="thinking",
            tool_calls=[{"name": "create_ticket_tool", "args": {"subject": "x"}, "id": "1"}],
        ),
        ToolMessage(content="ok", tool_call_id="1", name="create_ticket_tool"),
        AIMessage(content="done"),
    ]
    assert last_ai_content(msgs) == "done"
    assert find_tool_call(msgs, "create_ticket_tool")["args"]["subject"] == "x"
    assert find_tool_message(msgs, "create_ticket_tool").content == "ok"
