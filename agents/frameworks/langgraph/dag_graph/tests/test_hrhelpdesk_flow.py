"""End-to-end HR helpdesk workflow tests with mocked LLM chains/agents."""

from __future__ import annotations

import importlib
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.engine.handler_registry import get_handler_metadata
from src.hrhelpdesk.chains import EscapeOutput, RouterOutput, RouterTopic
from src.hrhelpdesk.graph import build_graph as _build_helpdesk_graph
from src.hrhelpdesk.providers import _booking_store, reset_providers
from src.hrhelpdesk.state_transitions import State


@pytest.fixture(autouse=True)
def _ensure_handlers_registered():
    if get_handler_metadata(State.IDLE.value) is None:
        import src.hrhelpdesk.handlers as hd_handlers

        importlib.reload(hd_handlers)


@pytest.fixture(autouse=True)
def _reset_provider_stores():
    reset_providers()
    yield
    reset_providers()


def build_graph():
    sessions_dir = tempfile.mkdtemp(prefix=f"hd_test_{uuid4()}_")
    return _build_helpdesk_graph(sessions_dir=sessions_dir)


def _router(topic: str, confidence: float = 0.9):
    return RouterOutput(topic=RouterTopic(topic), confidence=confidence)


def _escape(escape: bool = False):
    return EscapeOutput(escape=escape)


def _mock_faq_agent(answer: str = "You get 15 days PTO per year. [pto-001]"):
    mock = MagicMock()
    mock.invoke.return_value = {"messages": [AIMessage(content=answer)]}
    return mock


def _mock_escalate_agent(subject: str = "Pay issue", body: str = "Paycheck wrong"):
    tool_call = {
        "name": "create_ticket_tool",
        "args": {"subject": subject, "body": body},
        "id": "tc-1",
    }
    mock = MagicMock()
    mock.invoke.return_value = {
        "messages": [
            AIMessage(content="", tool_calls=[tool_call]),
            ToolMessage(content="TICKET-1", name="create_ticket_tool", tool_call_id="tc-1"),
        ]
    }
    return mock


def _mock_booking_agent_turn(
    *,
    date: str | None = None,
    location: str | None = None,
    seat_pref: str | None = None,
    confirm: bool = False,
    reply: str = "Working on your booking.",
):
    messages: list = [AIMessage(content=reply)]
    if date and location and not confirm:
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "check_desk_availability",
                        "args": {"date": date, "location": location},
                        "id": "avail-1",
                    }
                ],
            ),
            ToolMessage(content="true", name="check_desk_availability", tool_call_id="avail-1"),
            AIMessage(content=reply),
        ]
    if confirm and date and location and seat_pref:
        messages = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "confirm_booking",
                        "args": {
                            "date": date,
                            "location": location,
                            "seat_pref": seat_pref,
                        },
                        "id": "book-1",
                    }
                ],
            ),
            ToolMessage(content="BOOK-1", name="confirm_booking", tool_call_id="book-1"),
            AIMessage(content=f"Booked {date} at {location}."),
        ]
    mock = MagicMock()
    mock.invoke.return_value = {"messages": messages}
    return mock


@pytest.mark.asyncio
async def test_faq_happy_path_clears_active_topic():
    graph = build_graph()
    thread_id = str(uuid4())

    with (
        patch("src.hrhelpdesk.handlers.run_router", return_value=_router("faq")),
        patch("src.hrhelpdesk.handlers.run_escape", return_value=_escape(False)),
        patch("src.hrhelpdesk.handlers.faq_agent", _mock_faq_agent()),
    ):
        result = await graph.aemit_event(
            thread_id=thread_id,
            source="human",
            event_type="message",
            payload={"text": "How much PTO do I get?"},
        )

    assert result["status"] == "ok"
    assert result.get("active_topic") is None
    assert any("15 days" in msg for msg in result.get("output_messages", []))


@pytest.mark.asyncio
async def test_escalate_creates_one_ticket_ledger_dedupes():
    graph = build_graph()
    thread_id = str(uuid4())

    with (
        patch("src.hrhelpdesk.handlers.run_router", return_value=_router("escalate")),
        patch("src.hrhelpdesk.handlers.run_escape", return_value=_escape(False)),
        patch("src.hrhelpdesk.handlers.escalate_agent", _mock_escalate_agent()),
    ):
        first = await graph.aemit_event(
            thread_id=thread_id,
            source="human",
            event_type="message",
            payload={"text": "My paycheck is wrong"},
        )
        second = await graph.aemit_event(
            thread_id=thread_id,
            source="human",
            event_type="message",
            payload={"text": "My paycheck is wrong again"},
        )

    assert first["status"] == "ok"
    assert len(first.get("open_tickets", [])) == 1
    assert second["status"] == "ok"
    assert len(second.get("open_tickets", [])) == 1


@pytest.mark.asyncio
async def test_booking_sticky_two_turn_confirm():
    graph = build_graph()
    thread_id = str(uuid4())

    with (
        patch("src.hrhelpdesk.handlers.run_router", return_value=_router("booking")),
        patch("src.hrhelpdesk.handlers.run_escape", return_value=_escape(False)),
        patch(
            "src.hrhelpdesk.handlers.booking_agent",
            _mock_booking_agent_turn(date="2026-08-01", location="NYC"),
        ),
    ):
        turn1 = await graph.aemit_event(
            thread_id=thread_id,
            source="human",
            event_type="message",
            payload={"text": "Book a desk Monday NYC"},
        )

    assert turn1["status"] == "ok"
    assert turn1.get("active_topic") == "booking"
    assert turn1.get("topic_data", {}).get("date") == "2026-08-01"
    assert turn1.get("topic_data", {}).get("location") == "NYC"
    assert not turn1.get("topic_data", {}).get("booking_confirmed")

    with (
        patch("src.hrhelpdesk.handlers.run_escape", return_value=_escape(False)),
        patch(
            "src.hrhelpdesk.handlers.booking_agent",
            _mock_booking_agent_turn(
                date="2026-08-01",
                location="NYC",
                seat_pref="window",
                confirm=True,
            ),
        ),
    ):
        turn2 = await graph.aemit_event(
            thread_id=thread_id,
            source="human",
            event_type="message",
            payload={"text": "Window seat please"},
        )

    assert turn2["status"] == "ok"
    assert turn2.get("active_topic") is None
    assert len(turn2.get("bookings", [])) == 1
    assert turn2["bookings"][0]["seat_pref"] == "window"


@pytest.mark.asyncio
async def test_escape_mid_booking_reroutes():
    graph = build_graph()
    thread_id = str(uuid4())

    with (
        patch("src.hrhelpdesk.handlers.run_router", return_value=_router("booking")),
        patch("src.hrhelpdesk.handlers.run_escape", return_value=_escape(False)),
        patch(
            "src.hrhelpdesk.handlers.booking_agent",
            _mock_booking_agent_turn(date="2026-08-01", location="NYC"),
        ),
    ):
        await graph.aemit_event(
            thread_id=thread_id,
            source="human",
            event_type="message",
            payload={"text": "Book a desk"},
        )

    with (
        patch("src.hrhelpdesk.handlers.run_router", return_value=_router("faq")),
        patch("src.hrhelpdesk.handlers.run_escape", return_value=_escape(True)),
        patch("src.hrhelpdesk.handlers.faq_agent", _mock_faq_agent()),
    ):
        result = await graph.aemit_event(
            thread_id=thread_id,
            source="human",
            event_type="message",
            payload={"text": "Actually how much PTO?"},
        )

    assert result["status"] == "ok"
    assert result.get("active_topic") is None


@pytest.mark.asyncio
async def test_illegal_system_event_ignored():
    graph = build_graph()
    thread_id = str(uuid4())

    result = await graph.aemit_event(
        thread_id=thread_id,
        source="system",
        event_type="ticket_resolved",
        event_id="evt-bad",
        payload={"ticket_id": "TICKET-999"},
    )

    assert result["status"] == "ignored"


@pytest.mark.asyncio
async def test_topic_timeout_clears_booking():
    graph = build_graph()
    thread_id = str(uuid4())

    state = graph._get_or_init_state(session_id=thread_id, user_id="")
    stale = (datetime.now(timezone.utc) - timedelta(hours=49)).isoformat()
    state.update(
        {
            "current_state": State.IDLE.value,
            "active_topic": "booking",
            "topic_started_at": stale,
            "topic_data": {"date": "2026-08-01", "location": "NYC"},
        }
    )
    graph.compiled_graph.update_state(
        {"configurable": {"thread_id": thread_id}},
        state,
    )

    result = await graph.aemit_event(
        thread_id=thread_id,
        source="system",
        event_type="topic_timeout",
        event_id="evt-timeout",
    )

    assert result["status"] == "ok"
    assert result.get("active_topic") is None
    assert result.get("topic_data") == {}


@pytest.mark.asyncio
async def test_faq_path_never_calls_booking_tools():
    graph = build_graph()
    thread_id = str(uuid4())

    def _forbidden_booking(*_a, **_k):
        raise AssertionError("booking tools must not run on FAQ path")

    with (
        patch("src.hrhelpdesk.handlers.run_router", return_value=_router("faq")),
        patch("src.hrhelpdesk.handlers.run_escape", return_value=_escape(False)),
        patch("src.hrhelpdesk.handlers.faq_agent", _mock_faq_agent()),
        patch("src.hrhelpdesk.providers.confirm_booking", side_effect=_forbidden_booking),
    ):
        result = await graph.aemit_event(
            thread_id=thread_id,
            source="human",
            event_type="message",
            payload={"text": "PTO policy?"},
        )

    assert result["status"] == "ok"
    assert len(_booking_store) == 0
