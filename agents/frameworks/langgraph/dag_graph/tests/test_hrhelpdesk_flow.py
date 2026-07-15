"""End-to-end HR helpdesk workflow tests with mocked LLM chains/agents."""

from __future__ import annotations

import importlib
import tempfile
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.engine.chat_engine_graph import TopicDecision
from src.engine.escape_checker import EscapeDecision
from src.engine.handler_registry import get_handler_metadata
from src.hrhelpdesk.graph import build_graph as _build_helpdesk_graph
from src.hrhelpdesk.services import _booking_store, reset_providers
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


def _decision(topic: str, confidence: float = 0.9) -> TopicDecision:
    return TopicDecision(topic=topic, confidence=confidence)


def _escape(escape: bool = False):
    return EscapeDecision(escape=escape)


def _mock_faq_agent(answer: str = "You get 15 days PTO per year. [pto-001]"):
    mock = MagicMock()
    mock.invoke.return_value = {"messages": [AIMessage(content=answer)]}
    return mock


def _mock_escalate_chain(
    *,
    subject: str = "Pay issue",
    body: str = "Paycheck wrong",
    should_create: bool = True,
    reply: str = "I've filed an HR ticket for your issue.",
):
    from src.hrhelpdesk.chains import EscalateDecision

    mock = MagicMock()
    mock.invoke.return_value = EscalateDecision(
        should_create=should_create,
        subject=subject,
        body=body,
        reply=reply,
    )
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
        patch(
            "src.hrhelpdesk.handlers.classify_utterance",
            return_value=_decision("faq"),
        ),
        patch("src.hrhelpdesk.handlers.run_escape", return_value=_escape(False)),
        patch("src.hrhelpdesk.handlers.faq_agent", _mock_faq_agent()),
    ):
        result = await graph.aemit_event(
            thread_id=thread_id,
            source="human",
            event_type="message",
            input_message="How much PTO do I get?",
        )

    assert result["status"] == "ok"
    assert result.get("active_topic") is None
    assert any("15 days" in msg for msg in result.get("output_messages", []))


@pytest.mark.asyncio
async def test_clarify_reroutes_user_reply_to_faq():
    """Unclear first turn parks at clarify; next reply is re-routed (not ignored)."""
    graph = build_graph()
    thread_id = str(uuid4())

    with (
        patch(
            "src.hrhelpdesk.handlers.classify_utterance",
            side_effect=[_decision("unclear", 0.4), _decision("faq")],
        ),
        patch("src.hrhelpdesk.handlers.run_escape", return_value=_escape(False)),
        patch("src.hrhelpdesk.handlers.faq_agent", _mock_faq_agent()),
    ):
        turn1 = await graph.aemit_event(
            thread_id=thread_id,
            source="human",
            event_type="message",
            input_message="help",
        )
        assert turn1["status"] == "ok"
        assert turn1.get("current_state") == State.CLARIFY.value
        assert turn1.get("pending_clarify") is True
        assert any("FAQ" in m or "book" in m.lower() for m in turn1.get("output_messages", []))

        turn2 = await graph.aemit_event(
            thread_id=thread_id,
            source="human",
            event_type="message",
            input_message="policy question about PTO",
        )

    assert turn2["status"] == "ok"
    assert turn2.get("pending_clarify") is False
    assert turn2.get("active_topic") is None  # FAQ one-shot closes topic
    assert turn2.get("current_state") == State.IDLE.value
    assert any("15 days" in msg for msg in turn2.get("output_messages", []))


@pytest.mark.asyncio
async def test_escalate_creates_one_ticket_ledger_dedupes():
    graph = build_graph()
    thread_id = str(uuid4())

    with (
        patch(
            "src.hrhelpdesk.handlers.classify_utterance",
            return_value=_decision("escalate"),
        ),
        patch("src.hrhelpdesk.handlers.run_escape", return_value=_escape(False)),
        patch("src.hrhelpdesk.handlers.escalate_chain", _mock_escalate_chain()),
    ):
        first = await graph.aemit_event(
            thread_id=thread_id,
            source="human",
            event_type="message",
            input_message="My paycheck is wrong",
        )
        second = await graph.aemit_event(
            thread_id=thread_id,
            source="human",
            event_type="message",
            input_message="My paycheck is wrong again",
        )

    assert first["status"] == "ok"
    assert len(first.get("open_tickets", [])) == 1
    assert any("TICKET-" in m for m in first.get("output_messages", []))
    assert second["status"] == "ok"
    assert len(second.get("open_tickets", [])) == 1


@pytest.mark.asyncio
async def test_escalate_create_ticket_called_once():
    """Handler owns the side effect — create_ticket exactly once per decision."""
    graph = build_graph()
    thread_id = str(uuid4())
    create_calls: list[tuple[str, str]] = []

    def _tracking_create(subject: str, body: str) -> str:
        create_calls.append((subject, body))
        return f"TICKET-{len(create_calls)}"

    with (
        patch(
            "src.hrhelpdesk.handlers.classify_utterance",
            return_value=_decision("escalate"),
        ),
        patch("src.hrhelpdesk.handlers.run_escape", return_value=_escape(False)),
        patch("src.hrhelpdesk.handlers.escalate_chain", _mock_escalate_chain()),
        patch("src.hrhelpdesk.handlers.create_ticket", side_effect=_tracking_create),
    ):
        result = await graph.aemit_event(
            thread_id=thread_id,
            source="human",
            event_type="message",
            input_message="Please open a ticket — paycheck short $200",
        )

    assert result["status"] == "ok"
    assert create_calls == [("Pay issue", "Paycheck wrong")]
    assert result.get("open_tickets") == ["TICKET-1"]
    assert any("TICKET-1" in m for m in result.get("output_messages", []))


@pytest.mark.asyncio
async def test_booking_sticky_two_turn_confirm():
    graph = build_graph()
    thread_id = str(uuid4())

    with (
        patch(
            "src.hrhelpdesk.handlers.classify_utterance",
            return_value=_decision("booking"),
        ),
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
            input_message="Book a desk Monday NYC",
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
            input_message="Window seat please",
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
        patch(
            "src.hrhelpdesk.handlers.classify_utterance",
            return_value=_decision("booking"),
        ),
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
            input_message="Book a desk",
        )

    with (
        patch(
            "src.hrhelpdesk.handlers.classify_utterance",
            return_value=_decision("faq"),
        ),
        patch("src.hrhelpdesk.handlers.run_escape", return_value=_escape(True)),
        patch("src.hrhelpdesk.handlers.faq_agent", _mock_faq_agent()),
    ):
        result = await graph.aemit_event(
            thread_id=thread_id,
            source="human",
            event_type="message",
            input_message="Actually how much PTO?",
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
    """topic_timeout delivered while genuinely parked at topic_booking must be
    handled by a short-circuit, never by invoking run_escape/booking_agent
    with empty input (regression for the sticky-topic system-event incident
    documented in CLAUDE.md)."""
    graph = build_graph()
    thread_id = str(uuid4())

    with (
        patch(
            "src.hrhelpdesk.handlers.classify_utterance",
            return_value=_decision("booking"),
        ),
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
            input_message="Book a desk Monday NYC",
        )

    assert turn1["status"] == "ok"
    assert turn1.get("current_state") == State.TOPIC_BOOKING.value
    assert turn1.get("active_topic") == "booking"

    def _forbidden(*_a, **_k):
        raise AssertionError("LLM/tool call must not run for a system-sourced turn")

    with (
        patch("src.hrhelpdesk.handlers.run_escape", side_effect=_forbidden),
        patch("src.hrhelpdesk.handlers.booking_agent", MagicMock(invoke=_forbidden)),
    ):
        result = await graph.aemit_event(
            thread_id=thread_id,
            source="system",
            event_type="topic_timeout",
            event_id="evt-timeout",
        )

    assert result["status"] == "ok"
    assert result.get("active_topic") is None
    assert result.get("topic_data") == {}
    assert result.get("current_state") == State.IDLE.value
    assert any("timed out" in m.lower() for m in result.get("output_messages", []))


@pytest.mark.asyncio
async def test_ticket_resolved_during_clarify_short_circuits():
    """ticket_resolved is legal regardless of current_state (Graph._is_system_event_legal
    only checks open_tickets) and can legally arrive while parked at clarify —
    must not invoke the router LLM with empty input."""
    graph = build_graph()
    thread_id = str(uuid4())

    with (
        patch(
            "src.hrhelpdesk.handlers.classify_utterance",
            return_value=_decision("escalate"),
        ),
        patch("src.hrhelpdesk.handlers.run_escape", return_value=_escape(False)),
        patch("src.hrhelpdesk.handlers.escalate_chain", _mock_escalate_chain()),
    ):
        first = await graph.aemit_event(
            thread_id=thread_id,
            source="human",
            event_type="message",
            input_message="My paycheck is wrong",
        )
    ticket_id = first["open_tickets"][0]

    with (
        patch(
            "src.hrhelpdesk.handlers.classify_utterance",
            return_value=_decision("unclear", 0.3),
        ),
        patch("src.hrhelpdesk.handlers.run_escape", return_value=_escape(False)),
    ):
        second = await graph.aemit_event(
            thread_id=thread_id,
            source="human",
            event_type="message",
            input_message="hmm",
        )

    assert second.get("current_state") == State.CLARIFY.value
    assert second.get("pending_clarify") is True

    def _forbidden(*_a, **_k):
        raise AssertionError("router must not run for a system-sourced turn")

    with patch("src.hrhelpdesk.handlers.classify_utterance", side_effect=_forbidden):
        result = await graph.aemit_event(
            thread_id=thread_id,
            source="system",
            event_type="ticket_resolved",
            event_id="evt-ticket-1",
            payload={"ticket_id": ticket_id},
        )

    assert result["status"] == "ok"
    assert ticket_id not in (result.get("open_tickets") or [])
    assert any(ticket_id in m for m in result.get("output_messages", []))
    assert result.get("current_state") == State.IDLE.value
    assert any("resolved" in m.lower() for m in result.get("output_messages", []))


@pytest.mark.asyncio
async def test_ticket_resolved_from_idle_removes_open_ticket():
    """ticket_id must be a HelpdeskState channel or notify reads None and
    never removes the id from open_tickets."""
    graph = build_graph()
    thread_id = str(uuid4())

    with (
        patch(
            "src.hrhelpdesk.handlers.classify_utterance",
            return_value=_decision("escalate"),
        ),
        patch("src.hrhelpdesk.handlers.run_escape", return_value=_escape(False)),
        patch("src.hrhelpdesk.handlers.escalate_chain", _mock_escalate_chain()),
    ):
        first = await graph.aemit_event(
            thread_id=thread_id,
            source="human",
            event_type="message",
            input_message="My paycheck is wrong — please open a ticket",
        )
    ticket_id = first["open_tickets"][0]
    assert first.get("current_state") == State.IDLE.value

    result = await graph.aemit_event(
        thread_id=thread_id,
        source="system",
        event_type="ticket_resolved",
        event_id="evt-ticket-idle",
        payload={"ticket_id": ticket_id},
    )

    assert result["status"] == "ok"
    assert ticket_id not in (result.get("open_tickets") or [])
    assert result.get("output_messages") == [
        f"Good news — ticket {ticket_id} has been resolved."
    ]
    assert result.get("current_state") == State.IDLE.value


@pytest.mark.asyncio
async def test_faq_path_never_calls_booking_tools():
    graph = build_graph()
    thread_id = str(uuid4())

    def _forbidden_booking(*_a, **_k):
        raise AssertionError("booking tools must not run on FAQ path")

    with (
        patch(
            "src.hrhelpdesk.handlers.classify_utterance",
            return_value=_decision("faq"),
        ),
        patch("src.hrhelpdesk.handlers.run_escape", return_value=_escape(False)),
        patch("src.hrhelpdesk.handlers.faq_agent", _mock_faq_agent()),
        patch("src.hrhelpdesk.services.confirm_booking", side_effect=_forbidden_booking),
    ):
        result = await graph.aemit_event(
            thread_id=thread_id,
            source="human",
            event_type="message",
            input_message="PTO policy?",
        )

    assert result["status"] == "ok"
    assert len(_booking_store) == 0
