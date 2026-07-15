"""Handler functions for the HR helpdesk chatbot."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage

from src.engine.chat_engine_graph import ChatEngineGraph
from src.engine.event_ledger import effect_key
from src.engine.handler_registry import handler
from src.engine.utils import (
    close_topic_delta,
    content_hash,
    find_tool_call,
    last_ai_content,
    ledger_is_processed_sync,
    ledger_mark_processed_sync,
    log_handler_enter,
    log_handler_exit,
)

from .chains import (
    booking_agent,
    escalate_agent,
    faq_agent,
)
from .services import confirm_booking, create_ticket, retrieve_policy
from .session_state import HelpdeskState
from .state_transitions import State

log = logging.getLogger(__name__)

# Injected by build_graph() — same Graph instance that owns topic_router /
# escape_checker / confidence_threshold (avoids a second ChatEngineGraph).
_chat_graph: ChatEngineGraph | None = None
_ledger: Any = None


def set_chat_graph(graph: ChatEngineGraph) -> None:
    """Inject the domain Graph used for classify / escape / topic deltas."""
    global _chat_graph
    _chat_graph = graph


def set_ledger(ledger: Any) -> None:
    """Inject the graph's EventLedger for sync idempotency checks in handlers."""
    global _ledger
    _ledger = ledger


def _require_chat_graph() -> ChatEngineGraph:
    if _chat_graph is None:
        raise RuntimeError(
            "Helpdesk handlers require build_graph() first "
            "(set_chat_graph was never called)."
        )
    return _chat_graph


def classify_utterance(input_message: str, history: Any = None):
    """Patchable wrapper around the bound Graph's classifier."""
    return _require_chat_graph().classify_utterance(input_message, history)


def topic_decision_to_delta(decision: Any, *, source: str = "chat"):
    """Patchable wrapper around the bound Graph's topic delta helper."""
    return _require_chat_graph().topic_decision_to_delta(decision, source=source)


def run_escape(text: str):
    """Patchable wrapper around the bound Graph's escape checker."""
    return _require_chat_graph().run_escape(text)


CLARIFY_PROMPT = (
    "I'm not sure I understood. Would you like help with a policy question "
    "(FAQ), escalating an issue, or booking a desk?"
)


def _process_booking_turn(state: HelpdeskState, user_text: str) -> dict[str, Any]:
    """Shared booking specialist logic for IDLE bootstrap and TOPIC_BOOKING."""
    topic_data = dict(state.get("topic_data") or {})
    prior_messages = state.get("messages") or []

    try:
        result = booking_agent.invoke(
            {"messages": [HumanMessage(content=user_text)]}
        )
    except Exception as exc:
        log.error("[HANDLER] booking agent failed: %s", exc)
        return {
            "handler_status": "error",
            "error_message": str(exc),
            "audit_trail": [f"booking failed: {exc}"],
        }

    agent_messages = result.get("messages", [])
    delta: dict[str, Any] = {
        "handler_status": "ok",
        "topic_data": topic_data,
        "audit_trail": ["booking: agent turn"],
    }

    avail_call = find_tool_call(agent_messages, "check_desk_availability")
    if avail_call:
        args = avail_call.get("args") or {}
        if args.get("date"):
            topic_data["date"] = args["date"]
        if args.get("location"):
            topic_data["location"] = args["location"]

    confirm_call = find_tool_call(agent_messages, "confirm_booking")
    if confirm_call:
        args = confirm_call.get("args") or {}
        date = args.get("date") or topic_data.get("date")
        location = args.get("location") or topic_data.get("location")
        seat_pref = args.get("seat_pref") or topic_data.get("seat_pref")
        if date:
            topic_data["date"] = date
        if location:
            topic_data["location"] = location
        if seat_pref:
            topic_data["seat_pref"] = seat_pref

        if date and location and seat_pref:
            thread_id = state.get("session_id") or ""
            key = effect_key(thread_id, "booking", date, location)
            bookings = list(state.get("bookings") or [])
            if _ledger and ledger_is_processed_sync(_ledger, key):
                log.info("[HANDLER] booking confirm skipped (ledger)")
            else:
                booking_id = confirm_booking(date, location, seat_pref)
                bookings.append(
                    {
                        "booking_id": booking_id,
                        "date": date,
                        "location": location,
                        "seat_pref": seat_pref,
                        "status": "confirmed",
                    }
                )
                if _ledger:
                    ledger_mark_processed_sync(_ledger, key)
                delta["bookings"] = bookings
            topic_data["booking_confirmed"] = True
            summary = f"Desk booked for {date} at {location} ({seat_pref})."
            delta.update(close_topic_delta(prior_messages, summary))
            delta["output_messages"] = [
                last_ai_content(agent_messages) or summary,
            ]
            return delta

    last_ai = last_ai_content(agent_messages)
    if last_ai:
        delta["output_messages"] = [last_ai]

    delta["topic_data"] = topic_data
    return delta


def _route_human_message(
    state: HelpdeskState,
    input_message: str,
    *,
    source: str,
) -> dict[str, Any]:
    """Semantic-route a human utterance into a lane (or stay in clarify).

    Shared by IDLE (first classification) and CLARIFY (user's answer
    after a disambiguation ask). Never guesses: unclear / low confidence
    keeps ``pending_clarify`` and re-asks. A router failure degrades to
    ``unclear`` (handled inside ``HelpdeskSemanticRouter.classify``) rather
    than surfacing as ``handler_status="error"``.
    """
    decision = classify_utterance(input_message, state.get("messages"))
    base = topic_decision_to_delta(decision, source=source)
    if base.get("pending_clarify"):
        base["output_messages"] = [CLARIFY_PROMPT]
        return base

    if decision.topic == "booking":
        booking_delta = _process_booking_turn({**state, **base}, input_message)
        base.update(booking_delta)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# HANDLERS
# ─────────────────────────────────────────────────────────────────────────────


@handler(
    state=State.IDLE.value,
    waits_for_input=True,
    wait_kind="either",
    expected_events=["ticket_resolved", "booking_cancelled_by_system", "topic_timeout"],
    description="Hub park — route human turns; accept legal system events",
)
def handle_idle(state: HelpdeskState) -> HelpdeskState:
    """Thin hub handler: escape check, semantic route, booking bootstrap."""
    log_handler_enter("idle", state)
    event_source = state.get("current_event_source", "human")

    if event_source == "system":
        return log_handler_exit(
            "idle",
            {
                "last_event": state.get("current_event_type"),
                "last_event_at": datetime.now(timezone.utc).isoformat(),
                "audit_trail": [f"idle: system event {state.get('current_event_type')}"],
            },
        )

    input_message = state.get("input_message") or ""
    active_topic = state.get("active_topic")
    messages = state.get("messages") or []

    if active_topic:
        try:
            if run_escape(input_message).escape:
                cleared = close_topic_delta(messages, "User changed topic.")
                cleared["pending_clarify"] = False
                cleared["audit_trail"] = ["idle: escape — topic cleared"]
                state = {**state, **cleared}
                active_topic = None
        except Exception as exc:
            log.error("[HANDLER] escape check failed: %s", exc)

    if active_topic == "booking":
        return log_handler_exit("idle", _process_booking_turn(state, input_message))

    if active_topic in ("faq", "escalate"):
        return log_handler_exit(
            "idle",
            {"audit_trail": [f"idle: continue sticky lane {active_topic}"]},
        )

    return log_handler_exit(
        "idle",
        _route_human_message(state, input_message, source="idle"),
    )


@handler(
    state=State.CLARIFY.value,
    waits_for_input=True,
    wait_kind="human",
    description="Re-route after unclear classification using the user's reply",
)
def handle_clarify(state: HelpdeskState) -> HelpdeskState:
    """Process the user's disambiguation reply — do not ignore input_message."""
    log_handler_enter("clarify", state)
    event_source = state.get("current_event_source", "human")

    if event_source == "system":
        return log_handler_exit(
            "clarify",
            {
                "last_event": state.get("current_event_type"),
                "last_event_at": datetime.now(timezone.utc).isoformat(),
                "audit_trail": [f"clarify: system event {state.get('current_event_type')}"],
            },
        )

    input_message = state.get("input_message") or ""
    return log_handler_exit(
        "clarify",
        _route_human_message(state, input_message, source="clarify"),
    )


@handler(state=State.TOPIC_FAQ.value, waits_for_input=False, description="Policy Q&A specialist")
def handle_topic_faq(state: HelpdeskState) -> HelpdeskState:
    log_handler_enter("topic_faq", state)
    input_message = state.get("input_message") or ""
    messages = state.get("messages") or []
    chunks = retrieve_policy(input_message)
    context = "\n\n".join(f"[{c['id']}] {c['text']}" for c in chunks)

    prompt = (
        f"Knowledge snippets:\n{context}\n\nUser question: {input_message}\n"
        "Answer using only the snippets above and cite source ids."
    )

    try:
        result = faq_agent.invoke({"messages": [HumanMessage(content=prompt)]})
    except Exception as exc:
        log.error("[HANDLER] faq agent failed: %s", exc)
        return log_handler_exit(
            "topic_faq",
            {
                "handler_status": "error",
                "error_message": str(exc),
                "audit_trail": [f"faq failed: {exc}"],
            },
        )

    answer = last_ai_content(result.get("messages", [])) or "I could not find an answer."
    delta = close_topic_delta(messages, "FAQ answered.")
    delta.update(
        {
            "handler_status": "ok",
            "output_messages": [answer],
            "audit_trail": ["topic_faq: answered"],
        }
    )
    return log_handler_exit("topic_faq", delta)


@handler(state=State.TOPIC_ESCALATE.value, waits_for_input=False, description="Ticket specialist")
def handle_topic_escalate(state: HelpdeskState) -> HelpdeskState:
    log_handler_enter("topic_escalate", state)
    input_message = state.get("input_message") or ""
    messages = state.get("messages") or []

    try:
        result = escalate_agent.invoke({"messages": [HumanMessage(content=input_message)]})
    except Exception as exc:
        log.error("[HANDLER] escalate agent failed: %s", exc)
        return log_handler_exit(
            "topic_escalate",
            {
                "handler_status": "error",
                "error_message": str(exc),
                "audit_trail": [f"escalate failed: {exc}"],
            },
        )

    agent_messages = result.get("messages", [])
    delta: dict[str, Any] = {
        "handler_status": "ok",
        "audit_trail": ["topic_escalate: processed"],
    }

    tool_call = find_tool_call(agent_messages, "create_ticket_tool")
    if tool_call:
        args = tool_call.get("args") or {}
        subject = str(args.get("subject", "HR issue"))
        body = str(args.get("body", input_message))
        thread_id = state.get("session_id") or ""
        key = effect_key(thread_id, "ticket", content_hash(subject, body))
        open_tickets = list(state.get("open_tickets") or [])

        if _ledger and ledger_is_processed_sync(_ledger, key):
            log.info("[HANDLER] ticket create skipped (ledger)")
        else:
            ticket_id = create_ticket(subject, body)
            open_tickets.append(ticket_id)
            if _ledger:
                ledger_mark_processed_sync(_ledger, key)
            delta["open_tickets"] = open_tickets

    reply = last_ai_content(agent_messages)
    close = close_topic_delta(messages, "Ticket escalated.")
    delta.update(close)
    if reply:
        delta["output_messages"] = [reply]
    return log_handler_exit("topic_escalate", delta)


@handler(
    state=State.TOPIC_BOOKING.value,
    waits_for_input=True,
    wait_kind="human",
    description="Sticky desk booking specialist",
)
def handle_topic_booking(state: HelpdeskState) -> HelpdeskState:
    log_handler_enter("topic_booking", state)
    event_source = state.get("current_event_source", "human")

    if event_source == "system":
        return log_handler_exit(
            "topic_booking",
            {
                "last_event": state.get("current_event_type"),
                "last_event_at": datetime.now(timezone.utc).isoformat(),
                "audit_trail": [f"topic_booking: system event {state.get('current_event_type')}"],
            },
        )

    input_message = state.get("input_message") or ""
    messages = state.get("messages") or []

    try:
        if run_escape(input_message).escape:
            cleared = close_topic_delta(messages, "User changed topic.")
            cleared["pending_clarify"] = False
            decision = classify_utterance(input_message, messages)
            routed = topic_decision_to_delta(
                decision, source="topic_booking"
            )
            cleared.update(routed)
            if cleared.get("pending_clarify"):
                cleared["output_messages"] = [CLARIFY_PROMPT]
            cleared["audit_trail"] = ["topic_booking: escape — re-routed"]
            return log_handler_exit("topic_booking", cleared)
    except Exception as exc:
        log.error("[HANDLER] escape check failed: %s", exc)

    delta = _process_booking_turn(state, input_message)
    return log_handler_exit("topic_booking", delta)


@handler(state=State.NOTIFY_USER.value, waits_for_input=False, description="Render system events")
def handle_notify_user(state: HelpdeskState) -> HelpdeskState:
    log_handler_enter("notify_user", state)
    event_type = state.get("current_event_type")
    messages = state.get("messages") or []
    delta: dict[str, Any] = {
        "last_event": None,
        "audit_trail": [f"notify_user: {event_type}"],
    }
    output: list[str] = []

    if event_type == "ticket_resolved":
        ticket_id = state.get("ticket_id")
        open_tickets = [t for t in (state.get("open_tickets") or []) if t != ticket_id]
        delta["open_tickets"] = open_tickets
        delta["ticket_id"] = None
        output.append(f"Good news — ticket {ticket_id} has been resolved.")

    elif event_type == "booking_cancelled_by_system":
        booking_id = state.get("booking_id")
        bookings = []
        for record in state.get("bookings") or []:
            if record.get("booking_id") == booking_id:
                bookings.append({**record, "status": "cancelled"})
            else:
                bookings.append(record)
        delta["bookings"] = bookings
        delta["booking_id"] = None
        output.append(f"Your booking {booking_id} was cancelled by the system.")

    elif event_type == "topic_timeout":
        delta.update(close_topic_delta(messages, "Booking session timed out."))
        output.append("Your desk booking session timed out. Start again anytime.")

    if output:
        delta["output_messages"] = output

    return log_handler_exit("notify_user", delta)


@handler(state=State.ERROR.value, waits_for_input=False, description="Terminal error")
def handle_error(state: HelpdeskState) -> HelpdeskState:
    log_handler_enter("error", state)
    return log_handler_exit(
        "error",
        {"audit_trail": [f"ERROR: {state.get('error_message', 'unknown')}"]},
    )


handler_map = {
    State.CLARIFY: handle_clarify,
    State.TOPIC_FAQ: handle_topic_faq,
    State.TOPIC_ESCALATE: handle_topic_escalate,
    State.TOPIC_BOOKING: handle_topic_booking,
    State.NOTIFY_USER: handle_notify_user,
    State.IDLE: handle_idle,
    State.ERROR: handle_error,
}

__all__ = [
    "set_ledger",
    "set_chat_graph",
    "classify_utterance",
    "topic_decision_to_delta",
    "run_escape",
    "handle_idle",
    "handle_clarify",
    "handle_topic_faq",
    "handle_topic_escalate",
    "handle_topic_booking",
    "handle_notify_user",
    "handle_error",
    "handler_map",
]
