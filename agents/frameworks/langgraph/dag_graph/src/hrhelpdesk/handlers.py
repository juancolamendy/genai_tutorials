"""Handler functions for the HR helpdesk chatbot."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.engine.event_ledger import effect_key
from src.engine.handler_registry import handler
from src.engine.message_hygiene import close_topic_delta

from .chains import (
    booking_agent,
    escalate_agent,
    faq_agent,
    run_escape,
    run_router,
)
from .providers import confirm_booking, create_ticket, retrieve_policy
from .session_state import HelpdeskState
from .state_transitions import State

log = logging.getLogger(__name__)

ROUTER_CONFIDENCE_THRESHOLD = 0.7

_ledger: Any = None


def set_ledger(ledger: Any) -> None:
    """Inject the graph's EventLedger for sync idempotency checks in handlers."""
    global _ledger
    _ledger = ledger


def _sync_is_processed(ledger: Any, key: str) -> bool:
    return ledger._marker_path(key).exists()


def _sync_mark(ledger: Any, key: str) -> None:
    ledger._mark_processed_sync(key)


def _content_hash(*parts: str) -> str:
    return hashlib.sha256(":".join(parts).encode()).hexdigest()[:16]


def _log_enter(handler_name: str, state: HelpdeskState) -> None:
    log.info(
        "[HANDLER] ▶ %s  current_state=%s  active_topic=%s  event=%s",
        handler_name,
        state.get("current_state"),
        state.get("active_topic"),
        state.get("current_event_type"),
    )


def _log_exit(handler_name: str, delta: dict[str, Any]) -> dict[str, Any]:
    log.info("[HANDLER] ◀ %s  delta_keys=%s", handler_name, sorted(delta.keys()))
    return delta


def _last_ai_content(messages: list[Any]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return str(msg.content)
    return ""


def _find_tool_message(messages: list[Any], tool_name: str) -> Optional[ToolMessage]:
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == tool_name:
            return msg
    return None


def _find_tool_call(messages: list[Any], tool_name: str) -> Optional[dict[str, Any]]:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                if call.get("name") == tool_name:
                    return call
    return None


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

    avail_call = _find_tool_call(agent_messages, "check_desk_availability")
    if avail_call:
        args = avail_call.get("args") or {}
        if args.get("date"):
            topic_data["date"] = args["date"]
        if args.get("location"):
            topic_data["location"] = args["location"]

    confirm_call = _find_tool_call(agent_messages, "confirm_booking")
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
            if _ledger and _sync_is_processed(_ledger, key):
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
                    _sync_mark(_ledger, key)
                delta["bookings"] = bookings
            topic_data["booking_confirmed"] = True
            summary = f"Desk booked for {date} at {location} ({seat_pref})."
            delta.update(close_topic_delta(prior_messages, summary))
            delta["output_messages"] = [
                _last_ai_content(agent_messages) or summary,
            ]
            return delta

    last_ai = _last_ai_content(agent_messages)
    if last_ai:
        delta["output_messages"] = [last_ai]

    delta["topic_data"] = topic_data
    return delta


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
    _log_enter("idle", state)
    event_source = state.get("current_event_source", "human")

    if event_source == "system":
        return _log_exit(
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
        return _log_exit("idle", _process_booking_turn(state, input_message))

    if active_topic in ("faq", "escalate"):
        return _log_exit(
            "idle",
            {"audit_trail": [f"idle: continue sticky lane {active_topic}"]},
        )

    try:
        router_out = run_router(input_message)
    except Exception as exc:
        log.error("[HANDLER] router failed: %s", exc)
        return _log_exit(
            "idle",
            {
                "handler_status": "error",
                "error_message": str(exc),
                "audit_trail": [f"idle router failed: {exc}"],
            },
        )

    topic = router_out.topic.value
    confidence = router_out.confidence
    base: dict[str, Any] = {
        "router_confidence": confidence,
        "semantic_context": {"router_topic": topic},
        "pending_clarify": False,
        "handler_status": "ok",
    }

    if topic == "unclear" or confidence < ROUTER_CONFIDENCE_THRESHOLD:
        base["pending_clarify"] = True
        base["active_topic"] = None
        base["output_messages"] = [
            "I'm not sure I understood. Would you like help with a policy question "
            "(FAQ), escalating an issue, or booking a desk?"
        ]
        base["audit_trail"] = ["idle: routed to clarify"]
        return _log_exit("idle", base)

    base["active_topic"] = topic
    base["topic_started_at"] = datetime.now(timezone.utc).isoformat()
    base["topic_data"] = {}
    base["audit_trail"] = [f"idle: routed to {topic}"]

    if topic == "booking":
        booking_delta = _process_booking_turn({**state, **base}, input_message)
        base.update(booking_delta)
        return _log_exit("idle", base)

    return _log_exit("idle", base)


@handler(state=State.ROUTE.value, waits_for_input=False, description="Deterministic route pass-through")
def handle_route(state: HelpdeskState) -> HelpdeskState:
    _log_enter("route", state)
    return _log_exit("route", {"audit_trail": ["route: pass-through"]})


@handler(
    state=State.HUB_CLARIFY.value,
    waits_for_input=True,
    wait_kind="human",
    description="Disambiguate unclear routing",
)
def handle_hub_clarify(state: HelpdeskState) -> HelpdeskState:
    _log_enter("hub_clarify", state)
    return _log_exit(
        "hub_clarify",
        {
            "pending_clarify": True,
            "output_messages": [
                "Please tell me if you need FAQ help, want to escalate an issue, "
                "or book a desk."
            ],
            "audit_trail": ["hub_clarify: asked for lane"],
        },
    )


@handler(state=State.TOPIC_FAQ.value, waits_for_input=False, description="Policy Q&A specialist")
def handle_topic_faq(state: HelpdeskState) -> HelpdeskState:
    _log_enter("topic_faq", state)
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
        return _log_exit(
            "topic_faq",
            {
                "handler_status": "error",
                "error_message": str(exc),
                "audit_trail": [f"faq failed: {exc}"],
            },
        )

    answer = _last_ai_content(result.get("messages", [])) or "I could not find an answer."
    delta = close_topic_delta(messages, "FAQ answered.")
    delta.update(
        {
            "handler_status": "ok",
            "output_messages": [answer],
            "audit_trail": ["topic_faq: answered"],
        }
    )
    return _log_exit("topic_faq", delta)


@handler(state=State.TOPIC_ESCALATE.value, waits_for_input=False, description="Ticket specialist")
def handle_topic_escalate(state: HelpdeskState) -> HelpdeskState:
    _log_enter("topic_escalate", state)
    input_message = state.get("input_message") or ""
    messages = state.get("messages") or []

    try:
        result = escalate_agent.invoke({"messages": [HumanMessage(content=input_message)]})
    except Exception as exc:
        log.error("[HANDLER] escalate agent failed: %s", exc)
        return _log_exit(
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

    tool_call = _find_tool_call(agent_messages, "create_ticket_tool")
    if tool_call:
        args = tool_call.get("args") or {}
        subject = str(args.get("subject", "HR issue"))
        body = str(args.get("body", input_message))
        thread_id = state.get("session_id") or ""
        key = effect_key(thread_id, "ticket", _content_hash(subject, body))
        open_tickets = list(state.get("open_tickets") or [])

        if _ledger and _sync_is_processed(_ledger, key):
            log.info("[HANDLER] ticket create skipped (ledger)")
        else:
            ticket_id = create_ticket(subject, body)
            open_tickets.append(ticket_id)
            if _ledger:
                _sync_mark(_ledger, key)
            delta["open_tickets"] = open_tickets

    reply = _last_ai_content(agent_messages)
    close = close_topic_delta(messages, "Ticket escalated.")
    delta.update(close)
    if reply:
        delta["output_messages"] = [reply]
    return _log_exit("topic_escalate", delta)


@handler(
    state=State.TOPIC_BOOKING.value,
    waits_for_input=True,
    wait_kind="human",
    description="Sticky desk booking specialist",
)
def handle_topic_booking(state: HelpdeskState) -> HelpdeskState:
    _log_enter("topic_booking", state)
    input_message = state.get("input_message") or ""
    messages = state.get("messages") or []

    try:
        if run_escape(input_message).escape:
            cleared = close_topic_delta(messages, "User changed topic.")
            cleared["pending_clarify"] = False
            try:
                router_out = run_router(input_message)
            except Exception as exc:
                return _log_exit(
                    "topic_booking",
                    {
                        "handler_status": "error",
                        "error_message": str(exc),
                        **cleared,
                    },
                )
            topic = router_out.topic.value
            confidence = router_out.confidence
            cleared["router_confidence"] = confidence
            cleared["semantic_context"] = {"router_topic": topic}
            if topic == "unclear" or confidence < ROUTER_CONFIDENCE_THRESHOLD:
                cleared["pending_clarify"] = True
                cleared["output_messages"] = [
                    "I'm not sure I understood. Would you like FAQ, escalate, or book a desk?"
                ]
            else:
                cleared["active_topic"] = topic
                cleared["topic_started_at"] = datetime.now(timezone.utc).isoformat()
                cleared["topic_data"] = {}
            cleared["audit_trail"] = ["topic_booking: escape — re-routed"]
            return _log_exit("topic_booking", cleared)
    except Exception as exc:
        log.error("[HANDLER] escape check failed: %s", exc)

    delta = _process_booking_turn(state, input_message)
    return _log_exit("topic_booking", delta)


@handler(state=State.NOTIFY_USER.value, waits_for_input=False, description="Render system events")
def handle_notify_user(state: HelpdeskState) -> HelpdeskState:
    _log_enter("notify_user", state)
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
        output.append(f"Your booking {booking_id} was cancelled by the system.")

    elif event_type == "topic_timeout":
        delta.update(close_topic_delta(messages, "Booking session timed out."))
        output.append("Your desk booking session timed out. Start again anytime.")

    if output:
        delta["output_messages"] = output

    return _log_exit("notify_user", delta)


@handler(state=State.ERROR.value, waits_for_input=False, description="Terminal error")
def handle_error(state: HelpdeskState) -> HelpdeskState:
    _log_enter("error", state)
    return _log_exit(
        "error",
        {"audit_trail": [f"ERROR: {state.get('error_message', 'unknown')}"]},
    )


handler_map = {
    State.ROUTE: handle_route,
    State.HUB_CLARIFY: handle_hub_clarify,
    State.TOPIC_FAQ: handle_topic_faq,
    State.TOPIC_ESCALATE: handle_topic_escalate,
    State.TOPIC_BOOKING: handle_topic_booking,
    State.NOTIFY_USER: handle_notify_user,
    State.IDLE: handle_idle,
    State.ERROR: handle_error,
}

__all__ = [
    "set_ledger",
    "handle_idle",
    "handle_route",
    "handle_hub_clarify",
    "handle_topic_faq",
    "handle_topic_escalate",
    "handle_topic_booking",
    "handle_notify_user",
    "handle_error",
    "handler_map",
    "ROUTER_CONFIDENCE_THRESHOLD",
]
