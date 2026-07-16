"""Handler functions for the onboarding pipeline.

Each handler executes business logic for a state and must:
  1. Read from state dict
  2. Process the data
  3. Return ONLY the fields it's updating (a partial delta)

current_state and session_status are stamped centrally by EngineGraph._dispatch_handler
— handlers never set them. audit_trail/output_messages are reducer-backed
(operator.add), so handlers return only the new entry/entries, not the
accumulated list.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.engine.chains import chain_field
from src.engine.handler_registry import handler

from .session_state import OnboardingState
from .state_transitions import State

log = logging.getLogger(__name__)


def _log_enter(handler_name: str, state: OnboardingState) -> None:
    """Trace handler entry: current park/progress state and event context."""
    log.info(
        "[HANDLER] ▶ %s  current_state=%s  event_source=%s  event_type=%s  "
        "session_id=%s  turn=%s",
        handler_name,
        state.get("current_state"),
        state.get("current_event_source", "human"),
        state.get("current_event_type", "message"),
        state.get("session_id") or "(none)",
        state.get("turn_number", 0),
    )


def _log_exit(handler_name: str, delta: dict[str, Any]) -> dict[str, Any]:
    """Trace the partial state delta returned (what this handler transforms)."""
    keys = sorted(delta.keys())
    summary: dict[str, Any] = {}
    for key in keys:
        value = delta[key]
        if key == "output_messages" and isinstance(value, list):
            summary[key] = [str(m)[:80] for m in value]
        elif key == "audit_trail" and isinstance(value, list):
            summary[key] = value
        elif key == "new_hire_details" and isinstance(value, dict):
            summary[key] = {
                k: value.get(k) for k in ("full_name", "role", "start_date") if k in value
            }
        else:
            summary[key] = value
    log.info("[HANDLER] ◀ %s  delta_keys=%s  delta=%s", handler_name, keys, summary)
    return delta


# ─────────────────────────────────────────────────────────────────────────────
# HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

@handler(
    state=State.COLLECT.value,
    waits_for_input=True,
    wait_kind="human",
    description="Collect new-hire details via conversation",
)
async def handle_collect(state: OnboardingState) -> OnboardingState:
    """Gather new-hire details through a tool-calling agent.

    Tool execution — not model prose — performs the transition: this node
    intercepts the submit_new_hire tool call and applies the guarded,
    idempotent state delta itself. The router always proposes WELCOME_SENT
    next regardless of whether details are complete (code routing, no LLM
    in the transition decision — design spec §2.3); it's
    check_new_hire_details_complete (guardrails.py) that actually
    implements the collect --> collect self-loop when they aren't.

    Args:
        state: OnboardingState with input_message set for this turn

    Returns:
        Delta with new_hire_details if the agent called submit_new_hire
        this turn, or output_messages asking for whatever's still missing
    """
    from .chains import collect_agent

    _log_enter("collect", state)
    input_message = state.get("input_message") or ""
    log.info("[HANDLER] collect  input=%r", input_message[:80])

    try:
        result = await collect_agent.ainvoke(
            {"messages": [HumanMessage(content=input_message)]}
        )
    except Exception as e:
        log.error("[HANDLER] collect agent failed: %s", e)
        return _log_exit(
            "collect",
            {
                "handler_status": "error",
                "error_message": str(e),
                "audit_trail": [f"collect failed: {e}"],
            },
        )

    agent_messages = result.get("messages", [])

    tool_result = next(
        (m for m in reversed(agent_messages) if isinstance(m, ToolMessage)),
        None,
    )
    if tool_result is not None:
        details = (
            json.loads(tool_result.content)
            if isinstance(tool_result.content, str)
            else tool_result.content
        )
        log.info(
            "[HANDLER] collect  tool=submit_new_hire  details=%s",
            {k: details.get(k) for k in ("full_name", "role", "start_date")},
        )
        return _log_exit(
            "collect",
            {
                "handler_status": "ok",
                "new_hire_details": details,
                "audit_trail": [f"collect: details submitted for {details.get('full_name')}"],
            },
        )

    last_ai = next(
        (m for m in reversed(agent_messages) if isinstance(m, AIMessage) and m.content),
        None,
    )
    log.info(
        "[HANDLER] collect  no tool call yet — clarifying (ai=%r)",
        (last_ai.content[:80] if last_ai else None),
    )
    return _log_exit(
        "collect",
        {
            "handler_status": "ok",
            "output_messages": [last_ai.content] if last_ai else [],
            "audit_trail": ["collect: awaiting more details"],
        },
    )


@handler(state=State.WELCOME_SENT.value, waits_for_input=False, description="Send welcome message")
def handle_welcome_sent(state: OnboardingState) -> OnboardingState:
    """Send the welcome message, once.

    Args:
        state: OnboardingState with new_hire_details set

    Returns:
        Delta with welcome_sent guard flag and the welcome output message
    """
    _log_enter("welcome_sent", state)
    if state.get("welcome_sent"):
        log.info("[HANDLER] welcome_sent  skip — already sent (idempotent)")
        return _log_exit(
            "welcome_sent",
            {"audit_trail": ["welcome_sent: already sent, skipping (idempotent)"]},
        )

    name = (state.get("new_hire_details") or {}).get("full_name", "there")
    log.info("[HANDLER] welcome_sent  sending welcome to %r", name)
    return _log_exit(
        "welcome_sent",
        {
            "welcome_sent": True,
            "output_messages": [f"Welcome aboard, {name}! We've started your onboarding."],
            "audit_trail": ["welcome_sent: sent"],
        },
    )


@handler(
    state=State.AWAIT_DOCUMENTS_SIGNED.value,
    waits_for_input=True,
    wait_kind="system_event",
    expected_events=["document_signed", "timeout_escalation"],
    description="Wait for signed documents (system event) or timeout",
)
def handle_await_documents_signed(state: OnboardingState) -> OnboardingState:
    """Being resumed does not mean the happy event fired — it may be a
    timeout_escalation resume that's merely legal, not desired (design
    spec §3). The actual redirect to ESCALATED happens one hop later via
    the guardrail diversion on IT_PROVISIONED; this only fixes what gets
    recorded here.
    """
    _log_enter("await_documents_signed", state)
    event_type = state.get("current_event_type")
    if event_type == "timeout_escalation":
        log.info("[HANDLER] await_documents_signed  branch=timeout_escalation")
        return _log_exit(
            "await_documents_signed",
            {"audit_trail": ["await_documents_signed: timeout, no signature received"]},
        )
    log.info("[HANDLER] await_documents_signed  branch=document_signed (or other legal)")
    return _log_exit(
        "await_documents_signed",
        {"audit_trail": ["await_documents_signed: document_signed event received"]},
    )


@handler(
    state=State.IT_PROVISIONED.value,
    waits_for_input=False,
    description="Provision IT account (username selection)",
)
async def handle_it_provisioned(state: OnboardingState) -> OnboardingState:
    """Select a username prefix, once.

    Args:
        state: OnboardingState with new_hire_details set

    Returns:
        Delta with it_provisioned guard flag and the selected username_prefix
    """
    _log_enter("it_provisioned", state)
    if state.get("it_provisioned"):
        log.info("[HANDLER] it_provisioned  skip — already provisioned (idempotent)")
        return _log_exit(
            "it_provisioned",
            {
                "handler_status": "ok",
                "audit_trail": ["it_provisioned: already provisioned, skipping (idempotent)"],
            },
        )

    from .chains import username_chain

    details = state.get("new_hire_details") or {}
    log.info("[HANDLER] it_provisioned  selecting username from details=%s", details)
    try:
        result = await username_chain.ainvoke({"input": str(details)})
        prefix = chain_field(result, "username_prefix", "user")
    except Exception as e:
        log.error("[HANDLER] username chain failed: %s", e)
        return _log_exit(
            "it_provisioned",
            {
                "handler_status": "error",
                "error_message": str(e),
                "audit_trail": [f"it_provisioned failed: {e}"],
            },
        )

    log.info("[HANDLER] it_provisioned  username_prefix=%r", prefix)
    return _log_exit(
        "it_provisioned",
        {
            "handler_status": "ok",
            "it_provisioned": True,
            "username_prefix": prefix,
            "audit_trail": [f"it_provisioned: username={prefix}"],
        },
    )


@handler(
    state=State.AWAIT_HARDWARE_DELIVERED.value,
    waits_for_input=True,
    wait_kind="system_event",
    expected_events=["hardware_delivered", "timeout_escalation"],
    description="Wait for hardware delivery confirmation (system event) or timeout",
)
def handle_await_hardware_delivered(state: OnboardingState) -> OnboardingState:
    """Same current_event_type branching as handle_await_documents_signed
    (design spec §3) — hardware_tracking_id, if supplied, already landed
    in state via aemit_event's payload merge before this handler ran."""
    _log_enter("await_hardware_delivered", state)
    event_type = state.get("current_event_type")
    tracking = state.get("hardware_tracking_id")
    if event_type == "timeout_escalation":
        log.info("[HANDLER] await_hardware_delivered  branch=timeout_escalation")
        return _log_exit(
            "await_hardware_delivered",
            {"audit_trail": ["await_hardware_delivered: timeout, hardware not delivered"]},
        )
    log.info(
        "[HANDLER] await_hardware_delivered  branch=hardware_delivered  tracking_id=%r",
        tracking,
    )
    return _log_exit(
        "await_hardware_delivered",
        {"audit_trail": ["await_hardware_delivered: hardware_delivered event received"]},
    )


@handler(
    state=State.SCHEDULE_SENT.value,
    waits_for_input=False,
    description="Send hardware delivery/setup schedule",
)
def handle_schedule_sent(state: OnboardingState) -> OnboardingState:
    """Send the setup schedule message, once."""
    _log_enter("schedule_sent", state)
    if state.get("schedule_sent"):
        log.info("[HANDLER] schedule_sent  skip — already sent (idempotent)")
        return _log_exit(
            "schedule_sent",
            {"audit_trail": ["schedule_sent: already sent, skipping (idempotent)"]},
        )

    log.info("[HANDLER] schedule_sent  sending setup schedule")
    return _log_exit(
        "schedule_sent",
        {
            "schedule_sent": True,
            "output_messages": [
                "Your hardware has been delivered and your setup schedule is confirmed."
            ],
            "audit_trail": ["schedule_sent: sent"],
        },
    )


@handler(state=State.COMPLETE.value, waits_for_input=False, description="Mark onboarding complete")
def handle_complete(state: OnboardingState) -> OnboardingState:
    """Notify HR that onboarding finished, once."""
    _log_enter("complete", state)
    if state.get("hr_notified"):
        log.info("[HANDLER] complete  skip — HR already notified (idempotent)")
        return _log_exit(
            "complete",
            {"audit_trail": ["complete: HR already notified, skipping (idempotent)"]},
        )

    log.info("[HANDLER] complete  notifying HR")
    return _log_exit(
        "complete",
        {
            "hr_notified": True,
            "output_messages": ["Onboarding complete! HR has been notified."],
            "audit_trail": ["COMPLETE"],
        },
    )


@handler(
    state=State.ESCALATED.value, waits_for_input=False, description="Handle escalation (timeout)"
)
def handle_escalated(state: OnboardingState) -> OnboardingState:
    """Terminal escalation state — no guard flag needed, nothing to
    idempotently protect (a run only ever reaches ESCALATED once)."""
    _log_enter("escalated", state)
    log.info(
        "[HANDLER] escalated  terminal diversion  prior_event_type=%s",
        state.get("current_event_type"),
    )
    return _log_exit(
        "escalated",
        {
            "output_messages": [
                "This onboarding step has been escalated to a human for follow-up."
            ],
            "audit_trail": ["ESCALATED"],
        },
    )


@handler(
    state=State.ERROR.value,
    waits_for_input=False,
    description="Handle onboarding pipeline error",
)
def handle_error(state: OnboardingState) -> OnboardingState:
    """Terminal error state, mirroring docprocessing's handle_error."""
    _log_enter("error", state)
    log.error(
        "[HANDLER] onboarding ERROR  reason=%s",
        state.get("error_message", "unknown"),
    )
    return _log_exit(
        "error",
        {
            "audit_trail": [f"ERROR: {state.get('error_message', 'unknown')}"],
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# HANDLER MAP (exported for use in state machine graph)
# ─────────────────────────────────────────────────────────────────────────────

handler_map = {
    State.COLLECT: handle_collect,
    State.WELCOME_SENT: handle_welcome_sent,
    State.AWAIT_DOCUMENTS_SIGNED: handle_await_documents_signed,
    State.IT_PROVISIONED: handle_it_provisioned,
    State.AWAIT_HARDWARE_DELIVERED: handle_await_hardware_delivered,
    State.SCHEDULE_SENT: handle_schedule_sent,
    State.COMPLETE: handle_complete,
    State.ESCALATED: handle_escalated,
    State.ERROR: handle_error,
}

__all__ = [
    "handle_collect",
    "handle_welcome_sent",
    "handle_await_documents_signed",
    "handle_it_provisioned",
    "handle_await_hardware_delivered",
    "handle_schedule_sent",
    "handle_complete",
    "handle_escalated",
    "handle_error",
    "handler_map",
]
