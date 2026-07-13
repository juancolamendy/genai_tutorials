"""Tests for onboarding state handlers and graph wiring (design spec §3,
§9's handler-side counterpart, §2.2). collect_agent is mocked throughout —
no real LLM calls in unit tests, matching this codebase's precedent (no
existing test file invokes a real chain/agent directly).
"""

import importlib
from unittest.mock import MagicMock, patch

import pytest

from src.engine.handler_registry import get_handler_metadata
from src.onboarding.graph import Graph, build_graph
from src.onboarding.handlers import (
    handle_await_documents_signed,
    handle_await_hardware_delivered,
    handle_collect,
    handle_complete,
    handle_escalated,
    handle_it_provisioned,
    handle_schedule_sent,
    handle_welcome_sent,
    handler_map,
)
from src.onboarding.session_state import new_onboarding_session_state
from src.onboarding.state_transitions import State


@pytest.fixture(autouse=True)
def _ensure_onboarding_handlers_registered():
    """See tests/test_invoke_helpers.py for why this guard is needed: the
    process-global handler registry gets cleared by other test files'
    clear_metadata() calls (e.g. test_aemit_event.py, test_arun_to_completion.py)
    and never re-registers on its own, since @handler decorators only run
    once at import time.

    Checks AWAIT_DOCUMENTS_SIGNED specifically, not COLLECT: found by
    execution that tests/test_handler_registry.py's own
    test_handler_decorator_registers_wait_kind_human example happens to
    register a state literally named "collect" (coincidentally matching
    this domain's real state name) as its last action, so checking
    COLLECT's presence alone produced a false negative — "collect" looked
    registered while every other onboarding state stayed wiped.
    """
    if get_handler_metadata(State.AWAIT_DOCUMENTS_SIGNED.value) is None:
        import src.onboarding.handlers as onboarding_handlers

        importlib.reload(onboarding_handlers)


# ─────────────────────────────────────────────────────────────────────────────
# AWAIT_* handlers must branch on current_event_type (design spec §3)
# ─────────────────────────────────────────────────────────────────────────────

def test_await_documents_signed_records_timeout_not_signature():
    state = {**new_onboarding_session_state(), "current_event_type": "timeout_escalation"}
    result = handle_await_documents_signed(state)
    assert "timeout" in result["audit_trail"][0]
    assert "document_signed" not in result["audit_trail"][0]


def test_await_documents_signed_records_signature_on_legal_event():
    state = {**new_onboarding_session_state(), "current_event_type": "document_signed"}
    result = handle_await_documents_signed(state)
    assert "document_signed event received" in result["audit_trail"][0]


def test_await_hardware_delivered_records_timeout_not_delivery():
    state = {**new_onboarding_session_state(), "current_event_type": "timeout_escalation"}
    result = handle_await_hardware_delivered(state)
    assert "timeout" in result["audit_trail"][0]


def test_await_hardware_delivered_records_delivery_on_legal_event():
    state = {**new_onboarding_session_state(), "current_event_type": "hardware_delivered"}
    result = handle_await_hardware_delivered(state)
    assert "hardware_delivered event received" in result["audit_trail"][0]


# ─────────────────────────────────────────────────────────────────────────────
# Idempotent guard flags (design spec §2.3)
# ─────────────────────────────────────────────────────────────────────────────

def test_welcome_sent_sets_guard_flag_once():
    state = {**new_onboarding_session_state(), "new_hire_details": {"full_name": "Jane Doe"}}
    result = handle_welcome_sent(state)
    assert result["welcome_sent"] is True
    assert result["output_messages"]

    already_sent_state = {**state, "welcome_sent": True}
    second_result = handle_welcome_sent(already_sent_state)
    assert "already sent" in second_result["audit_trail"][0]
    assert "welcome_sent" not in second_result or second_result.get("output_messages") in (None, [])


def test_schedule_sent_sets_guard_flag_once():
    state = new_onboarding_session_state()
    result = handle_schedule_sent(state)
    assert result["schedule_sent"] is True

    already_sent_state = {**state, "schedule_sent": True}
    second_result = handle_schedule_sent(already_sent_state)
    assert "already sent" in second_result["audit_trail"][0]


def test_complete_sets_hr_notified_once():
    state = new_onboarding_session_state()
    result = handle_complete(state)
    assert result["hr_notified"] is True

    already_notified_state = {**state, "hr_notified": True}
    second_result = handle_complete(already_notified_state)
    assert "already" in second_result["audit_trail"][0]


def test_it_provisioned_sets_guard_flag_once():
    state = {**new_onboarding_session_state(), "new_hire_details": {"full_name": "Jane Doe"}}
    with patch("src.onboarding.chains.username_chain") as mock_chain:
        mock_chain.invoke.return_value = {"username_prefix": "jdoe", "reasoning": "first+last"}
        result = handle_it_provisioned(state)
    assert result["it_provisioned"] is True
    assert result["username_prefix"] == "jdoe"

    already_provisioned_state = {**state, "it_provisioned": True}
    second_result = handle_it_provisioned(already_provisioned_state)
    assert "already provisioned" in second_result["audit_trail"][0]


def test_escalated_produces_output_message():
    result = handle_escalated(new_onboarding_session_state())
    assert result["output_messages"]


# ─────────────────────────────────────────────────────────────────────────────
# handle_collect — tool-calling agent, mocked
# ─────────────────────────────────────────────────────────────────────────────

def test_handle_collect_extracts_details_from_tool_call():
    from langchain_core.messages import AIMessage, ToolMessage

    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {
        "messages": [
            AIMessage(content="", tool_calls=[{"name": "submit_new_hire", "args": {}, "id": "1"}]),
            ToolMessage(
                content='{"full_name": "Jane Doe", "role": "Engineer", "start_date": "2026-08-01"}',
                name="submit_new_hire",
                tool_call_id="1",
            ),
        ]
    }
    state = {
        **new_onboarding_session_state(),
        "input_message": "Jane Doe, Engineer, starts Aug 1 2026",
    }

    with patch("src.onboarding.chains.collect_agent", mock_agent):
        result = handle_collect(state)

    assert result["new_hire_details"]["full_name"] == "Jane Doe"
    assert result["new_hire_details"]["role"] == "Engineer"


def test_handle_collect_asks_for_more_when_incomplete():
    from langchain_core.messages import AIMessage

    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {
        "messages": [AIMessage(content="What's your start date?")],
    }
    state = {**new_onboarding_session_state(), "input_message": "Jane Doe, Engineer"}

    with patch("src.onboarding.chains.collect_agent", mock_agent):
        result = handle_collect(state)

    assert "new_hire_details" not in result
    assert result["output_messages"] == ["What's your start date?"]


# ─────────────────────────────────────────────────────────────────────────────
# graph.py wiring
# ─────────────────────────────────────────────────────────────────────────────

def test_handler_map_covers_every_non_terminal_and_terminal_state_with_a_handler():
    from src.onboarding.state_transitions import State as S

    expected_dispatchable = {
        S.COLLECT,
        S.WELCOME_SENT,
        S.AWAIT_DOCUMENTS_SIGNED,
        S.IT_PROVISIONED,
        S.AWAIT_HARDWARE_DELIVERED,
        S.SCHEDULE_SENT,
        S.COMPLETE,
        S.ESCALATED,
        S.ERROR,
    }
    assert set(handler_map.keys()) == expected_dispatchable


def test_build_graph_produces_compiled_graph():
    graph = build_graph(sessions_dir="/tmp/test_onboarding_graph_smoke")
    assert graph.compiled_graph is not None
    assert graph.state_enum is State
    assert isinstance(graph, Graph)


def test_invoke_drives_collect_to_await_documents_signed_end_to_end():
    """Plan success criterion: build_graph() must produce a working
    compiled graph drivable with plain invoke() from COLLECT to the first
    park state, using only synthetic state deltas — no aemit_event needed
    yet (that's Phase 10's integration test).

    Two turns, matching docprocessing's own UPLOAD_DOCUMENTS convention
    exactly: turn 1 parks at COLLECT without ever dispatching its handler
    (COLLECT waits_for_input=True, so the graph run ends at the park
    point before running it — confirmed by tracing _guardrail_router);
    turn 2's resume-dispatch step is what actually runs handle_collect
    with the real input.
    """
    from uuid import uuid4

    from langchain_core.messages import AIMessage, ToolMessage

    graph = build_graph(sessions_dir=f"/tmp/test_onboarding_e2e_{uuid4()}")
    session_id = str(uuid4())

    turn1 = graph.invoke(user_id="user-1", session_id=session_id, input_message="start onboarding")
    assert turn1["current_state"] == State.COLLECT.value

    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {
        "messages": [
            AIMessage(content="", tool_calls=[{"name": "submit_new_hire", "args": {}, "id": "1"}]),
            ToolMessage(
                content=(
                    '{"full_name": "Jane Doe", "role": "Engineer", '
                    '"start_date": "2026-08-01"}'
                ),
                name="submit_new_hire",
                tool_call_id="1",
            ),
        ]
    }
    mock_username_chain = MagicMock()
    mock_username_chain.invoke.return_value = {"username_prefix": "jdoe", "reasoning": "n/a"}

    with patch("src.onboarding.chains.collect_agent", mock_agent), patch(
        "src.onboarding.chains.username_chain", mock_username_chain
    ):
        turn2 = graph.invoke(
            user_id="user-1",
            session_id=session_id,
            input_message="Jane Doe, Engineer, starting 2026-08-01",
        )

    # COLLECT (resume-dispatch, succeeds) -> WELCOME_SENT (auto-progress) ->
    # AWAIT_DOCUMENTS_SIGNED (waits_for_input=True, parks here)
    assert turn2["status"] != "error"
    assert turn2["current_state"] == State.AWAIT_DOCUMENTS_SIGNED.value
    assert turn2["new_hire_details"]["full_name"] == "Jane Doe"
    assert turn2["welcome_sent"] is True
