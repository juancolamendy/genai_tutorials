"""End-to-end integration test driving the full onboarding state chart
(design spec §3) entirely through aemit_event/ainvoke — the final proof
that every phase's pieces compose correctly.

collect_agent and username_chain are mocked (no real LLM calls); every
other mechanism (aemit_event's gate, guardrails, handlers, the event
ledger, auto-progression) is exercised for real.
"""

import importlib
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.engine.handler_registry import get_handler_metadata
from src.onboarding.graph import build_graph as _build_onboarding_graph
from src.onboarding.state_transitions import State


@pytest.fixture(autouse=True)
def _ensure_onboarding_handlers_registered():
    """See tests/test_onboarding_handlers.py for why AWAIT_DOCUMENTS_SIGNED
    specifically (not COLLECT — collides with an unrelated test's example
    state name)."""
    if get_handler_metadata(State.AWAIT_DOCUMENTS_SIGNED.value) is None:
        import src.onboarding.handlers as onboarding_handlers

        importlib.reload(onboarding_handlers)


def _mock_collect_agent(complete=True):
    mock_agent = MagicMock()
    if complete:
        tool_call = {"name": "submit_new_hire", "args": {}, "id": "1"}
        value = {
            "messages": [
                AIMessage(content="", tool_calls=[tool_call]),
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
    else:
        value = {
            "messages": [AIMessage(content="What role?")],
        }
    mock_agent.ainvoke = AsyncMock(return_value=value)
    mock_agent.invoke.return_value = value
    return mock_agent


def _mock_username_chain():
    mock_chain = MagicMock()
    value = {"username_prefix": "jdoe", "reasoning": "n/a"}
    mock_chain.ainvoke = AsyncMock(return_value=value)
    mock_chain.invoke.return_value = value
    return mock_chain


def build_graph(sessions_dir):
    """Unique per-test sessions_dir (and thus colocated ledger dir from
    onboarding.build_graph) — avoids event_id collisions across runs."""
    return _build_onboarding_graph(sessions_dir=sessions_dir)


async def _kickoff_and_collect(graph, thread_id):
    """Turn 1 (ainvoke, parks at COLLECT) + human aemit_event turn that
    supplies complete details, advancing through WELCOME_SENT to the
    first system-event park state."""
    turn1 = await graph.ainvoke(user_id="", session_id=thread_id, input_message="start")
    assert turn1["current_state"] == State.COLLECT.value

    with patch("src.onboarding.chains.collect_agent", _mock_collect_agent()):
        result = await graph.aemit_event(
            thread_id=thread_id,
            event_source="human",
            event_type="message",
            input_message="Jane Doe, Engineer, starting 2026-08-01",
        )
    assert result["emit_status"] == "ok"
    assert result["current_state"] == State.AWAIT_DOCUMENTS_SIGNED.value
    return result


@pytest.mark.asyncio
async def test_full_happy_path_through_aemit_event():
    graph = build_graph(sessions_dir=f"/tmp/test_onboarding_flow_{uuid4()}")
    thread_id = str(uuid4())
    await _kickoff_and_collect(graph, thread_id)

    # username_chain runs inside IT_PROVISIONED's auto-progress, which
    # happens synchronously within this same aemit_event call — the patch
    # must be active around the call, not after it.
    with patch("src.onboarding.chains.username_chain", _mock_username_chain()):
        signed = await graph.aemit_event(
            thread_id=thread_id,
            event_source="system",
            event_type="document_signed",
            event_id="evt-signed",
        )
    assert signed["emit_status"] == "ok"
    assert signed["current_state"] == State.AWAIT_HARDWARE_DELIVERED.value
    assert signed["it_provisioned"] is True
    assert signed["username_prefix"] == "jdoe"

    delivered = await graph.aemit_event(
        thread_id=thread_id,
        event_source="system",
        event_type="hardware_delivered",
        event_id="evt-delivered",
    )
    assert delivered["emit_status"] == "ok"
    assert delivered["current_state"] == State.COMPLETE.value
    assert delivered["schedule_sent"] is True
    assert delivered["hr_notified"] is True


@pytest.mark.asyncio
async def test_timeout_escalation_from_await_documents_signed():
    graph = build_graph(sessions_dir=f"/tmp/test_onboarding_flow_{uuid4()}")
    thread_id = str(uuid4())
    await _kickoff_and_collect(graph, thread_id)

    result = await graph.aemit_event(
        thread_id=thread_id,
        event_source="system",
        event_type="timeout_escalation",
        event_id="evt-timeout-1",
    )

    assert result["emit_status"] == "ok"
    assert result["current_state"] == State.ESCALATED.value


@pytest.mark.asyncio
async def test_timeout_escalation_from_await_hardware_delivered():
    graph = build_graph(sessions_dir=f"/tmp/test_onboarding_flow_{uuid4()}")
    thread_id = str(uuid4())
    await _kickoff_and_collect(graph, thread_id)

    with patch("src.onboarding.chains.username_chain", _mock_username_chain()):
        signed = await graph.aemit_event(
            thread_id=thread_id, event_source="system", event_type="document_signed", event_id="evt-s"
        )
    assert signed["current_state"] == State.AWAIT_HARDWARE_DELIVERED.value

    result = await graph.aemit_event(
        thread_id=thread_id,
        event_source="system",
        event_type="timeout_escalation",
        event_id="evt-timeout-2",
    )

    assert result["emit_status"] == "ok"
    assert result["current_state"] == State.ESCALATED.value


@pytest.mark.asyncio
async def test_duplicate_event_id_does_not_double_process():
    graph = build_graph(sessions_dir=f"/tmp/test_onboarding_flow_{uuid4()}")
    thread_id = str(uuid4())
    await _kickoff_and_collect(graph, thread_id)

    with patch("src.onboarding.chains.username_chain", _mock_username_chain()):
        first = await graph.aemit_event(
            thread_id=thread_id,
            event_source="system",
            event_type="document_signed",
            event_id="evt-dup",
        )
    assert first["emit_status"] == "ok"
    audit_len_after_first = len(first.get("audit_trail", []))

    second = await graph.aemit_event(
        thread_id=thread_id,
        event_source="system",
        event_type="document_signed",
        event_id="evt-dup",
    )

    assert second["emit_status"] == "duplicate"
    # State genuinely untouched: get_active_sessions() reads the true
    # persisted checkpoint (unlike _get_or_init_state, which always resets
    # audit_trail/messages to [] since it's meant to feed back into the
    # next compiled_graph.invoke() call, not for inspection) — its
    # audit_trail must match the length right after the first call, not
    # have grown from a second dispatch.
    sessions = {s["thread_id"]: s["state"] for s in graph.get_active_sessions()}
    reloaded = sessions[thread_id]
    assert len(reloaded["audit_trail"]) == audit_len_after_first
    assert reloaded["current_state"] == State.AWAIT_HARDWARE_DELIVERED.value
