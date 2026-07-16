"""Tests for onboarding sweep.py and cli.py (design spec §13, §1.2/§2.2).
"""

import importlib
import time
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.engine.handler_registry import get_handler_metadata
from src.onboarding import cli
from src.onboarding.graph import build_graph as _build_onboarding_graph
from src.onboarding.state_transitions import State
from src.onboarding.sweep import sweep


@pytest.fixture(autouse=True)
def _ensure_onboarding_handlers_registered():
    """See tests/test_onboarding_handlers.py for why AWAIT_DOCUMENTS_SIGNED
    specifically."""
    if get_handler_metadata(State.AWAIT_DOCUMENTS_SIGNED.value) is None:
        import src.onboarding.handlers as onboarding_handlers

        importlib.reload(onboarding_handlers)


def _build_graph(sessions_dir):
    """Unique per-test sessions_dir (and thus colocated ledger from
    onboarding.build_graph)."""
    return _build_onboarding_graph(sessions_dir=sessions_dir)


# ─────────────────────────────────────────────────────────────────────────────
# sweep.py
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sweep_escalates_a_stale_thread():
    graph = _build_graph(f"/tmp/test_sweep_{uuid4()}")
    thread_id = str(uuid4())

    # started_at is set once, at session creation (new_engine_session_state()),
    # and never touched again — patch time.time() there so it's baked into
    # the real checkpoint naturally, the same way it would look after 8
    # real days elapsed. (Backdating via a direct update_state() call
    # afterwards doesn't work: it creates a new checkpoint the thread's
    # still-current pause-point pointer doesn't follow — the same class of
    # staleness _get_or_init_state's pause-point fix addresses, just not
    # a scenario a real sweep ever hits, since nothing else ever rewrites
    # started_at out from under the original pause-point checkpoint.)
    eight_days_ago = time.time() - (8 * 24 * 3600)
    with patch("src.engine.engine_session_state.time.time", return_value=eight_days_ago):
        await graph.ainvoke(user_id="", session_id=thread_id, input_message="start")

    from src.onboarding.chains import NewHireDetails

    mock_collect = MagicMock()
    mock_collect.ainvoke = AsyncMock(
        return_value=NewHireDetails(
            full_name="Jane Doe",
            role="Engineer",
            start_date="2026-08-01",
            complete=True,
            reply="Thanks.",
        )
    )
    with patch("src.onboarding.chains.collect_chain", mock_collect):
        result = await graph.aemit_event(
            thread_id=thread_id, event_source="human", event_type="message", input_message="go"
        )
    assert result["current_state"] == State.AWAIT_DOCUMENTS_SIGNED.value

    results = await sweep(graph, {"await_documents_signed": 7 * 24 * 3600.0})

    assert len(results) == 1
    assert results[0]["emit_status"] == "ok"
    assert results[0]["current_state"] == State.ESCALATED.value


@pytest.mark.asyncio
async def test_sweep_does_not_escalate_a_fresh_thread():
    graph = _build_graph(f"/tmp/test_sweep_{uuid4()}")
    thread_id = str(uuid4())
    await graph.ainvoke(user_id="", session_id=thread_id, input_message="start")

    results = await sweep(graph, {"collect": 7 * 24 * 3600.0})

    assert results == []


@pytest.mark.asyncio
async def test_sweep_ignores_states_not_in_thresholds():
    graph = _build_graph(f"/tmp/test_sweep_{uuid4()}")
    thread_id = str(uuid4())
    await graph.ainvoke(user_id="", session_id=thread_id, input_message="start")

    # current_state is "collect" but thresholds only cover a different state
    results = await sweep(graph, {"await_hardware_delivered": 1.0})

    assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# cli.py — command functions, with a mocked graph
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_cmd_chat_calls_aemit_event_with_human_source(capsys):
    mock_graph = MagicMock()
    mock_graph.aemit_event = AsyncMock(
        return_value={
            "emit_status": "ok",
            "current_state": "collect",
            "output_messages": ["I still need a start date."],
        }
    )

    output = await cli._cmd_chat(mock_graph, "thread-1", "hello")

    mock_graph.aemit_event.assert_awaited_once_with(
        thread_id="thread-1", event_source="human", event_type="message", input_message="hello"
    )
    assert "emit_status=ok" in output
    assert "current_state=collect" in output
    assert "I still need a start date." in capsys.readouterr().out


@pytest.mark.asyncio
async def test_cmd_event_calls_aemit_event_with_system_source_and_payload():
    mock_graph = MagicMock()
    mock_graph.aemit_event = AsyncMock(
        return_value={"emit_status": "ok", "current_state": "it_provisioned"}
    )

    await cli._cmd_event(
        mock_graph, "thread-1", "document_signed", "evt-1", {"tracking_id": "abc"}
    )

    mock_graph.aemit_event.assert_awaited_once_with(
        thread_id="thread-1",
        event_source="system",
        event_type="document_signed",
        event_id="evt-1",
        payload={"tracking_id": "abc"},
    )


@pytest.mark.asyncio
async def test_cmd_sweep_calls_sweep_module_function():
    mock_graph = MagicMock()
    with patch("src.onboarding.cli.run_sweep", new=AsyncMock(return_value=[{"emit_status": "ok"}])):
        output = await cli._cmd_sweep(mock_graph)

    assert "Swept 1 stale thread(s)" in output


def test_cmd_status_reads_via_get_or_init_state():
    mock_graph = MagicMock()
    mock_graph._get_or_init_state.return_value = {
        "current_state": "collect",
        "session_status": "ok",
    }

    output = cli._cmd_status(mock_graph, "thread-1")

    mock_graph._get_or_init_state.assert_called_once_with(session_id="thread-1", user_id="")
    assert "current_state=collect" in output


def test_parse_payload_handles_key_value_pairs():
    assert cli._parse_payload(["a=1", "b=2"]) == {"a": "1", "b": "2"}
    assert cli._parse_payload(None) == {}
    assert cli._parse_payload([]) == {}


def test_build_parser_requires_a_subcommand():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_build_parser_chat_subcommand():
    parser = cli.build_parser()
    args = parser.parse_args(["chat", "thread-1", "hello there"])
    assert args.command == "chat"
    assert args.thread_id == "thread-1"
    assert args.message == "hello there"


def test_build_parser_event_subcommand_requires_event_id():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["event", "thread-1", "document_signed"])

    args = parser.parse_args(["event", "thread-1", "document_signed", "--event-id", "evt-1"])
    assert args.event_id == "evt-1"
