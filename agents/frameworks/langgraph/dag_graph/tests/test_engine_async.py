"""Tests for EngineGraph.ainvoke() and the per-thread asyncio.Lock registry
(design spec §5/§14, Phase 4), plus the router bypass, message convention,
and get_active_sessions() helper added in Phase 6 (design spec §9/§11/§13).

Phase 6's plan named tests/test_router_integration.py as a file to extend,
but that file currently fails with pre-existing, unrelated failures
(build_router_prompt()'s turn_input/input_message kwarg mismatch) —
confirmed identical on main before this design's work started. Extending
it further would only add more failures to an already-broken file for
reasons unrelated to the router bypass. Added here instead, alongside the
already-clean async engine tests.
"""

import asyncio
import importlib
from enum import Enum
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from src.docprocessing.graph import build_graph
from src.engine.engine_graph import EngineGraph
from src.engine.engine_session_state import new_engine_session_state
from src.engine.handler_registry import get_handler_metadata, handler
from src.engine.json_checkpointer import JsonCheckpointer
from src.engine.router import RouterDecision


@pytest.fixture(autouse=True)
def _ensure_docprocessing_handlers_registered():
    """See tests/test_invoke_helpers.py for why this guard is needed: a
    process-global handler registry gets cleared by an earlier test file
    and never re-registers on its own."""
    if get_handler_metadata("upload_documents") is None:
        import src.docprocessing.handlers as dp_handlers

        importlib.reload(dp_handlers)


def _build_graph():
    return build_graph(sessions_dir=f"/tmp/test_engine_async_{uuid4()}")


@pytest.mark.asyncio
async def test_ainvoke_resumes_multi_turn_session_with_empty_user_id():
    """Mirrors test_invoke_helpers.py's resume regression test, but for the
    new async twin — the exact scenario aemit_event (a later phase) drives."""
    graph = _build_graph()
    session_id = str(uuid4())

    with patch("src.docprocessing.handlers.random.random", return_value=0.9):
        turn1 = await graph.ainvoke(
            user_id="",
            session_id=session_id,
            input_message="please process this document",
            state_delta={"document_id": "doc1"},
        )
    assert turn1["current_state"] == "upload_documents"
    assert turn1["turn_number"] == 1

    with patch("src.docprocessing.handlers.random.random", return_value=0.9):
        turn2 = await graph.ainvoke(
            user_id="",
            session_id=session_id,
            input_message="here are the supporting documents",
            state_delta={"supporting_docs": [{"name": "a.pdf"}]},
        )

    assert turn2["turn_number"] == 2
    assert turn2["current_state"] == "complete"


@pytest.mark.asyncio
async def test_ainvoke_with_empty_input_message_does_not_raise():
    """ainvoke() must share invoke()'s _prepare_input() fix — every
    system-sourced/bg call supplies input_message=""."""
    graph = _build_graph()
    result = await graph.ainvoke(
        user_id="some-user",
        session_id=str(uuid4()),
        input_message="",
        state_delta={"document_id": "doc1"},
    )
    assert result["status"] != "error"
    assert result.get("error_message") is None


@pytest.mark.asyncio
async def test_ainvoke_and_invoke_produce_consistent_thread_ids():
    """A session started with invoke() (sync) must be resumable via
    ainvoke() (async) and vice versa — they must agree on thread_id."""
    graph = _build_graph()
    session_id = str(uuid4())

    with patch("src.docprocessing.handlers.random.random", return_value=0.9):
        turn1 = graph.invoke(
            user_id="",
            session_id=session_id,
            input_message="start",
            state_delta={"document_id": "doc1"},
        )
    assert turn1["current_state"] == "upload_documents"

    with patch("src.docprocessing.handlers.random.random", return_value=0.9):
        turn2 = await graph.ainvoke(
            user_id="",
            session_id=session_id,
            input_message="continue",
            state_delta={"supporting_docs": [{"name": "a.pdf"}]},
        )
    assert turn2["current_state"] == "complete"
    assert turn2["turn_number"] == 2


def test_locks_registry_is_defaultdict_of_asyncio_lock():
    graph = _build_graph()
    lock = graph._locks["thread-1"]
    assert isinstance(lock, asyncio.Lock)
    # Same key returns the same lock instance
    assert graph._locks["thread-1"] is lock
    # Different keys get independent locks
    assert graph._locks["thread-2"] is not lock


@pytest.mark.asyncio
async def test_concurrent_calls_for_same_thread_id_serialize():
    """Two coroutines racing to acquire the same thread's lock must run
    one-at-a-time, not interleaved — the mechanism aemit_event (a later
    phase) relies on to prevent double-processing a concurrently-delivered
    duplicate event for the same thread."""
    graph = _build_graph()
    order = []

    async def _hold_lock(name, hold_seconds):
        async with graph._locks["shared-thread"]:
            order.append(f"{name}-start")
            await asyncio.sleep(hold_seconds)
            order.append(f"{name}-end")

    await asyncio.gather(_hold_lock("first", 0.05), _hold_lock("second", 0.0))

    # If they interleaved, "second-start" would appear before "first-end".
    assert order == ["first-start", "first-end", "second-start", "second-end"]


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6: router bypass for system-sourced turns (design spec §9)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_semantic_router_bypassed_for_system_sourced_turn():
    """A router-bypass hook: only attempt semantic routing if
    current_event_source != "system" — an LLM can never decide a
    system-event transition, structurally, regardless of whether a
    semantic_router is attached."""
    mock_router = MagicMock()
    mock_router.route.return_value = RouterDecision(proposed_next="validate", confidence=0.9)

    graph = _build_graph()
    graph.semantic_router = mock_router
    thread_id = str(uuid4())

    with patch("src.docprocessing.handlers.random.random", return_value=0.9):
        await graph.ainvoke(
            user_id="",
            session_id=thread_id,
            input_message="",
            state_delta={"document_id": "doc1", "current_event_source": "system"},
        )

    mock_router.route.assert_not_called()


@pytest.mark.asyncio
async def test_semantic_router_still_used_for_human_sourced_turn():
    """Sanity check that the mock is wired correctly — proves the previous
    test's assert_not_called() is meaningful, not vacuous."""
    mock_router = MagicMock()
    mock_router.route.return_value = RouterDecision(proposed_next="fetch", confidence=0.9)

    graph = _build_graph()
    graph.semantic_router = mock_router
    thread_id = str(uuid4())

    with patch("src.docprocessing.handlers.random.random", return_value=0.9):
        await graph.ainvoke(
            user_id="",
            session_id=thread_id,
            input_message="please process this",
            state_delta={"document_id": "doc1"},
        )

    # Not asserting an exact call count: the mock always proposes "fetch"
    # regardless of current_state, so auto-progress re-invokes the router
    # (and gets a guardrail-rejected fetch->fetch proposal) more than once.
    # What matters here is only that the semantic path is reachable at all
    # for a human-sourced turn — contrasting with the bypass test above.
    mock_router.route.assert_called()


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6: output_messages message convention (design spec §11)
# ─────────────────────────────────────────────────────────────────────────────

class _MsgTestState(str, Enum):
    INIT = "msgtest_init"
    DONE = "msgtest_done"
    ERROR = "msgtest_error"


def _register_msgtest_handler():
    @handler(state=_MsgTestState.DONE.value, waits_for_input=False)
    def _handle_msgtest_done(state):
        if state.get("current_event_type") == "custom_greeting":
            return {"output_messages": ["custom greeting message"]}
        return {}

    @handler(state=_MsgTestState.ERROR.value, waits_for_input=False)
    def _handle_msgtest_error(state):
        return {}

    return {_MsgTestState.DONE: _handle_msgtest_done, _MsgTestState.ERROR: _handle_msgtest_error}


class _MsgTestGraph(EngineGraph):
    state_enum = _MsgTestState
    terminal_states = {_MsgTestState.DONE, _MsgTestState.ERROR}

    def __init__(self, handler_map):
        super().__init__()
        self.handler_map = handler_map

    def _build_routing_table(self):
        return {_MsgTestState.INIT: _MsgTestState.DONE}

    def _get_current_state(self, state):
        return _MsgTestState(state.get("current_state", _MsgTestState.INIT.value))

    def _get_proposed_state(self, state):
        return _MsgTestState(state.get("proposed_next", _MsgTestState.DONE.value))

    def _get_guardrails(self):
        return {}

    def _new_session_state(self):
        return {
            **new_engine_session_state(),
            "current_state": _MsgTestState.INIT.value,
            "proposed_next": _MsgTestState.DONE.value,
        }


def _build_msgtest_graph():
    graph = _MsgTestGraph(_register_msgtest_handler())
    checkpointer = JsonCheckpointer(sessions_dir=f"/tmp/test_msgtest_{uuid4()}")
    from src.engine.engine_session_state import EngineSessionState

    graph.compiled_graph = graph.build_graph(EngineSessionState, checkpointer=checkpointer)
    return graph


@pytest.mark.asyncio
async def test_human_turn_with_no_output_messages_gets_fallback_message():
    """Backward-compatible default: no handler sets output_messages (matches
    every existing docprocessing handler), so a human turn still gets the
    generic "Transitioned to X" message it always has."""
    graph = _build_msgtest_graph()
    result = await graph.ainvoke(
        user_id="", session_id=str(uuid4()), input_message="hello", state_delta={}
    )

    messages = result["messages"]
    assert len(messages) == 2
    assert messages[0].content == "hello"
    assert messages[1].content == "Transitioned to msgtest_done"


@pytest.mark.asyncio
async def test_system_turn_with_output_messages_skips_fallback():
    graph = _build_msgtest_graph()
    result = await graph.ainvoke(
        user_id="",
        session_id=str(uuid4()),
        input_message="",
        state_delta={"current_event_source": "system", "current_event_type": "custom_greeting"},
    )

    messages = result["messages"]
    assert len(messages) == 1
    assert messages[0].content == "custom greeting message"


@pytest.mark.asyncio
async def test_system_turn_with_nothing_to_say_leaves_messages_untouched():
    """No meaningless empty-turn noise: a system-sourced turn with no
    output_messages and no human text produces zero new messages."""
    graph = _build_msgtest_graph()
    result = await graph.ainvoke(
        user_id="",
        session_id=str(uuid4()),
        input_message="",
        state_delta={"current_event_source": "system", "current_event_type": "other_event"},
    )

    assert result["messages"] == []


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6: get_active_sessions() (design spec §13)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_active_sessions_returns_unwrapped_state():
    graph = _build_graph()
    thread_id = str(uuid4())

    with patch("src.docprocessing.handlers.random.random", return_value=0.9):
        await graph.ainvoke(
            user_id="",
            session_id=thread_id,
            input_message="start",
            state_delta={"document_id": "doc1"},
        )

    sessions = graph.get_active_sessions()

    assert len(sessions) == 1
    assert sessions[0]["thread_id"] == thread_id
    assert "state" in sessions[0]
    assert sessions[0]["state"]["current_state"] == "upload_documents"
    assert sessions[0]["state"]["document_id"] == "doc1"


@pytest.mark.asyncio
async def test_get_active_sessions_enumerates_multiple_threads():
    graph = _build_graph()

    with patch("src.docprocessing.handlers.random.random", return_value=0.9):
        await graph.ainvoke(
            user_id="",
            session_id=str(uuid4()),
            input_message="a",
            state_delta={"document_id": "d1"},
        )
        await graph.ainvoke(
            user_id="",
            session_id=str(uuid4()),
            input_message="b",
            state_delta={"document_id": "d2"},
        )

    sessions = graph.get_active_sessions()
    assert len(sessions) == 2


@pytest.mark.asyncio
async def test_get_active_sessions_prefers_latest_when_pause_is_stale():
    """Mirrors _get_or_init_state's stale-pause check: after a thread
    resumes past a pause point to a terminal state, enumeration must
    report the latest state — not the abandoned park."""
    from src.engine.event_ledger import EventLedger
    from src.engine.engine_session_state import EngineSessionState

    class _StalePauseState(str, Enum):
        INIT = "stale_pause_init"
        AWAIT = "stale_pause_await"
        DONE = "stale_pause_done"
        ERROR = "stale_pause_error"

    @handler(
        state=_StalePauseState.AWAIT.value,
        waits_for_input=True,
        wait_kind="system_event",
        expected_events=["go"],
    )
    def _handle_await(state):
        return {"audit_trail": ["await handled"]}

    @handler(state=_StalePauseState.DONE.value, waits_for_input=False)
    def _handle_done(state):
        return {"audit_trail": ["done"]}

    @handler(state=_StalePauseState.ERROR.value, waits_for_input=False)
    def _handle_error(state):
        return {}

    class _StalePauseGraph(EngineGraph):
        state_enum = _StalePauseState
        terminal_states = {_StalePauseState.DONE, _StalePauseState.ERROR}

        def __init__(self):
            super().__init__()
            self.handler_map = {
                _StalePauseState.AWAIT: _handle_await,
                _StalePauseState.DONE: _handle_done,
                _StalePauseState.ERROR: _handle_error,
            }

        def _build_routing_table(self):
            return {
                _StalePauseState.INIT: _StalePauseState.AWAIT,
                _StalePauseState.AWAIT: _StalePauseState.DONE,
            }

        def _get_current_state(self, state):
            return _StalePauseState(state.get("current_state", _StalePauseState.INIT.value))

        def _get_proposed_state(self, state):
            return _StalePauseState(state.get("proposed_next", _StalePauseState.AWAIT.value))

        def _get_guardrails(self):
            return {}

        def _new_session_state(self):
            return {
                **new_engine_session_state(),
                "current_state": _StalePauseState.INIT.value,
                "proposed_next": _StalePauseState.AWAIT.value,
            }

    graph = _StalePauseGraph()
    sessions_dir = f"/tmp/test_stale_pause_{uuid4()}"
    checkpointer = JsonCheckpointer(sessions_dir=sessions_dir)
    graph.compiled_graph = graph.build_graph(EngineSessionState, checkpointer=checkpointer)
    graph._ledger = EventLedger(ledger_dir=f"{sessions_dir}_ledger")

    thread_id = str(uuid4())
    parked = await graph.ainvoke(user_id="", session_id=thread_id, input_message="")
    assert parked["current_state"] == _StalePauseState.AWAIT.value

    done = await graph.aemit_event(
        thread_id=thread_id, source="system", event_type="go", event_id="evt-go"
    )
    assert done["status"] == "ok"
    assert done["current_state"] == _StalePauseState.DONE.value

    sessions = {s["thread_id"]: s["state"] for s in graph.get_active_sessions()}
    assert sessions[thread_id]["current_state"] == _StalePauseState.DONE.value


@pytest.mark.asyncio
async def test_get_or_init_state_clears_output_messages():
    """output_messages is reducer-backed like audit_trail/messages — loading
    a prior turn's accumulated list without resetting would feed it back into
    the reducer and into _build_new_messages on the next turn."""
    graph = _build_msgtest_graph()
    session_id = str(uuid4())
    result = await graph.ainvoke(
        user_id="",
        session_id=session_id,
        input_message="",
        state_delta={"current_event_source": "system", "current_event_type": "custom_greeting"},
    )
    assert any(m.content == "custom greeting message" for m in result["messages"])
    # Checkpoint still holds the prior turn's reducer value:
    assert result.get("output_messages") == ["custom greeting message"]

    loaded = graph._get_or_init_state(session_id=session_id, user_id="")
    assert loaded.get("output_messages") == []


@pytest.mark.asyncio
async def test_ainvoke_forwards_max_auto_iters_to_auto_progress():
    graph = _build_msgtest_graph()
    session_id = str(uuid4())

    async def _fake_auto_progress(state, config, max_auto_iters=10):
        assert max_auto_iters == 3
        return state

    with patch.object(graph, "_auto_progress_langgraph_async", side_effect=_fake_auto_progress):
        result = await graph.ainvoke(
            user_id="",
            session_id=session_id,
            input_message="hi",
            state_delta={},
            max_auto_iters=3,
        )

    assert result["status"] == "ok"
