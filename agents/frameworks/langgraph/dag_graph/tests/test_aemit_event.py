"""Tests for EngineGraph.aemit_event() — the unified human/system event gate
(design spec §6, including the round-2 symmetric wait_kind=="human" guard).

A minimal, test-local 4-state graph (states prefixed "aemit_test_" so they
can never collide with the global handler_metadata_map registrations any
other test file makes) exercises the system-event-specific behavior
(wait_kind="system_event", expected_events) that no domain graph provides
yet (onboarding doesn't exist until a later phase). The docprocessing graph
is used for the human-branch smoke test, since it already has a real
wait_kind="either" (default) park state.
"""

import importlib
import uuid
from enum import Enum
from unittest.mock import AsyncMock, patch

import pytest

from src.engine.engine_graph import EngineGraph
from src.engine.engine_session_state import EngineSessionState, new_engine_session_state
from src.engine.handler_registry import clear_metadata, get_handler_metadata, handler
from src.engine.json_checkpointer import JsonCheckpointer


@pytest.fixture(autouse=True)
def _ensure_docprocessing_handlers_registered():
    """See tests/test_invoke_helpers.py for why this guard is needed."""
    if get_handler_metadata("upload_documents") is None:
        import src.docprocessing.handlers as dp_handlers

        importlib.reload(dp_handlers)


class _TestState(str, Enum):
    INIT = "aemit_test_init"
    AWAIT_SYS = "aemit_test_await_sys"
    DONE = "aemit_test_done"
    ERROR = "aemit_test_error"
    HUMAN_ONLY = "aemit_test_human_only"


def _register_test_handlers():
    clear_metadata()
    import src.docprocessing.handlers as dp_handlers

    importlib.reload(dp_handlers)  # restore docprocessing's registrations too

    @handler(
        state=_TestState.AWAIT_SYS.value,
        waits_for_input=True,
        wait_kind="system_event",
        expected_events=["thing_happened"],
        description="test-only system-event park state",
    )
    def _handle_await_sys(state):
        return {"audit_trail": ["await_sys handled"]}

    @handler(state=_TestState.DONE.value, waits_for_input=False)
    def _handle_done(state):
        return {"audit_trail": ["done"]}

    @handler(state=_TestState.ERROR.value, waits_for_input=False)
    def _handle_error(state):
        return {"audit_trail": ["error"]}

    return {
        _TestState.AWAIT_SYS: _handle_await_sys,
        _TestState.DONE: _handle_done,
        _TestState.ERROR: _handle_error,
    }


class _TestGraph(EngineGraph):
    state_enum = _TestState
    terminal_states = {_TestState.DONE, _TestState.ERROR}

    def __init__(self, handler_map):
        super().__init__()
        self.handler_map = handler_map

    def _build_routing_table(self):
        return {_TestState.INIT: _TestState.AWAIT_SYS, _TestState.AWAIT_SYS: _TestState.DONE}

    def _get_current_state(self, state):
        return _TestState(state.get("current_state", _TestState.INIT.value))

    def _get_proposed_state(self, state):
        return _TestState(state.get("proposed_next", _TestState.AWAIT_SYS.value))

    def _get_guardrails(self):
        return {}

    def _new_session_state(self):
        return {
            **new_engine_session_state(),
            "current_state": _TestState.INIT.value,
            "proposed_next": _TestState.AWAIT_SYS.value,
        }


def _build_test_graph():
    handler_map = _register_test_handlers()
    graph = _TestGraph(handler_map)
    checkpointer = JsonCheckpointer(sessions_dir=f"/tmp/test_aemit_event_{uuid.uuid4()}")
    graph.compiled_graph = graph.build_graph(EngineSessionState, checkpointer=checkpointer)
    from src.engine.event_ledger import EventLedger

    graph._ledger = EventLedger(ledger_dir=f"/tmp/test_aemit_event_ledger_{uuid.uuid4()}")
    return graph


async def _park_at_await_sys(graph, thread_id):
    """Drive a fresh session from INIT to AWAIT_SYS via ainvoke() directly
    (bypassing aemit_event, which is what's under test)."""
    result = await graph.ainvoke(user_id="", session_id=thread_id, input_message="", state_delta={})
    assert result["current_state"] == _TestState.AWAIT_SYS.value
    return result


@pytest.mark.asyncio
async def test_status_ok_for_system_sourced_legal_event():
    graph = _build_test_graph()
    thread_id = str(uuid.uuid4())
    await _park_at_await_sys(graph, thread_id)

    result = await graph.aemit_event(
        thread_id=thread_id, source="system", event_type="thing_happened", event_id="evt-1"
    )

    assert result["status"] == "ok"
    assert result["current_state"] == _TestState.DONE.value


@pytest.mark.asyncio
async def test_status_duplicate_for_repeated_event_id():
    graph = _build_test_graph()
    thread_id = str(uuid.uuid4())
    await _park_at_await_sys(graph, thread_id)

    first = await graph.aemit_event(
        thread_id=thread_id, source="system", event_type="thing_happened", event_id="evt-dup"
    )
    assert first["status"] == "ok"

    # Reset back to AWAIT_SYS to prove the second call is skipped because of
    # the ledger, not because the state simply isn't waiting anymore.
    second = await graph.aemit_event(
        thread_id=thread_id, source="system", event_type="thing_happened", event_id="evt-dup"
    )
    assert second["status"] == "duplicate"
    assert second["event_id"] == "evt-dup"


@pytest.mark.asyncio
async def test_status_ignored_for_event_type_not_in_expected_events():
    graph = _build_test_graph()
    thread_id = str(uuid.uuid4())
    await _park_at_await_sys(graph, thread_id)

    result = await graph.aemit_event(
        thread_id=thread_id, source="system", event_type="unrelated_event", event_id="evt-2"
    )

    assert result["status"] == "ignored"
    assert result["current_state"] == _TestState.AWAIT_SYS.value


@pytest.mark.asyncio
async def test_status_ignored_symmetric_guard_system_event_against_human_state():
    """Round-2 fix: a system event must never dispatch a wait_kind="human"
    handler, even if expected_events happened to be populated for it."""
    clear_metadata()
    import src.docprocessing.handlers as dp_handlers

    importlib.reload(dp_handlers)

    @handler(
        state=_TestState.HUMAN_ONLY.value,
        waits_for_input=True,
        wait_kind="human",
        expected_events=["should_never_matter"],
    )
    def _handle_human_only(state):
        return {"audit_trail": ["should not be reached"]}

    class _HumanOnlyGraph(_TestGraph):
        def _build_routing_table(self):
            return {_TestState.INIT: _TestState.HUMAN_ONLY}

        def _get_current_state(self, state):
            return _TestState(state.get("current_state", _TestState.INIT.value))

        def _get_proposed_state(self, state):
            return _TestState(state.get("proposed_next", _TestState.HUMAN_ONLY.value))

        def _new_session_state(self):
            return {
                **new_engine_session_state(),
                "current_state": _TestState.HUMAN_ONLY.value,
                "proposed_next": _TestState.HUMAN_ONLY.value,
            }

    handler_map = {_TestState.HUMAN_ONLY: _handle_human_only}
    graph = _HumanOnlyGraph(handler_map)
    checkpointer = JsonCheckpointer(sessions_dir=f"/tmp/test_aemit_event_{uuid.uuid4()}")
    graph.compiled_graph = graph.build_graph(EngineSessionState, checkpointer=checkpointer)
    from src.engine.event_ledger import EventLedger

    graph._ledger = EventLedger(ledger_dir=f"/tmp/test_aemit_event_ledger_{uuid.uuid4()}")

    thread_id = str(uuid.uuid4())
    graph._get_or_init_state(session_id=thread_id, user_id="")  # materialize a fresh session

    result = await graph.aemit_event(
        thread_id=thread_id, source="system", event_type="should_never_matter", event_id="evt-3"
    )

    assert result["status"] == "ignored"


@pytest.mark.asyncio
async def test_status_not_waiting_for_fresh_session():
    graph = _build_test_graph()
    thread_id = str(uuid.uuid4())
    graph._get_or_init_state(session_id=thread_id, user_id="")  # current_state=INIT, not waiting

    result = await graph.aemit_event(
        thread_id=thread_id, source="system", event_type="thing_happened", event_id="evt-4"
    )

    assert result["status"] == "not_waiting"


@pytest.mark.asyncio
async def test_status_already_terminal():
    graph = _build_test_graph()
    thread_id = str(uuid.uuid4())
    await _park_at_await_sys(graph, thread_id)
    await graph.aemit_event(
        thread_id=thread_id, source="system", event_type="thing_happened", event_id="evt-5"
    )  # advances to DONE

    result = await graph.aemit_event(
        thread_id=thread_id, source="system", event_type="thing_happened", event_id="evt-6"
    )

    assert result["status"] == "already_terminal"
    assert result["current_state"] == _TestState.DONE.value


@pytest.mark.asyncio
async def test_system_sourced_event_requires_event_id():
    graph = _build_test_graph()
    with pytest.raises(ValueError):
        await graph.aemit_event(
            thread_id=str(uuid.uuid4()), source="system", event_type="thing_happened", event_id=None
        )


@pytest.mark.asyncio
async def test_user_id_is_always_empty_regardless_of_source():
    """Thread identity must never depend on event source — a human turn and
    a system turn for the same thread_id must land in the SAME session,
    not fork into two checkpointed threads."""
    graph = _build_test_graph()
    thread_id = str(uuid.uuid4())
    await _park_at_await_sys(graph, thread_id)

    result = await graph.aemit_event(
        thread_id=thread_id, source="system", event_type="thing_happened", event_id="evt-7"
    )
    assert result["status"] == "ok"

    # Loading the same thread_id directly (user_id="") must see the
    # post-event state — proving both calls agreed on one thread identity.
    reloaded = graph._get_or_init_state(session_id=thread_id, user_id="")
    assert reloaded["current_state"] == _TestState.DONE.value


@pytest.mark.asyncio
async def test_human_branch_smoke_test_against_real_docprocessing_graph():
    """Validates the gate against a real, pre-existing graph — no
    onboarding-domain code needed to prove the human path works."""
    from src.docprocessing.graph import build_graph

    graph = build_graph(sessions_dir=f"/tmp/test_aemit_event_docprocessing_{uuid.uuid4()}")
    thread_id = str(uuid.uuid4())

    with patch("src.docprocessing.handlers.random.random", return_value=0.9):
        turn1 = await graph.ainvoke(
            user_id="",
            session_id=thread_id,
            input_message="start",
            state_delta={"document_id": "doc1"},
        )
    assert turn1["current_state"] == "upload_documents"

    with patch("src.docprocessing.handlers.random.random", return_value=0.9):
        result = await graph.aemit_event(
            thread_id=thread_id,
            source="human",
            event_type="message",
            payload={"text": "here are the docs"},
        )

    assert result["status"] == "ok"
    assert result["current_state"] == "complete"


@pytest.mark.asyncio
async def test_human_event_against_system_only_state_returns_describe_wait():
    """A human message against a wait_kind="system_event" state must not
    dispatch that handler — it gets a read-only description instead."""
    graph = _build_test_graph()
    thread_id = str(uuid.uuid4())
    await _park_at_await_sys(graph, thread_id)

    result = await graph.aemit_event(
        thread_id=thread_id, source="human", event_type="message", payload={"text": "hello?"}
    )

    assert result["status"] == "waiting"
    assert result["current_state"] == _TestState.AWAIT_SYS.value
    assert result["wait_kind"] == "system_event"
    assert result["expected_events"] == ["thing_happened"]

    # State must be untouched — still parked at AWAIT_SYS, not dispatched.
    reloaded = graph._get_or_init_state(session_id=thread_id, user_id="")
    assert reloaded["current_state"] == _TestState.AWAIT_SYS.value


@pytest.mark.asyncio
async def test_failed_ainvoke_does_not_mark_ledger():
    """Mark only after a successful turn — a status=error result must leave
    the event_id unmarked so a provider retry can recover."""
    graph = _build_test_graph()
    thread_id = str(uuid.uuid4())
    await _park_at_await_sys(graph, thread_id)

    with patch.object(
        graph,
        "ainvoke",
        new_callable=AsyncMock,
        return_value={
            "status": "error",
            "error_message": "boom",
            "current_state": "error",
            "turn_number": 0,
            "semantic_context": {},
            "router_confidence": 0.0,
        },
    ):
        result = await graph.aemit_event(
            thread_id=thread_id,
            source="system",
            event_type="thing_happened",
            event_id="evt-fail",
        )

    assert result["status"] == "error"
    assert await graph._ledger.is_processed("evt-fail") is False
