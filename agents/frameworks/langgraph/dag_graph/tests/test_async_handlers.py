"""Hybrid sync/async handler dispatch (option C).

Sync handlers keep working on invoke + ainvoke. Async handlers are awaited
on ainvoke/astream; sync invoke refuses them with a clear TypeError.
"""

from __future__ import annotations

import importlib
from enum import Enum
from uuid import uuid4

import pytest

from src.engine.engine_graph import EngineGraph, safe_node
from src.engine.engine_session_state import EngineSessionState, new_engine_session_state
from src.engine.handler_registry import clear_metadata, get_handler_metadata, handler
from src.engine.json_checkpointer import JsonCheckpointer


@pytest.fixture(autouse=True)
def _ensure_docprocessing_handlers_registered():
    if get_handler_metadata("upload_documents") is None:
        import src.docprocessing.handlers as dp_handlers

        importlib.reload(dp_handlers)


class _AsyncHState(str, Enum):
    INIT = "init"
    WORK = "async_h_work"
    DONE = "async_h_done"
    ERROR = "async_h_error"


def _register_handlers(*, work_async: bool):
    clear_metadata()
    import src.docprocessing.handlers as dp_handlers

    importlib.reload(dp_handlers)

    if work_async:

        @handler(state=_AsyncHState.WORK.value, waits_for_input=False)
        async def _handle_work(state):
            return {"audit_trail": ["work-async"], "handler_status": "ok"}

    else:

        @handler(state=_AsyncHState.WORK.value, waits_for_input=False)
        def _handle_work(state):
            return {"audit_trail": ["work-sync"], "handler_status": "ok"}

    @handler(state=_AsyncHState.DONE.value, waits_for_input=False)
    def _handle_done(state):
        return {"audit_trail": ["done"]}

    @handler(state=_AsyncHState.ERROR.value, waits_for_input=False)
    def _handle_error(state):
        return {"audit_trail": ["error"]}

    return {
        _AsyncHState.WORK: _handle_work,
        _AsyncHState.DONE: _handle_done,
        _AsyncHState.ERROR: _handle_error,
    }


class _AsyncHGraph(EngineGraph):
    state_enum = _AsyncHState
    terminal_states = {_AsyncHState.DONE, _AsyncHState.ERROR}

    def __init__(self, handler_map):
        super().__init__()
        self.handler_map = handler_map

    def _build_routing_table(self):
        return {_AsyncHState.INIT: _AsyncHState.WORK, _AsyncHState.WORK: _AsyncHState.DONE}

    def _get_current_state(self, state):
        return _AsyncHState(state.get("current_state", _AsyncHState.INIT.value))

    def _get_proposed_state(self, state):
        return _AsyncHState(state.get("proposed_next", _AsyncHState.WORK.value))

    def _get_guardrails(self):
        return {}

    def _new_session_state(self):
        return {
            **new_engine_session_state(),
            "current_state": _AsyncHState.INIT.value,
            "proposed_next": _AsyncHState.WORK.value,
        }


def _build(work_async: bool) -> _AsyncHGraph:
    graph = _AsyncHGraph(_register_handlers(work_async=work_async))
    checkpointer = JsonCheckpointer(sessions_dir=f"/tmp/test_async_h_{uuid4()}")
    graph.compiled_graph = graph.build_graph(EngineSessionState, checkpointer=checkpointer)
    return graph


@pytest.mark.asyncio
async def test_ainvoke_runs_async_handler():
    graph = _build(work_async=True)
    result = await graph.ainvoke(user_id="", session_id=str(uuid4()), input_message="")
    assert result["session_status"] == "ok"
    assert result["current_state"] == _AsyncHState.DONE.value
    assert "work-async" in (result.get("audit_trail") or [])


@pytest.mark.asyncio
async def test_ainvoke_still_runs_sync_handler():
    graph = _build(work_async=False)
    result = await graph.ainvoke(user_id="", session_id=str(uuid4()), input_message="")
    assert result["current_state"] == _AsyncHState.DONE.value
    assert "work-sync" in (result.get("audit_trail") or [])


def test_invoke_runs_sync_handler():
    graph = _build(work_async=False)
    result = graph.invoke(user_id="", session_id=str(uuid4()), input_message="")
    assert result["current_state"] == _AsyncHState.DONE.value
    assert "work-sync" in (result.get("audit_trail") or [])


def test_sync_dispatch_rejects_async_handler():
    graph = _build(work_async=True)
    with pytest.raises(TypeError, match="async"):
        graph._dispatch_handler(_AsyncHState.WORK, {"current_state": "init"})


@pytest.mark.asyncio
async def test_safe_node_wraps_async_and_catches():
    @safe_node
    async def boom(state):
        raise RuntimeError("kaboom")

    out = await boom({"x": 1})
    assert out["session_status"] == "error"
    assert out["proposed_next"] == "error"
    assert "kaboom" in out["error_message"]
