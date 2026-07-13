"""Tests for EngineGraph.ainvoke() and the per-thread asyncio.Lock registry
(design spec §5/§14, Phase 4).
"""

import asyncio
import importlib
from unittest.mock import patch
from uuid import uuid4

import pytest

from src.docprocessing.graph import build_graph
from src.engine.handler_registry import get_handler_metadata


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
