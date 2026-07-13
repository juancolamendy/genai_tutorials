"""Tests for async checkpointer support (design spec §5.2/§14).

BaseCheckpointSaver.aget_tuple/aget/aput default to `raise NotImplementedError`
in installed langgraph==1.2.6 — there is no sync-to-thread-executor fallback.
Confirmed by executing `await compiled_graph.ainvoke(...)` against the
existing JsonCheckpointer before this phase's fix: it raised NotImplementedError
immediately. Both JsonCheckpointer and SqliteCheckpointer need explicit async
methods before EngineGraph.ainvoke() (a later phase) can run at all.
"""

import asyncio
import uuid

import pytest

from src.docprocessing.graph import Graph
from src.docprocessing.session_state import SessionState, new_session_state
from src.engine.json_checkpointer import JsonCheckpointer
from src.engine.sqlite_checkpointing import SqliteCheckpointer


def _build_graph_with(checkpointer):
    graph = Graph()
    graph.compiled_graph = graph.build_graph(SessionState, checkpointer=checkpointer)
    return graph


@pytest.mark.asyncio
async def test_ainvoke_works_against_json_checkpointer():
    """Regression test for the confirmed crash: compiled_graph.ainvoke() must
    round-trip a real checkpoint through JsonCheckpointer without raising."""
    checkpointer = JsonCheckpointer(sessions_dir=f"/tmp/test_ckpt_async_{uuid.uuid4()}")
    graph = _build_graph_with(checkpointer)
    thread_id = f"user1:{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    state = {**new_session_state(), "document_id": "doc1"}

    result = await graph.compiled_graph.ainvoke(state, config=config)

    assert result["status"] != "error"
    # A real checkpoint round-trip: the state we get back must be loadable
    # again via the sync path too (proves aput actually persisted it).
    loaded = checkpointer.get_tuple(config)
    assert loaded is not None


@pytest.mark.asyncio
async def test_ainvoke_works_against_sqlite_checkpointer_memory():
    """Same regression test against the in-memory SqliteCheckpointer."""
    checkpointer = SqliteCheckpointer(db_path=":memory:")
    graph = _build_graph_with(checkpointer)
    thread_id = f"user1:{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    state = {**new_session_state(), "document_id": "doc1"}

    result = await graph.compiled_graph.ainvoke(state, config=config)

    assert result["status"] != "error"
    loaded = checkpointer.get_tuple(config)
    assert loaded is not None


@pytest.mark.asyncio
async def test_ainvoke_works_against_sqlite_checkpointer_file(tmp_path):
    """Same regression test against a file-backed SqliteCheckpointer."""
    db_path = str(tmp_path / "test.db")
    checkpointer = SqliteCheckpointer(db_path=db_path)
    graph = _build_graph_with(checkpointer)
    thread_id = f"user1:{uuid.uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    state = {**new_session_state(), "document_id": "doc1"}

    result = await graph.compiled_graph.ainvoke(state, config=config)

    assert result["status"] != "error"
    loaded = checkpointer.get_tuple(config)
    assert loaded is not None


@pytest.mark.asyncio
async def test_sqlite_checkpointer_concurrent_access_different_threads():
    """Two asyncio.to_thread-wrapped calls against the same in-memory
    SqliteCheckpointer, for two different thread_ids, run concurrently
    without raising sqlite3.OperationalError and both persist correctly.
    This is the scenario the design flagged as unsafe: sqlite3.Connection
    objects are not safe for concurrent access from multiple threads even
    with check_same_thread=False."""
    checkpointer = SqliteCheckpointer(db_path=":memory:")
    graph1 = _build_graph_with(checkpointer)
    graph2 = _build_graph_with(checkpointer)

    thread_id_1 = f"user1:{uuid.uuid4()}"
    thread_id_2 = f"user2:{uuid.uuid4()}"
    config1 = {"configurable": {"thread_id": thread_id_1}}
    config2 = {"configurable": {"thread_id": thread_id_2}}
    state1 = {**new_session_state(), "document_id": "doc1"}
    state2 = {**new_session_state(), "document_id": "doc2"}

    results = await asyncio.gather(
        graph1.compiled_graph.ainvoke(state1, config=config1),
        graph2.compiled_graph.ainvoke(state2, config=config2),
    )

    assert all(r["status"] != "error" for r in results)
    assert checkpointer.get_tuple(config1) is not None
    assert checkpointer.get_tuple(config2) is not None


@pytest.mark.asyncio
async def test_json_checkpointer_async_methods_round_trip_directly():
    """Unit-level test of the async methods themselves, not just via the
    graph: aput then aget_tuple must return what was stored."""
    checkpointer = JsonCheckpointer(sessions_dir=f"/tmp/test_ckpt_async_{uuid.uuid4()}")
    config = {"configurable": {"thread_id": "direct-test"}}
    checkpoint = {"v": 1, "channel_values": {"foo": "bar"}}
    metadata = {"checkpoint_id": "cp1"}

    await checkpointer.aput(config, checkpoint, metadata, None)
    result = await checkpointer.aget_tuple(config)

    assert result is not None
    assert result.checkpoint["channel_values"]["foo"] == "bar"
