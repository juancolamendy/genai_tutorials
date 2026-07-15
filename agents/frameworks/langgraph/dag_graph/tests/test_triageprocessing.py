"""Tests for triageprocessing harness (engine gate, not interrupt())."""

import importlib
from pathlib import Path
from uuid import uuid4

import pytest

from src.engine.handler_registry import get_handler_metadata
from src.triageprocessing.graph import build_graph
from src.triageprocessing.state_transitions import State


@pytest.fixture(autouse=True)
def _ensure_handlers_registered():
    if get_handler_metadata(State.AWAIT_APPROVAL.value) is None:
        import src.triageprocessing.handlers as handlers

        importlib.reload(handlers)


def _dirs(tmp_path: Path):
    sessions = str(tmp_path / "sessions")
    arts = str(tmp_path / "artifacts")
    return sessions, arts


@pytest.mark.asyncio
async def test_run_parks_at_await_approval(tmp_path: Path):
    sessions, arts = _dirs(tmp_path)
    graph = build_graph(sessions_dir=sessions, artifacts_dir=arts)
    run_id = f"run-{uuid4().hex[:8]}"

    result = await graph.arun_to_completion(
        user_id="",
        session_id=run_id,
        initial_state_delta={
            "run_id": run_id,
            "trigger": {"source": "sentry", "issue": "PROJ-142"},
            "inputs": {"repo": str(tmp_path / "repo"), "issue": "PROJ-142"},
            "artifacts_dir": arts,
        },
    )

    assert result["status"] == "blocked_needs_input"
    assert result["current_state"] == State.AWAIT_APPROVAL.value
    assert result["run_status"] == "awaiting_approval"
    assert result.get("diff_path")
    assert Path(result["diff_path"]).exists()
    assert "sentry" in result.get("artifacts", {})


@pytest.mark.asyncio
async def test_system_approve_publishes_and_writes_report(tmp_path: Path):
    sessions, arts = _dirs(tmp_path)
    graph = build_graph(sessions_dir=sessions, artifacts_dir=arts)
    run_id = f"run-{uuid4().hex[:8]}"

    await graph.arun_to_completion(
        user_id="",
        session_id=run_id,
        initial_state_delta={
            "run_id": run_id,
            "trigger": {"source": "sentry", "issue": "PROJ-1"},
            "inputs": {"repo": str(tmp_path / "repo"), "issue": "PROJ-1"},
            "artifacts_dir": arts,
        },
    )

    result = await graph.aemit_event(
        thread_id=run_id,
        source="system",
        event_type="approve",
        event_id=f"evt-approve-{run_id}",
        payload={"by": "alice", "note": "lgtm", "approved": True},
    )

    assert result["status"] == "ok"
    assert result["current_state"] == State.COMPLETE.value
    assert result["run_status"] == "published"
    assert result["published"] is True
    assert result.get("report_path")
    assert Path(result["report_path"]).exists()


@pytest.mark.asyncio
async def test_system_reject_lands_in_rejected(tmp_path: Path):
    sessions, arts = _dirs(tmp_path)
    graph = build_graph(sessions_dir=sessions, artifacts_dir=arts)
    run_id = f"run-{uuid4().hex[:8]}"

    await graph.arun_to_completion(
        user_id="",
        session_id=run_id,
        initial_state_delta={
            "run_id": run_id,
            "trigger": {"source": "sentry", "issue": "PROJ-2"},
            "inputs": {"repo": str(tmp_path / "repo"), "issue": "PROJ-2"},
            "artifacts_dir": arts,
        },
    )

    result = await graph.aemit_event(
        thread_id=run_id,
        source="system",
        event_type="reject",
        event_id=f"evt-reject-{run_id}",
        payload={"by": "bob", "note": "needs tests", "approved": False},
    )

    assert result["status"] == "ok"
    assert result["current_state"] == State.REJECTED.value
    assert result["run_status"] == "rejected"
    assert result.get("published") is not True
    assert result.get("report_path")


@pytest.mark.asyncio
async def test_human_chat_approve_either_path(tmp_path: Path):
    sessions, arts = _dirs(tmp_path)
    graph = build_graph(sessions_dir=sessions, artifacts_dir=arts)
    run_id = f"run-{uuid4().hex[:8]}"

    await graph.arun_to_completion(
        user_id="",
        session_id=run_id,
        initial_state_delta={
            "run_id": run_id,
            "trigger": {"source": "sentry", "issue": "PROJ-3"},
            "inputs": {"repo": str(tmp_path / "repo"), "issue": "PROJ-3"},
            "artifacts_dir": arts,
        },
    )

    result = await graph.aemit_event(
        thread_id=run_id,
        source="human",
        event_type="message",
        input_message="approve",
    )

    assert result["status"] == "ok"
    assert result["current_state"] == State.COMPLETE.value
    assert result["published"] is True


@pytest.mark.asyncio
async def test_duplicate_approve_event_id(tmp_path: Path):
    sessions, arts = _dirs(tmp_path)
    graph = build_graph(sessions_dir=sessions, artifacts_dir=arts)
    run_id = f"run-{uuid4().hex[:8]}"
    event_id = f"evt-dup-{run_id}"

    await graph.arun_to_completion(
        user_id="",
        session_id=run_id,
        initial_state_delta={
            "run_id": run_id,
            "trigger": {"source": "sentry", "issue": "PROJ-4"},
            "inputs": {"repo": str(tmp_path / "repo"), "issue": "PROJ-4"},
            "artifacts_dir": arts,
        },
    )

    first = await graph.aemit_event(
        thread_id=run_id,
        source="system",
        event_type="approve",
        event_id=event_id,
        payload={"by": "alice", "approved": True},
    )
    assert first["status"] == "ok"

    second = await graph.aemit_event(
        thread_id=run_id,
        source="system",
        event_type="approve",
        event_id=event_id,
        payload={"by": "alice", "approved": True},
    )
    assert second["status"] == "duplicate"


def test_current_thread_id_persisted_under_sessions_dir(tmp_path: Path):
    from src.triageprocessing.cli import (
        load_current_thread_id,
        resolve_thread_id,
        save_current_thread_id,
    )

    sessions = str(tmp_path / "sessions")
    save_current_thread_id(sessions, "thread-1")
    assert load_current_thread_id(sessions) == "thread-1"
    assert resolve_thread_id(sessions, None) == "thread-1"
    assert resolve_thread_id(sessions, "thread-other") == "thread-other"
