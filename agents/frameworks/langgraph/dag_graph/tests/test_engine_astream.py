"""Tests for EngineGraph.astream / aemit_event_stream and get_model."""

import importlib
from uuid import uuid4

import pytest

from src.docprocessing.graph import build_graph
from src.engine.chains import DEFAULT_MODEL, get_model
from src.engine.handler_registry import get_handler_metadata


@pytest.fixture(autouse=True)
def _ensure_docprocessing_handlers_registered():
    if get_handler_metadata("upload_documents") is None:
        import src.docprocessing.handlers as dp_handlers

        importlib.reload(dp_handlers)


def test_get_model_returns_default_for_known_roles():
    assert get_model("router") == DEFAULT_MODEL
    assert get_model("topic") == DEFAULT_MODEL
    assert get_model("unknown_role") == DEFAULT_MODEL


@pytest.mark.asyncio
async def test_astream_yields_result_matching_ainvoke_state(monkeypatch):
    """Streaming path ends with a result whose current_state matches ainvoke."""
    sessions = f"/tmp/test_astream_{uuid4()}"
    graph = build_graph(sessions_dir=sessions)
    session_id = str(uuid4())

    from unittest.mock import patch

    with patch("src.docprocessing.handlers.random.random", return_value=0.9):
        ainvoke_state = await graph.ainvoke(
            user_id="",
            session_id=session_id + "-a",
            input_message="please process",
            state_delta={"document_id": "doc1"},
        )

    graph2 = build_graph(sessions_dir=sessions)
    tokens = []
    result_state = None
    with patch("src.docprocessing.handlers.random.random", return_value=0.9):
        async for chunk in graph2.astream(
            user_id="",
            session_id=session_id + "-b",
            input_message="please process",
            state_delta={"document_id": "doc1"},
        ):
            if chunk.get("type") == "token":
                tokens.append(chunk["text"])
            elif chunk.get("type") == "result":
                result_state = chunk["state"]

    assert result_state is not None
    assert result_state.get("current_state") == ainvoke_state.get("current_state")
    assert result_state.get("session_status") != "error"


@pytest.mark.asyncio
async def test_aemit_event_stream_human_yields_ok_result():
    sessions = f"/tmp/test_aemit_stream_{uuid4()}"
    graph = build_graph(sessions_dir=sessions)
    thread_id = str(uuid4())

    from unittest.mock import patch

    result = None
    with patch("src.docprocessing.handlers.random.random", return_value=0.9):
        # Seed via non-stream so we park at upload_documents
        await graph.aemit_event(
            thread_id=thread_id,
            source="human",
            event_type="message",
            input_message="start",
        )
        # Need document_id — docprocessing may need state; use ainvoke seed
    graph = build_graph(sessions_dir=sessions)
    with patch("src.docprocessing.handlers.random.random", return_value=0.9):
        await graph.ainvoke(
            user_id="",
            session_id=thread_id,
            input_message="start",
            state_delta={"document_id": "doc1"},
        )
        async for chunk in graph.aemit_event_stream(
            thread_id=thread_id,
            source="human",
            event_type="message",
            input_message="here are docs",
        ):
            if chunk.get("type") == "result":
                result = chunk

    assert result is not None
    assert result.get("emit_status") == "ok"
    assert result["state"].get("current_state") in {
        "upload_documents",
        "complete",
        "validate",
        "enrich",
        "store",
        "human_review",
        "retry",
        "error",
    }
