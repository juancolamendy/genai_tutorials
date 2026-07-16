"""Tests for EngineGraph.arun_to_completion() (design spec §12 — corrected
twice: the round-1 mid-flow guard, and the round-2 fix for terminal-thread
reuse, plus the user-confirmed graceful-degradation outcome for the
needs-input case).

A minimal test-local graph (INIT -> WAITING (wait) -> DONE, with a
current_event_type=="force_error" guardrail diverting WAITING's proposal to
ERROR) exercises all
four outcomes without needing a domain that pauses unconditionally
(docprocessing always parks at UPLOAD_DOCUMENTS, which would make it
impossible to reach a terminal state via arun_to_completion alone in one
call — not representative of the bg/no-interaction use case this method
is actually for).
"""

import importlib
import uuid
from enum import Enum
from unittest.mock import AsyncMock, patch

import pytest

from src.engine.engine_graph import EngineGraph
from src.engine.engine_session_state import EngineSessionState, new_engine_session_state
from src.engine.errors import (
    GraphAlreadyCompleteError,
    GraphAlreadyInteractiveError,
    GraphRunError,
)
from src.engine.guardrail import GuardrailResult, make_guardrail
from src.engine.handler_registry import clear_metadata, get_handler_metadata, handler
from src.engine.json_checkpointer import JsonCheckpointer


@pytest.fixture(autouse=True)
def _ensure_docprocessing_handlers_registered():
    """See tests/test_invoke_helpers.py for why this guard is needed."""
    if get_handler_metadata("upload_documents") is None:
        import src.docprocessing.handlers as dp_handlers

        importlib.reload(dp_handlers)


class _RtcState(str, Enum):
    # "init" is a literal-string assumption baked into arun_to_completion's
    # freshness check (matches the reviewed design's §12 pseudocode exactly:
    # `if current not in ("init",) ...`) — every real domain in this
    # codebase (docprocessing's State.INIT) already uses this same literal
    # value, so this test fixture matches that convention rather than
    # exercise a case the design doesn't actually support.
    INIT = "init"
    WAITING = "rtc_waiting"
    DONE = "rtc_done"
    ERROR = "rtc_error"


def _check_force_error(state):
    # Uses current_event_type (declared on EngineSessionState since Phase 1)
    # rather than an ad-hoc scratch field: an undeclared key passed via
    # state_delta is silently dropped by LangGraph's channel machinery
    # (StateGraph only tracks fields the schema declares), so a guardrail
    # running on a later node invocation would never actually see it.
    if state.get("current_event_type") == "force_error":
        return GuardrailResult(passed=False, reason="forced for test", fallback=_RtcState.ERROR)
    return GuardrailResult(passed=True)


def _register_rtc_handlers():
    clear_metadata()
    import src.docprocessing.handlers as dp_handlers

    importlib.reload(dp_handlers)  # restore docprocessing's registrations too

    @handler(state=_RtcState.WAITING.value, waits_for_input=True)
    def _handle_waiting(state):
        return {"audit_trail": ["waiting handled"]}

    @handler(state=_RtcState.DONE.value, waits_for_input=False)
    def _handle_done(state):
        return {"audit_trail": ["done"]}

    @handler(state=_RtcState.ERROR.value, waits_for_input=False)
    def _handle_error(state):
        return {"audit_trail": ["error"]}

    return {
        _RtcState.WAITING: _handle_waiting,
        _RtcState.DONE: _handle_done,
        _RtcState.ERROR: _handle_error,
    }


class _RtcGraph(EngineGraph):
    state_enum = _RtcState
    terminal_states = {_RtcState.DONE, _RtcState.ERROR}

    def __init__(self, handler_map):
        super().__init__()
        self.handler_map = handler_map

    def _build_routing_table(self):
        return {_RtcState.INIT: _RtcState.WAITING, _RtcState.WAITING: _RtcState.DONE}

    def _get_current_state(self, state):
        return _RtcState(state.get("current_state", _RtcState.INIT.value))

    def _get_proposed_state(self, state):
        return _RtcState(state.get("proposed_next", _RtcState.WAITING.value))

    def _get_guardrails(self):
        return {_RtcState.WAITING: make_guardrail(_check_force_error)}

    def _new_session_state(self):
        return {
            **new_engine_session_state(),
            "current_state": _RtcState.INIT.value,
            "proposed_next": _RtcState.WAITING.value,
        }


def _build_rtc_graph():
    graph = _RtcGraph(_register_rtc_handlers())
    checkpointer = JsonCheckpointer(sessions_dir=f"/tmp/test_rtc_{uuid.uuid4()}")
    graph.compiled_graph = graph.build_graph(EngineSessionState, checkpointer=checkpointer)
    return graph


@pytest.mark.asyncio
async def test_blocked_needs_input_when_parked_at_wait_state():
    """User-confirmed: graceful degradation, not a raised exception, for a
    run that ends up needing interaction."""
    graph = _build_rtc_graph()
    session_id = str(uuid.uuid4())

    result = await graph.arun_to_completion(
        user_id="", session_id=session_id, initial_state_delta={}
    )

    assert result["emit_status"] == "blocked_needs_input"
    assert result["current_state"] == _RtcState.WAITING.value


@pytest.mark.asyncio
async def test_already_interactive_error_for_mid_flow_reuse():
    graph = _build_rtc_graph()
    session_id = str(uuid.uuid4())
    await graph.arun_to_completion(user_id="", session_id=session_id, initial_state_delta={})

    with pytest.raises(GraphAlreadyInteractiveError):
        await graph.arun_to_completion(user_id="", session_id=session_id, initial_state_delta={})


@pytest.mark.asyncio
async def test_already_complete_error_for_terminal_reuse():
    """Regression guard for the round-2 finding: a naive
    `current not in terminal_states` check would have let this second call
    silently clobber the finished record's fields instead of raising."""
    graph = _build_rtc_graph()
    session_id = str(uuid.uuid4())

    # Reach WAITING via arun_to_completion, then push past it to DONE via
    # ainvoke() directly (simulating aemit_event resuming it elsewhere).
    await graph.arun_to_completion(user_id="", session_id=session_id, initial_state_delta={})
    resumed = await graph.ainvoke(
        user_id="", session_id=session_id, input_message="", state_delta={}
    )
    assert resumed["current_state"] == _RtcState.DONE.value

    with pytest.raises(GraphAlreadyCompleteError):
        await graph.arun_to_completion(user_id="", session_id=session_id, initial_state_delta={})


@pytest.mark.asyncio
async def test_graph_run_error_for_status_error_outcome():
    graph = _build_rtc_graph()
    session_id = str(uuid.uuid4())

    with pytest.raises(GraphRunError):
        await graph.arun_to_completion(
            user_id="",
            session_id=session_id,
            initial_state_delta={"current_event_type": "force_error"},
        )


@pytest.mark.asyncio
async def test_fresh_session_id_after_completion_starts_a_new_run():
    """Sanity check that GraphAlreadyCompleteError is about session_id
    reuse specifically, not about the graph itself becoming unusable."""
    graph = _build_rtc_graph()
    session_id_1 = str(uuid.uuid4())
    await graph.arun_to_completion(user_id="", session_id=session_id_1, initial_state_delta={})

    session_id_2 = str(uuid.uuid4())
    result = await graph.arun_to_completion(
        user_id="", session_id=session_id_2, initial_state_delta={}
    )
    assert result["emit_status"] == "blocked_needs_input"


@pytest.mark.asyncio
async def test_arun_to_completion_success_sets_emit_status_ok():
    """Terminal success must stamp emit_status (not only session_status) so
    callers/CLIs that switched off the overloaded status field keep working."""
    graph = _build_rtc_graph()
    session_id = str(uuid.uuid4())

    with patch.object(graph, "ainvoke", new_callable=AsyncMock) as mock_ainvoke:
        mock_ainvoke.return_value = {
            "session_status": "ok",
            "current_state": _RtcState.DONE.value,
            "turn_number": 1,
            "semantic_context": {},
            "router_confidence": 0.0,
            "messages": [],
        }
        result = await graph.arun_to_completion(
            user_id="", session_id=session_id, initial_state_delta={}
        )

    assert result["emit_status"] == "ok"
    assert result["session_status"] == "ok"
    assert result["current_state"] == _RtcState.DONE.value


@pytest.mark.asyncio
async def test_arun_to_completion_forwards_max_auto_iters_to_ainvoke():
    """max_auto_iters must reach ainvoke (and thus auto-progress), not only
    appear in GraphIncompleteError message text. Default is 10."""
    graph = _build_rtc_graph()
    session_id = str(uuid.uuid4())

    with patch.object(graph, "ainvoke", new_callable=AsyncMock) as mock_ainvoke:
        mock_ainvoke.return_value = {
            "session_status": "ok",
            "current_state": _RtcState.WAITING.value,
            "turn_number": 1,
            "semantic_context": {},
            "router_confidence": 0.0,
            "messages": [],
        }
        await graph.arun_to_completion(
            user_id="",
            session_id=session_id,
            initial_state_delta={},
            max_auto_iters=7,
        )

    assert mock_ainvoke.await_args.kwargs["max_auto_iters"] == 7


@pytest.mark.asyncio
async def test_arun_to_completion_default_max_auto_iters_is_10():
    graph = _build_rtc_graph()
    session_id = str(uuid.uuid4())

    with patch.object(graph, "ainvoke", new_callable=AsyncMock) as mock_ainvoke:
        mock_ainvoke.return_value = {
            "session_status": "ok",
            "current_state": _RtcState.WAITING.value,
            "turn_number": 1,
            "semantic_context": {},
            "router_confidence": 0.0,
            "messages": [],
        }
        await graph.arun_to_completion(
            user_id="", session_id=session_id, initial_state_delta={}
        )

    assert mock_ainvoke.await_args.kwargs["max_auto_iters"] == 10
