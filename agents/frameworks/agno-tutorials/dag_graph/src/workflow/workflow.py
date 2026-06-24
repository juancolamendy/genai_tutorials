"""
workflow/workflow.py
────────────────────────────────────────────────────────────────────────────
DocPipelineWorkflow — document processing state machine implementation.

Inherits generic state machine logic from engine.workflow.StateMachineWorkflow
and provides only business-specific behavior:
  • Routing table (_HAPPY_PATH) for document processing
  • Guardrail runner that delegates to guardrails module
  • Session initialization with document-specific defaults
  • Multi-run resume and audit trail persistence
"""

from __future__ import annotations

import logging
from typing import Any

from agno.db.json.json_db import JsonDb

from engine.statemachine_workflow import StateMachineWorkflow

from .handlers import HANDLER_MAP
from .guardrails import run_guardrail
from .pipeline_state import PipelineState, new_pipeline, pretty_audit
from .router import DocPipelineRouter
from .session import init_session_defaults
from .state_machine import State, TERMINAL_STATES

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# Happy-path routing table: default transitions when no guardrail redirects.
# Guardrails can override this to route to RETRY, HUMAN_REVIEW, or ERROR.
_HAPPY_PATH: dict[State, State] = {
    State.INIT:         State.FETCH,
    State.FETCH:        State.VALIDATE,
    State.VALIDATE:     State.ENRICH,
    State.ENRICH:       State.STORE,
    State.STORE:        State.COMPLETE,
    State.RETRY:        State.FETCH,
    State.HUMAN_REVIEW: State.ENRICH,
}

_PIPELINE_KEYS = (
    "current_state", "proposed_next", "guardrail_ok",
    "retry_count", "error_message", "audit_trail",
    "document_id", "raw_data", "validated_data", "enriched_data",
)


class DocPipelineWorkflow(StateMachineWorkflow):
    """
    Multi-run document processing workflow with state machine + guardrails.

    session_state keys (all persisted to db):
        current_state   str   — active pipeline state
        proposed_next   str   — router's candidate
        guardrail_ok    bool  — True iff the last guardrail passed
        retry_count     int
        error_message   str | None
        audit_trail     list[str]
        document_id     str
        raw_data        dict | None
        validated_data  dict | None
        enriched_data   dict | None
        pipeline_runs   list  — summary of all completed runs
    """

    # Bind class variables for base class
    _STATE_KEYS = _PIPELINE_KEYS
    _STATE_ENUM = State
    _TERMINAL_STATES = TERMINAL_STATES
    HANDLER_MAP = HANDLER_MAP

    def __post_init__(self) -> None:
        """Initialize base class and semantic router for multi-turn support."""
        super().__post_init__()
        # Initialize semantic router for LLM-powered state classification
        self.router = DocPipelineRouter()

    def _init_session_defaults(self) -> None:
        """Initialize session with pipeline-specific defaults."""
        init_session_defaults(self.session_state, mode="pipeline")

    def _build_routing_table(self) -> dict[State, State]:
        """Return the happy-path routing table."""
        return _HAPPY_PATH

    def _get_current_state(self, session_state: dict[str, Any]) -> State:
        """Extract current state from session_state."""
        return State(session_state.get("current_state", State.INIT.value))

    def _get_proposed_state(self, session_state: dict[str, Any]) -> State:
        """Extract proposed next state from session_state."""
        return State(session_state.get("proposed_next", State.FETCH.value))

    def _run_guardrail(self, state_dict: dict[str, Any]) -> tuple[dict[str, Any], Any]:
        """Run guardrails on proposed transition."""
        return run_guardrail(state_dict)

    def _new_session_state(self, entity_id: str) -> dict[str, Any]:
        """Initialize fresh session state for a new document."""
        return new_pipeline(entity_id)

    def _build_response(self, entity_id: str) -> PipelineState:
        """Build PipelineState from current session_state."""
        final = PipelineState(
            current_state=self.session_state["current_state"],
            proposed_next=self.session_state["proposed_next"],
            retry_count=self.session_state["retry_count"],
            error_message=self.session_state.get("error_message"),
            guardrail_ok=self.session_state["guardrail_ok"],
            audit_trail=self.session_state["audit_trail"],
            document_id=self.session_state["document_id"],
            raw_data=self.session_state.get("raw_data"),
            validated_data=self.session_state.get("validated_data"),
            enriched_data=self.session_state.get("enriched_data"),
        )
        print(pretty_audit(final))
        return final


# ── Factory ────────────────────────────────────────────────────────────────────

def build_doc_pipeline(
    session_id:   str,
    sessions_dir: str  = ".doc_sessions",
    *,
    debug:        bool = False,
) -> DocPipelineWorkflow:
    """
    Create (or resume) a DocPipelineWorkflow for the given session_id.

    JsonDb persists session_state to disk so audit trails and pipeline_runs
    history survive process restarts.
    """
    return DocPipelineWorkflow(
        name       = "DocPipelineWorkflow",
        session_id = session_id,
        db         = JsonDb(db_path=sessions_dir),
        debug_mode = debug,
    )