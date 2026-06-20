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

from engine.workflow import StateMachineWorkflow

from .handlers import HANDLER_MAP
from .guardrails import run_guardrail
from .pipeline_state import PipelineState, new_pipeline, pretty_audit
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

    # ── Workflow entry: initialise and run ────────────────────────────────────

    def process(self, document_id: str) -> PipelineState:
        """
        Run the full state machine for `document_id` and return the final
        PipelineState.

        If the session already has a completed pipeline for this document, a
        fresh PipelineState is created (the session accumulates runs in
        pipeline_runs for audit purposes).
        """
        fresh = new_pipeline(document_id)
        # Ensure session_state is initialized
        if self.session_state is None:
            self.session_state = {}
        # Ensure steps are initialized
        if self.steps is None or len(self.steps) == 0:
            self._init_steps()
        # Seed session_state from the fresh pipeline
        _pipeline_to_ss(fresh, self.session_state)

        self.run(input=document_id)

        # Snapshot to pipeline_runs history
        final = _ss_to_pipeline(self.session_state)
        self.session_state.setdefault("pipeline_runs", []).append({
            "document_id": document_id,
            "final_state": final["current_state"],
            "retry_count": final["retry_count"],
            "audit_trail": final["audit_trail"],
        })

        print(pretty_audit(final))
        return final


def _ss_to_pipeline(ss: dict[str, Any]) -> PipelineState:
    """Read pipeline-related keys from session_state into a PipelineState dict."""
    return PipelineState(
        current_state  = ss.get("current_state",  State.INIT.value),
        proposed_next  = ss.get("proposed_next",  State.FETCH.value),
        retry_count    = ss.get("retry_count",    0),
        error_message  = ss.get("error_message"),
        guardrail_ok   = ss.get("guardrail_ok",   True),
        audit_trail    = list(ss.get("audit_trail", [])),
        document_id    = ss.get("document_id",    ""),
        raw_data       = ss.get("raw_data"),
        validated_data = ss.get("validated_data"),
        enriched_data  = ss.get("enriched_data"),
    )


def _pipeline_to_ss(p: PipelineState, ss: dict[str, Any]) -> None:
    """Write all pipeline state keys back into session_state in-place."""
    for k in _PIPELINE_KEYS:
        if k in p:
            ss[k] = p[k]


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