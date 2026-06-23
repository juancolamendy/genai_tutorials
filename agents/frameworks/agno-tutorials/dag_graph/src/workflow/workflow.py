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
        self._ensure_initialized()
        self.session_state.update(new_pipeline(document_id))

        self.run(input=document_id)

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

        self.session_state.setdefault("pipeline_runs", []).append({
            "document_id": document_id,
            "final_state": final["current_state"],
            "retry_count": final["retry_count"],
            "audit_trail": final["audit_trail"],
        })

        print(pretty_audit(final))
        return final

    def process_turn(self,
                     user_id: str,
                     session_id: str,
                     turn_input: str,
                     timeout_sec: float = 10.0) -> dict[str, Any]:
        """
        Execute one turn of a multi-turn conversation.

        Args:
            user_id: Caller identity
            session_id: Multi-turn session ID
            turn_input: User's input text
            timeout_sec: LLM router timeout

        Returns:
            {
              "current_state": str,
              "waits_for_input": bool,
              "turn_number": int,
              "semantic_context": dict,
              "router_confidence": float,
              "error": str | None
            }
        """
        from engine.input_validation import validate_turn_input, escape_for_llm, InputValidationError

        try:
            validate_turn_input(turn_input)
            escaped = escape_for_llm(turn_input)
            self._ensure_initialized()

            self._prepare_turn_metadata(escaped, timeout_sec)
            self.run(session_id=session_id, user_id=user_id)
            self._trim_history()

            return self._build_turn_response()

        except InputValidationError as e:
            return {"error": str(e), "current_state": None, "waits_for_input": False}
        except Exception as e:
            log.exception("process_turn failed: %s", e)
            return {"error": str(e), "current_state": "error", "waits_for_input": False}

    def _trim_history(self) -> None:
        """Keep only last max_history_turns in session_state."""
        max_turns = self.session_state.get("max_history_turns", 10)
        turns = self.session_state.get("turns", [])
        if len(turns) > max_turns:
            dropped = len(turns) - max_turns
            self.session_state["turns"] = turns[-max_turns:]
            log.info(f"Trimmed {dropped} turns; keeping last {max_turns}")

    def _prepare_turn_metadata(self, turn_input: str, timeout_sec: float) -> None:
        """Prepare session_state for a new turn."""
        turn_num = self.session_state.get("turn_number", 0)
        self.session_state.update({
            "turn_input": turn_input,
            "turn_number": turn_num + 1,
            "router_timeout_sec": timeout_sec,
        })

    def _build_turn_response(self) -> dict[str, Any]:
        """Build response dict from current session_state."""
        from engine.handler_registry import does_state_wait_for_input

        current = self.session_state.get("current_state", "init")
        return {
            "current_state": current,
            "waits_for_input": does_state_wait_for_input(current),
            "turn_number": self.session_state.get("turn_number", 0),
            "semantic_context": self.session_state.get("semantic_context", {}),
            "router_confidence": self.session_state.get("router_confidence", 0.0),
            "error": self.session_state.get("error_message")
        }


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