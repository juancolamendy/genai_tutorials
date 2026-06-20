"""Public API: run_pipeline for document processing workflow."""

import time
import logging
from src.workflow.state_machine import State, PipelineState
from src.workflow.graph import build_graph

log = logging.getLogger(__name__)


def run_pipeline(
    document_id: str,
    session_id: str = None,
    initial_state: PipelineState = None,
    timeout_seconds: int = 60,
) -> PipelineState:
    """Execute a document through the state machine graph.

    This is the primary public API for running the pipeline.

    Args:
        document_id: Unique document identifier (required, non-empty, ≤256 chars)
        session_id: Optional session ID for checkpoint resume (not implemented yet)
        initial_state: Optional pre-built state dict (validates shape)
        timeout_seconds: Global timeout per pipeline (default 60s)

    Returns:
        Final PipelineState with results, error_message (if error), audit_trail

    Raises:
        ValueError: If document_id invalid or initial_state malformed
    """
    # ── Input Validation ──────────────────────────────────────────────────
    if not document_id:
        raise ValueError("document_id cannot be empty")

    if len(document_id) > 256:
        raise ValueError("document_id exceeds max length (256)")

    log.info(
        "[run_pipeline] Starting document processing: %s (timeout=%ds)",
        document_id,
        timeout_seconds,
    )

    # ── Initialize State ──────────────────────────────────────────────────
    if initial_state is None:
        initial_state: PipelineState = {
            "current_state": State.INIT.value,
            "proposed_next": State.FETCH.value,
            "retry_count": 0,
            "error_message": None,
            "error_type": None,
            "audit_trail": ["init"],
            "fallback_depth": 0,
            "started_at": time.time(),
            "node_timeout_seconds": timeout_seconds,
            "document_id": document_id,
            "raw_data": None,
            "validated_data": None,
            "enriched_data": None,
        }
    else:
        # Validate pre-built state
        assert isinstance(initial_state, dict), "initial_state must be dict"
        assert "document_id" in initial_state, "initial_state missing document_id"
        initial_state.setdefault("started_at", time.time())
        initial_state.setdefault("node_timeout_seconds", timeout_seconds)
        initial_state.setdefault("fallback_depth", 0)
        initial_state.setdefault("error_type", None)

    # ── Resume from checkpoint if provided ─────────────────────────────────
    if session_id:
        log.warning(
            "[run_pipeline] session_id provided but checkpoint feature not yet implemented"
        )
        # In Phase 8+, would load checkpoint here
        # initial_state = load_checkpoint(session_id)

    # ── Invoke graph ──────────────────────────────────────────────────────
    log.info("[run_pipeline] Building and invoking graph")
    graph = build_graph()
    final_state = graph.invoke(initial_state)

    # ── Log completion ────────────────────────────────────────────────────
    elapsed = time.time() - initial_state["started_at"]
    log.info(
        "[run_pipeline] Completed: %s → %s (elapsed=%.2fs, retry_count=%d)",
        document_id,
        final_state["current_state"],
        elapsed,
        final_state["retry_count"],
    )

    return final_state


def run_pipeline_with_checkpoint(
    session_id: str,
    checkpoint_key: str = None,
) -> PipelineState:
    """Resume pipeline from a saved checkpoint.

    Args:
        session_id: Session ID to load checkpoint from
        checkpoint_key: Optional specific checkpoint key (default: latest)

    Returns:
        Final PipelineState after resuming from checkpoint

    Raises:
        KeyError: If session_id or checkpoint_key not found
    """
    log.warning("[run_pipeline_with_checkpoint] Checkpoint feature not yet implemented")
    raise NotImplementedError("Checkpoint resume not yet implemented in Phase 8")
