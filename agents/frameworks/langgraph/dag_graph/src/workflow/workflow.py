"""LangGraph document processing workflow entrypoint.

Orchestrates the complete pipeline: Router → Guardrail → Handler → Loop/End
Includes checkpointing, timeouts, and input validation.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from .graph import build_graph
from .state_machine import State, PipelineState
from .validation import validate_pipeline_state, ValidationError
from src.engine.checkpointing import init_checkpointer, get_checkpointer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def run_pipeline(
    document_id: str,
    timeout_seconds: float = 300,
    thread_id: Optional[str] = None,
    resume: bool = False,
) -> PipelineState:
    """Run the full document processing pipeline.

    Executes: State Machine → Router → Guardrail → Handler (loop) → End

    Features:
      • Input validation (document ID, content)
      • Timeout protection (default 5 minutes)
      • Checkpointing for resumable execution
      • Error handling with graceful degradation

    Args:
        document_id: ID of document to process
        timeout_seconds: Maximum pipeline execution time (default 300s)
        thread_id: Unique thread ID for checkpointing (auto-generated if None)
        resume: Resume from last checkpoint if True

    Returns:
        Final PipelineState after execution

    Raises:
        ValidationError: If input validation fails
    """
    from uuid import uuid4

    # Generate thread ID if not provided
    if thread_id is None:
        thread_id = f"{document_id}-{uuid4().hex[:8]}"

    # Validate inputs
    try:
        document_id = validate_pipeline_state({"document_id": document_id})["document_id"]
        timeout_seconds = validate_pipeline_state({"timeout_seconds": timeout_seconds})[
            "timeout_seconds"
        ]
    except ValidationError as e:
        log.error(f"Input validation failed: {e}")
        raise

    # Initialize checkpointer
    checkpointer = get_checkpointer()
    if checkpointer is None:
        init_checkpointer()
        checkpointer = get_checkpointer()

    # Try to resume from checkpoint
    if resume:
        log.info(f"Attempting to resume from checkpoint: {thread_id}")
        final_state = checkpointer.load(thread_id)
        if final_state:
            log.info(f"Resumed from checkpoint: {thread_id}")
            return final_state
        log.warning(f"No checkpoint found, starting fresh: {thread_id}")

    # Create initial state
    started_at = time.time()
    initial_state: PipelineState = {
        "current_state": State.INIT.value,
        "proposed_next": State.FETCH.value,
        "retry_count": 0,
        "error_message": None,
        "error_type": None,
        "audit_trail": ["init"],
        "fallback_depth": 0,
        "document_id": document_id,
        "raw_data": None,
        "validated_data": None,
        "enriched_data": None,
        "started_at": started_at,
        "timeout_seconds": timeout_seconds,
    }

    # Build and run graph
    graph = build_graph()

    try:
        final_state = graph.invoke(initial_state)

        # Save checkpoint on success
        if checkpointer:
            checkpointer.save(thread_id, "final", final_state)
            log.info(f"Checkpoint saved: {thread_id}")

        return final_state
    except Exception as e:
        log.error(f"Pipeline execution failed: {e}", exc_info=True)
        # Return error state
        error_state: PipelineState = {
            **initial_state,
            "current_state": State.ERROR.value,
            "error_message": f"Pipeline error: {str(e)}",
            "error_type": type(e).__name__,
        }
        if checkpointer:
            checkpointer.save(thread_id, "error", error_state)
        return error_state

    # Print audit trail
    print("\n─── Audit Trail ───────────────────────────────────────────")
    for i, entry in enumerate(final_state["audit_trail"], 1):
        print(f"  {i:>2}. {entry}")
    print(f"\n  Final state : {final_state['current_state'].upper()}")
    if final_state.get("error_message"):
        print(f"  Error       : {final_state['error_message']}")
    print("────────────────────────────────────────────────────────────\n")

    return final_state


def run_pipeline_with_checkpoint(document_id: str) -> PipelineState:
    """Run pipeline with ability to resume from checkpoints.

    This is a template for stateful execution using LangGraph persistence.
    In production, would integrate with LangGraph's built-in checkpointing.

    Args:
        document_id: ID of document to process

    Returns:
        Final PipelineState after execution
    """
    # For now, just run normally. In production:
    # - Use graph.with_config(configurable={...}) for checkpointing
    # - Store intermediate states in persistent storage
    # - Allow resumption from any node
    return run_pipeline(document_id)


if __name__ == "__main__":
    import random

    random.seed(42)
    run_pipeline("DOC-20240619-001")
