"""LangGraph document processing workflow entrypoint.

Orchestrates the complete pipeline: Router → Guardrail → Handler → Loop/End
Includes checkpointing, timeouts, and input validation.
"""

from __future__ import annotations

import logging
import time
from typing import Optional
from uuid import uuid4

from src.engine.checkpointing import SqliteCheckpointer

from .graph import build_graph
from .pipeline_state import PipelineState
from .state_machine import State
from .validation import ValidationError, validate_pipeline_state

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def run_pipeline(
    document_id: str,
    timeout_seconds: float = 300,
    thread_id: Optional[str] = None,
    db_path: str = ":memory:",
) -> PipelineState:
    """Run the full document processing pipeline with checkpointing.

    Executes: State Machine → Router → Guardrail → Handler (loop) → End

    Features:
      • Input validation (document ID, content)
      • Timeout protection (default 5 minutes)
      • Checkpointing via SqliteCheckpointer
      • Resumable execution from checkpoints
      • Error handling with graceful degradation

    Args:
        document_id: ID of document to process
        timeout_seconds: Maximum pipeline execution time (default 300s)
        thread_id: Unique thread ID for checkpointing (auto-generated if None)
        db_path: Path to SQLite checkpoint database (default ":memory:")

    Returns:
        Final PipelineState after execution

    Raises:
        ValidationError: If input validation fails
    """
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

    # Create checkpointer
    checkpointer = SqliteCheckpointer(db_path)

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

    # Build and compile graph with checkpointer
    compiled_graph = build_graph(checkpointer=checkpointer)

    try:
        # Invoke graph with thread_id config
        # LangGraph will automatically save intermediate checkpoints via the checkpointer
        final_state = compiled_graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )

        # Save final state as a complete snapshot for easy retrieval
        checkpointer.save(thread_id, "final", final_state)

        log.info(f"Pipeline completed: {thread_id}")
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
        return error_state


if __name__ == "__main__":
    import random

    random.seed(42)
    run_pipeline("DOC-20240619-001")
