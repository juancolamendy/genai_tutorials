"""Handler functions for document processing workflow.

Handler Contract:
- Input: PipelineState dict
- Output: Updated PipelineState dict
- MUST set current_state to own state value
- MUST catch ALL exceptions (not just specific types)
- SHOULD set error_type = "transient" or "permanent" on exception
- MUST log exceptions with exc_info=True for debugging
- MUST NOT raise exceptions to caller
"""

import logging
from src.workflow.state_machine import State, PipelineState

log = logging.getLogger(__name__)


def handle_fetch_template(state: PipelineState) -> PipelineState:
    """Template handler for FETCH state.

    Contract: Fetch document by document_id, populate raw_data.
    This is a template implementation for testing handler architecture.

    Args:
        state: PipelineState with document_id set

    Returns:
        Updated PipelineState with current_state set, raw_data populated or
        error_message/error_type set on exception
    """
    try:
        # Simulate document fetch (in real implementation, would call external API)
        raw_data = {"content": f"Document for {state['document_id']}"}

        return {
            **state,
            "current_state": State.FETCH.value,
            "raw_data": raw_data,
            "error_type": None,
            "error_message": None,
        }
    except TimeoutError as e:
        log.warning(
            "[FETCH] transient error (will retry): %s",
            e,
            exc_info=True,
        )
        return {
            **state,
            "current_state": State.FETCH.value,
            "error_message": str(e),
            "error_type": "transient",
            "raw_data": None,
        }
    except Exception as e:
        log.error("[FETCH] unexpected exception: %s", e, exc_info=True)
        return {
            **state,
            "current_state": State.FETCH.value,
            "error_message": f"Unexpected error: {type(e).__name__}",
            "error_type": "permanent",
            "raw_data": None,
        }


def handle_validate_template(state: PipelineState) -> PipelineState:
    """Template handler for VALIDATE state.

    Contract: Validate schema of raw_data, populate validated_data.

    Args:
        state: PipelineState with raw_data set

    Returns:
        Updated PipelineState with validated_data or error info
    """
    try:
        # Simulate schema validation
        validated_data = {
            "schema_version": "1.0",
            "content": state.get("raw_data", {}).get("content", ""),
        }

        return {
            **state,
            "current_state": State.VALIDATE.value,
            "validated_data": validated_data,
            "error_type": None,
            "error_message": None,
        }
    except Exception as e:
        log.error("[VALIDATE] unexpected exception: %s", e, exc_info=True)
        return {
            **state,
            "current_state": State.VALIDATE.value,
            "error_message": f"Unexpected error: {type(e).__name__}",
            "error_type": "permanent",
            "validated_data": None,
        }


def handle_enrich_template(state: PipelineState) -> PipelineState:
    """Template handler for ENRICH state.

    Contract: Add metadata and tags to validated_data, populate enriched_data.

    Args:
        state: PipelineState with validated_data set

    Returns:
        Updated PipelineState with enriched_data or error info
    """
    try:
        # Simulate enrichment (add tags and metadata)
        enriched_data = {
            **state.get("validated_data", {}),
            "tags": ["processed"],
            "metadata": {
                "document_id": state["document_id"],
                "version": "1.0",
            },
        }

        return {
            **state,
            "current_state": State.ENRICH.value,
            "enriched_data": enriched_data,
            "error_type": None,
            "error_message": None,
        }
    except Exception as e:
        log.error("[ENRICH] unexpected exception: %s", e, exc_info=True)
        return {
            **state,
            "current_state": State.ENRICH.value,
            "error_message": f"Unexpected error: {type(e).__name__}",
            "error_type": "permanent",
            "enriched_data": None,
        }


def handle_store_template(state: PipelineState) -> PipelineState:
    """Template handler for STORE state.

    Contract: Persist enriched_data to database.

    Args:
        state: PipelineState with enriched_data set

    Returns:
        Updated PipelineState after persistence
    """
    try:
        # Simulate database write (in real implementation, would call database API)
        # For testing, just simulate success

        return {
            **state,
            "current_state": State.STORE.value,
            "error_type": None,
            "error_message": None,
        }
    except Exception as e:
        log.error("[STORE] unexpected exception: %s", e, exc_info=True)
        return {
            **state,
            "current_state": State.STORE.value,
            "error_message": f"Unexpected error: {type(e).__name__}",
            "error_type": "permanent",
        }


def handle_retry_template(state: PipelineState) -> PipelineState:
    """Template handler for RETRY state.

    Contract: Increment retry_count, clear stale raw_data, loop back to router.

    Args:
        state: PipelineState with error_type set

    Returns:
        Updated PipelineState with retry_count incremented and raw_data cleared
    """
    try:
        new_retry_count = state["retry_count"] + 1
        assert new_retry_count == state["retry_count"] + 1, "Retry count increment failed"

        return {
            **state,
            "current_state": State.RETRY.value,
            "retry_count": new_retry_count,
            "raw_data": None,  # Clear stale data
            "error_type": None,
            "error_message": None,
        }
    except Exception as e:
        log.error("[RETRY] unexpected exception: %s", e, exc_info=True)
        return {
            **state,
            "current_state": State.RETRY.value,
            "error_message": f"Unexpected error: {type(e).__name__}",
            "error_type": "permanent",
        }


def handle_human_review_template(state: PipelineState) -> PipelineState:
    """Template handler for HUMAN_REVIEW state.

    Contract: Route to human reviewer, simulate auto-approval for demo.

    Args:
        state: PipelineState requiring manual review

    Returns:
        Updated PipelineState with manual approval (simulated)
    """
    try:
        # Simulate human review approval (auto-approve for demo)
        return {
            **state,
            "current_state": State.HUMAN_REVIEW.value,
            "error_type": None,
            "error_message": None,
        }
    except Exception as e:
        log.error("[HUMAN_REVIEW] unexpected exception: %s", e, exc_info=True)
        return {
            **state,
            "current_state": State.HUMAN_REVIEW.value,
            "error_message": f"Unexpected error: {type(e).__name__}",
            "error_type": "permanent",
        }


def handle_complete_template(state: PipelineState) -> PipelineState:
    """Template handler for COMPLETE state (terminal).

    Contract: No-op handler, just mark complete.

    Args:
        state: PipelineState ready for completion

    Returns:
        Updated PipelineState with COMPLETE as current_state
    """
    try:
        return {
            **state,
            "current_state": State.COMPLETE.value,
            "error_type": None,
            "error_message": None,
        }
    except Exception as e:
        log.error("[COMPLETE] unexpected exception: %s", e, exc_info=True)
        return {
            **state,
            "current_state": State.COMPLETE.value,
            "error_message": f"Unexpected error: {type(e).__name__}",
            "error_type": "permanent",
        }


def handle_error_template(state: PipelineState) -> PipelineState:
    """Template handler for ERROR state (terminal).

    Contract: No-op handler, just mark error.

    Args:
        state: PipelineState in error state

    Returns:
        Updated PipelineState with ERROR as current_state
    """
    try:
        return {
            **state,
            "current_state": State.ERROR.value,
        }
    except Exception as e:
        log.error("[ERROR] unexpected exception: %s", e, exc_info=True)
        return {
            **state,
            "current_state": State.ERROR.value,
            "error_message": f"Unexpected error: {type(e).__name__}",
        }
