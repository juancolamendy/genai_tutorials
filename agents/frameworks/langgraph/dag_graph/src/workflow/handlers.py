"""Handler functions for document processing workflow.

Each handler executes business logic for a state and must:
  1. Read from state dict
  2. Process the data
  3. Update state dict with results
  4. Return the updated state dict
  5. ALWAYS set current_state to its own state value
"""

from __future__ import annotations

import logging
import random

from engine.handler_registry import handler

from .pipeline_state import PipelineState
from .state_machine import State

log = logging.getLogger(__name__)


def _audit(state: PipelineState, msg: str) -> list[str]:
    """Helper: append audit message to trail."""
    return state["audit_trail"] + [msg]


# ─────────────────────────────────────────────────────────────────────────────
# HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

@handler(state="fetch", waits_for_input=False, description="Fetch document from source")
def handle_fetch(state: PipelineState) -> PipelineState:
    """Fetch document by document_id and populate raw_data.

    Args:
        state: PipelineState with document_id set

    Returns:
        Updated state with raw_data populated or error info set
    """
    log.info("[HANDLER] fetch  doc_id=%s", state["document_id"])

    # Simulate occasional fetch failures (30% of the time on first attempt)
    if random.random() < 0.30 and state["retry_count"] == 0:
        log.warning("[HANDLER] fetch failed – will retry")
        return {
            **state,
            "current_state": State.FETCH.value,
            "raw_data": None,
            "audit_trail": _audit(state, "fetch FAILED"),
        }

    raw = {
        "id": state["document_id"],
        "content": "Lorem ipsum dolor sit amet",
        "schema_version": "2.1",
    }
    return {
        **state,
        "current_state": State.FETCH.value,
        "raw_data": raw,
        "audit_trail": _audit(state, f"fetch OK  payload_id={raw['id']}"),
    }


@handler(
    state="validate",
    waits_for_input=False,
    description="Validate document schema and content",
)
def handle_validate(state: PipelineState) -> PipelineState:
    """Validate schema of raw_data and populate validated_data using VALIDATE_CHAIN.

    Args:
        state: PipelineState with raw_data set

    Returns:
        Updated state with validated_data or error info
    """
    log.info("[HANDLER] validate")
    from .chains import VALIDATE_CHAIN

    raw = state.get("raw_data") or {}

    try:
        # Invoke the validation chain
        result = VALIDATE_CHAIN.invoke({"input": str(raw)})

        # Handle dict result from JsonOutputParser
        is_valid = (
            result.get("is_valid", False)
            if isinstance(result, dict)
            else result.is_valid
        )
        sanitized = (
            result.get("sanitized_data", {})
            if isinstance(result, dict)
            else result.sanitized_data
        )
        issues = (
            result.get("issues", []) if isinstance(result, dict) else result.issues
        )

        if is_valid:
            validated = {**sanitized, "_validated": True}
            msg = f"validate OK – {'; '.join(issues) if issues else 'no issues'}"
            return {
                **state,
                "current_state": State.VALIDATE.value,
                "validated_data": validated,
                "audit_trail": _audit(state, msg),
            }
        else:
            log.warning("[HANDLER] validation failed – %s", "; ".join(issues))
            return {
                **state,
                "current_state": State.VALIDATE.value,
                "validated_data": None,
                "audit_trail": _audit(state, f"validate FAILED – {'; '.join(issues)}"),
            }
    except Exception as e:
        log.error("[HANDLER] validation chain error: %s", str(e))
        return {
            **state,
            "current_state": State.VALIDATE.value,
            "validated_data": None,
            "audit_trail": _audit(state, f"validate ERROR – {str(e)}"),
        }


@handler(state="enrich", waits_for_input=False, description="Add metadata and tags to document")
def handle_enrich(state: PipelineState) -> PipelineState:
    """Add metadata and tags to validated_data using ENRICH_CHAIN.

    Args:
        state: PipelineState with validated_data set

    Returns:
        Updated state with enriched_data
    """
    log.info("[HANDLER] enrich")
    from .chains import ENRICH_CHAIN

    base = state.get("validated_data") or state.get("raw_data") or {}

    try:
        # Invoke the enrichment chain
        result = ENRICH_CHAIN.invoke({"input": str(base)})

        # Handle dict result from JsonOutputParser
        tags = result.get("tags", []) if isinstance(result, dict) else result.tags
        summary = result.get("summary", "") if isinstance(result, dict) else result.summary
        word_count = result.get("word_count", 0) if isinstance(result, dict) else result.word_count
        language = result.get("language", "en") if isinstance(result, dict) else result.language
        metadata = result.get("metadata", {}) if isinstance(result, dict) else result.metadata

        enriched = {
            **base,
            "tags": tags,
            "summary": summary,
            "word_count": word_count,
            "language": language,
            "metadata": metadata,
        }
        return {
            **state,
            "current_state": State.ENRICH.value,
            "enriched_data": enriched,
            "audit_trail": _audit(state, f"enrich OK – tags={', '.join(tags)}"),
        }
    except Exception as e:
        log.error("[HANDLER] enrichment chain error: %s", str(e))
        # Fallback to simple enrichment
        enriched = {**base, "tags": ["unknown"], "word_count": len(str(base))}
        return {
            **state,
            "current_state": State.ENRICH.value,
            "enriched_data": enriched,
            "audit_trail": _audit(state, f"enrich FALLBACK – {str(e)}"),
        }


@handler(state="store", waits_for_input=False, description="Persist document to storage")
def handle_store(state: PipelineState) -> PipelineState:
    """Persist enriched_data to database.

    Args:
        state: PipelineState with enriched_data set

    Returns:
        Updated state after storage
    """
    log.info("[HANDLER] store")
    # Simulate write to database
    enriched = state.get("enriched_data")
    record_id = enriched.get("id", "unknown") if enriched else "unknown"
    return {
        **state,
        "current_state": State.STORE.value,
        "audit_trail": _audit(state, f"store OK  record_id={record_id}"),
    }


@handler(state="complete", waits_for_input=False, description="Mark pipeline as complete")
def handle_complete(state: PipelineState) -> PipelineState:
    """Mark pipeline as complete.

    Args:
        state: PipelineState

    Returns:
        Updated state with COMPLETE status
    """
    log.info("[HANDLER] ✅  pipeline complete for doc_id=%s", state["document_id"])
    return {
        **state,
        "current_state": State.COMPLETE.value,
        "audit_trail": _audit(state, "COMPLETE"),
    }


@handler(state="retry", waits_for_input=False, description="Retry last operation")
def handle_retry(state: PipelineState) -> PipelineState:
    """Increment retry counter and clear stale data.

    Args:
        state: PipelineState with retry_count

    Returns:
        Updated state with incremented retry_count
    """
    new_count = state["retry_count"] + 1
    log.info("[HANDLER] retry  attempt=%d", new_count)
    return {
        **state,
        "current_state": State.RETRY.value,
        "retry_count": new_count,
        "raw_data": None,  # clear stale payload
        "audit_trail": _audit(state, f"retry #{new_count}"),
    }


@handler(state="human_review", waits_for_input=True, description="Wait for human expert review")
def handle_human_review(state: PipelineState) -> PipelineState:
    """Route document to human review using REVIEW_CHAIN.

    Args:
        state: PipelineState

    Returns:
        Updated state with human review result
    """
    log.warning("[HANDLER] 🔍  document routed to HUMAN_REVIEW  doc_id=%s", state["document_id"])
    from .chains import REVIEW_CHAIN

    raw = state.get("raw_data") or {}

    try:
        # Invoke the review chain
        result = REVIEW_CHAIN.invoke({"input": str(raw)})

        # Handle dict result from JsonOutputParser
        approved = (
            result.get("approved", False)
            if isinstance(result, dict)
            else result.approved
        )
        fixed_data = (
            result.get("fixed_data", {})
            if isinstance(result, dict)
            else result.fixed_data
        )
        reviewer_note = (
            result.get("reviewer_note", "")
            if isinstance(result, dict)
            else result.reviewer_note
        )

        if approved:
            approved_data = {
                **(fixed_data or raw),
                "_human_approved": True,
                "_validated": True,
            }
            msg = f"human_review: APPROVED – {reviewer_note[:50]}"
            return {
                **state,
                "current_state": State.HUMAN_REVIEW.value,
                "validated_data": approved_data,
                "audit_trail": _audit(state, msg),
            }
        else:
            log.warning("[HANDLER] human_review REJECTED: %s", reviewer_note)
            msg = f"human_review: REJECTED – {reviewer_note[:50]}"
            return {
                **state,
                "current_state": State.HUMAN_REVIEW.value,
                "validated_data": None,
                "audit_trail": _audit(state, msg),
            }
    except Exception as e:
        log.error("[HANDLER] review chain error: %s", str(e))
        # Fallback: auto-approve
        approved_data = {
            **raw,
            "_human_approved": True,
            "_validated": True,
        }
        return {
            **state,
            "current_state": State.HUMAN_REVIEW.value,
            "validated_data": approved_data,
            "audit_trail": _audit(state, f"human_review: FALLBACK approved – {str(e)[:30]}"),
        }


@handler(state="error", waits_for_input=False, description="Handle pipeline error")
def handle_error(state: PipelineState) -> PipelineState:
    """Handle pipeline error state.

    Args:
        state: PipelineState with error_message set

    Returns:
        Updated state with ERROR status
    """
    log.error(
        "[HANDLER] 🔴  pipeline ERROR  doc_id=%s  reason=%s",
        state["document_id"],
        state.get("error_message", "unknown"),
    )
    return {
        **state,
        "current_state": State.ERROR.value,
        "audit_trail": _audit(state, f"ERROR: {state.get('error_message', 'unknown')}"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HANDLER MAP (exported for use in state machine graph)
# ─────────────────────────────────────────────────────────────────────────────

HANDLER_MAP = {
    State.FETCH: handle_fetch,
    State.VALIDATE: handle_validate,
    State.ENRICH: handle_enrich,
    State.STORE: handle_store,
    State.COMPLETE: handle_complete,
    State.RETRY: handle_retry,
    State.HUMAN_REVIEW: handle_human_review,
    State.ERROR: handle_error,
}

__all__ = [
    "handle_fetch",
    "handle_validate",
    "handle_enrich",
    "handle_store",
    "handle_complete",
    "handle_retry",
    "handle_human_review",
    "handle_error",
    "HANDLER_MAP",
]
