"""Handler functions for document processing pipeline.

Each handler executes business logic for a state and must:
  1. Read from state dict
  2. Process the data
  3. Return ONLY the fields it's updating (a partial delta)

current_state and session_status are stamped centrally by EngineGraph._dispatch_handler —
handlers never set them. audit_trail is reducer-backed (operator.add), so
handlers return only the new entry/entries, not the accumulated list.
"""

from __future__ import annotations

import logging
import random

from src.engine.handler_registry import handler

from .session_state import SessionState
from .state_transitions import State

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

@handler(state="fetch", waits_for_input=False, description="Fetch document from source")
def handle_fetch(state: SessionState) -> SessionState:
    """Fetch document by document_id and populate raw_data.

    Args:
        state: SessionState with document_id set

    Returns:
        Delta with raw_data populated or error info set
    """
    log.info("[HANDLER] fetch  doc_id=%s", state.get("document_id", "unknown"))

    # Simulate occasional fetch failures (30% of the time on first attempt)
    if random.random() < 0.30 and state["retry_count"] == 0:
        log.warning("[HANDLER] fetch failed – will retry")
        return {
            "raw_data": None,
            "audit_trail": ["fetch FAILED"],
        }

    raw = {
        "id": state["document_id"],
        "content": "Lorem ipsum dolor sit amet",
        "schema_version": "2.1",
    }
    return {
        "raw_data": raw,
        "audit_trail": [f"fetch OK  payload_id={raw['id']}"],
    }


@handler(
    state="upload_documents",
    waits_for_input=True,
    description="Wait for user to upload supporting documents",
)
def handle_upload_documents(state: SessionState) -> SessionState:
    """Wait for user to upload supporting documents.

    Args:
        state: SessionState with supporting_docs already merged in (via
            invoke()'s state_delta) for this turn

    Returns:
        Delta with an audit_trail entry reporting how many documents arrived
    """
    supporting_docs = state.get("supporting_docs") or []
    log.info(
        "[HANDLER] upload_documents  doc_id=%s  supporting_docs=%s",
        state.get("document_id", "unknown"),
        len(supporting_docs),
    )

    return {
        "audit_trail": [f"upload_documents OK – {len(supporting_docs)} documents uploaded"],
    }


@handler(
    state="validate",
    waits_for_input=False,
    description="Validate document schema and content",
)
def handle_validate(state: SessionState) -> SessionState:
    """Validate schema of raw_data and populate validated_data using VALIDATE_CHAIN.

    Args:
        state: SessionState with raw_data set

    Returns:
        Delta with validated_data or error info
    """
    log.info("[HANDLER] validate")
    from src.engine.chains import chain_field

    from .chains import validate_chain

    raw = state.get("raw_data") or {}

    try:
        # Invoke the validation chain
        result = validate_chain.invoke({"input": str(raw)})

        is_valid = chain_field(result, "is_valid", False)
        sanitized = chain_field(result, "sanitized_data", {})
        issues = chain_field(result, "issues", [])

        if is_valid:
            validated = {**sanitized, "_validated": True}
            msg = f"validate OK – {'; '.join(issues) if issues else 'no issues'}"
            return {
                "validated_data": validated,
                "audit_trail": [msg],
            }
        else:
            log.warning("[HANDLER] validation failed – %s", "; ".join(issues))
            return {
                "validated_data": None,
                "audit_trail": [f"validate FAILED – {'; '.join(issues)}"],
            }
    except Exception as e:
        log.error("[HANDLER] validation chain error: %s", str(e))
        return {
            "validated_data": None,
            "audit_trail": [f"validate ERROR – {str(e)}"],
        }


@handler(state="enrich", waits_for_input=False, description="Add metadata and tags to document")
def handle_enrich(state: SessionState) -> SessionState:
    """Add metadata and tags to validated_data using ENRICH_CHAIN.

    Args:
        state: SessionState with validated_data set

    Returns:
        Delta with enriched_data
    """
    log.info("[HANDLER] enrich")
    from src.engine.chains import chain_field

    from .chains import enrich_chain

    base = state.get("validated_data") or state.get("raw_data") or {}

    try:
        # Invoke the enrichment chain
        result = enrich_chain.invoke({"input": str(base)})

        tags = chain_field(result, "tags", [])
        summary = chain_field(result, "summary", "")
        word_count = chain_field(result, "word_count", 0)
        language = chain_field(result, "language", "en")
        metadata = chain_field(result, "metadata", {})

        enriched = {
            **base,
            "tags": tags,
            "summary": summary,
            "word_count": word_count,
            "language": language,
            "metadata": metadata,
        }
        return {
            "enriched_data": enriched,
            "audit_trail": [f"enrich OK – tags={', '.join(tags)}"],
        }
    except Exception as e:
        log.error("[HANDLER] enrichment chain error: %s", str(e))
        # Fallback to simple enrichment
        enriched = {**base, "tags": ["unknown"], "word_count": len(str(base))}
        return {
            "enriched_data": enriched,
            "audit_trail": [f"enrich FALLBACK – {str(e)}"],
        }


@handler(state="store", waits_for_input=False, description="Persist document to storage")
def handle_store(state: SessionState) -> SessionState:
    """Persist enriched_data to database.

    Args:
        state: SessionState with enriched_data set

    Returns:
        Delta with an audit_trail entry recording the store
    """
    log.info("[HANDLER] store")
    # Simulate write to database
    enriched = state.get("enriched_data")
    record_id = enriched.get("id", "unknown") if enriched else "unknown"
    return {
        "audit_trail": [f"store OK  record_id={record_id}"],
    }


@handler(state="complete", waits_for_input=False, description="Mark pipeline as complete")
def handle_complete(state: SessionState) -> SessionState:
    """Mark pipeline as complete.

    Args:
        state: SessionState

    Returns:
        Delta with an audit_trail entry marking completion
    """
    log.info("[HANDLER] ✅  pipeline complete for doc_id=%s", state["document_id"])
    return {
        "audit_trail": ["COMPLETE"],
    }


@handler(state="retry", waits_for_input=False, description="Retry last operation")
def handle_retry(state: SessionState) -> SessionState:
    """Increment retry counter and clear stale data.

    Args:
        state: SessionState with retry_count

    Returns:
        Delta with incremented retry_count
    """
    new_count = state["retry_count"] + 1
    log.info("[HANDLER] retry  attempt=%d", new_count)
    return {
        "retry_count": new_count,
        "raw_data": None,  # clear stale payload
        "audit_trail": [f"retry #{new_count}"],
    }


@handler(state="human_review", waits_for_input=True, description="Wait for human expert review")
def handle_human_review(state: SessionState) -> SessionState:
    """Route document to human review using REVIEW_CHAIN.

    Args:
        state: SessionState

    Returns:
        Delta with human review result
    """
    log.warning("[HANDLER] 🔍  document routed to HUMAN_REVIEW  doc_id=%s", state["document_id"])
    from src.engine.chains import chain_field

    from .chains import review_chain

    raw = state.get("raw_data") or {}

    try:
        # Invoke the review chain
        result = review_chain.invoke({"input": str(raw)})

        approved = chain_field(result, "approved", False)
        fixed_data = chain_field(result, "fixed_data", {})
        reviewer_note = chain_field(result, "reviewer_note", "")

        if approved:
            approved_data = {
                **(fixed_data or raw),
                "_human_approved": True,
                "_validated": True,
            }
            msg = f"human_review: APPROVED – {reviewer_note[:50]}"
            return {
                "validated_data": approved_data,
                "audit_trail": [msg],
            }
        else:
            log.warning("[HANDLER] human_review REJECTED: %s", reviewer_note)
            msg = f"human_review: REJECTED – {reviewer_note[:50]}"
            return {
                "validated_data": None,
                "audit_trail": [msg],
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
            "validated_data": approved_data,
            "audit_trail": [f"human_review: FALLBACK approved – {str(e)[:30]}"],
        }


@handler(state="error", waits_for_input=False, description="Handle pipeline error")
def handle_error(state: SessionState) -> SessionState:
    """Handle pipeline error state.

    Args:
        state: SessionState with error_message set

    Returns:
        Delta with an audit_trail entry recording the error
    """
    log.error(
        "[HANDLER] 🔴  pipeline ERROR  doc_id=%s  reason=%s",
        state["document_id"],
        state.get("error_message", "unknown"),
    )
    return {
        "audit_trail": [f"ERROR: {state.get('error_message', 'unknown')}"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# HANDLER MAP (exported for use in state machine graph)
# ─────────────────────────────────────────────────────────────────────────────

handler_map = {
    State.FETCH: handle_fetch,
    State.UPLOAD_DOCUMENTS: handle_upload_documents,
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
    "handle_upload_documents",
    "handle_validate",
    "handle_enrich",
    "handle_store",
    "handle_complete",
    "handle_retry",
    "handle_human_review",
    "handle_error",
    "handler_map",
]
