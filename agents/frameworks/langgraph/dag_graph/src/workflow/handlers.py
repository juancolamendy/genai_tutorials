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
from typing import Any

from .state_machine import State, PipelineState

log = logging.getLogger(__name__)


def _audit(state: PipelineState, msg: str) -> list[str]:
    """Helper: append audit message to trail."""
    return state["audit_trail"] + [msg]


# ─────────────────────────────────────────────────────────────────────────────
# HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

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


def handle_validate(state: PipelineState) -> PipelineState:
    """Validate schema of raw_data and populate validated_data.

    Args:
        state: PipelineState with raw_data set

    Returns:
        Updated state with validated_data or error info
    """
    log.info("[HANDLER] validate")
    raw = state["raw_data"] or {}

    # Require schema_version field
    if "schema_version" not in raw:
        log.warning("[HANDLER] validation failed – schema_version missing")
        return {
            **state,
            "current_state": State.VALIDATE.value,
            "validated_data": None,
            "audit_trail": _audit(state, "validate FAILED – schema_version missing"),
        }

    validated = {**raw, "_validated": True}
    return {
        **state,
        "current_state": State.VALIDATE.value,
        "validated_data": validated,
        "audit_trail": _audit(state, "validate OK"),
    }


def handle_enrich(state: PipelineState) -> PipelineState:
    """Add metadata and tags to validated_data and populate enriched_data.

    Args:
        state: PipelineState with validated_data set

    Returns:
        Updated state with enriched_data
    """
    log.info("[HANDLER] enrich")
    base = state.get("validated_data") or state.get("raw_data") or {}
    enriched = {**base, "tags": ["finance", "q3"], "word_count": 42}
    return {
        **state,
        "current_state": State.ENRICH.value,
        "enriched_data": enriched,
        "audit_trail": _audit(state, "enrich OK"),
    }


def handle_store(state: PipelineState) -> PipelineState:
    """Persist enriched_data to database.

    Args:
        state: PipelineState with enriched_data set

    Returns:
        Updated state after storage
    """
    log.info("[HANDLER] store")
    # Simulate write to database
    record_id = state["enriched_data"].get("id", "unknown") if state.get("enriched_data") else "unknown"
    return {
        **state,
        "current_state": State.STORE.value,
        "audit_trail": _audit(state, f"store OK  record_id={record_id}"),
    }


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


def handle_human_review(state: PipelineState) -> PipelineState:
    """Route document to human review.

    In production: push to review queue / Slack / ticketing system.
    Here: auto-approve for demo purposes.

    Args:
        state: PipelineState

    Returns:
        Updated state with human review result
    """
    log.warning("[HANDLER] 🔍  document routed to HUMAN_REVIEW  doc_id=%s", state["document_id"])

    # Auto-approve for demo
    approved_data = {
        **(state["raw_data"] or {}),
        "_human_approved": True,
        "_validated": True,
    }
    return {
        **state,
        "current_state": State.HUMAN_REVIEW.value,
        "validated_data": approved_data,
        "audit_trail": _audit(state, "human_review: auto-approved for demo"),
    }


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
