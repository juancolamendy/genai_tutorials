"""
doc_pipeline/handlers.py
────────────────────────────────────────────────────────────────────────────
One handler function per pipeline state.

Handler contract
────────────────
  • Signature:  (pipeline: PipelineState) -> PipelineState
  • Each handler MUST set "current_state" to its own State value on return.
  • Handlers that use LLM agents call them synchronously and embed the result
    in the returned pipeline state.
  • Handlers MUST NOT raise — they catch exceptions and move to ERROR.

HANDLER_MAP (exported)
──────────────────────
  Maps each State to its handler function.
  The workflow dispatcher reads proposed_next from session_state, looks up
  the handler in HANDLER_MAP, and calls it.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from .agents import ENRICH_AGENT, REVIEW_AGENT, VALIDATE_AGENT

from .pipeline_state import PipelineState, audit
from .state_machine import State


# variables
log = logging.getLogger(__name__)


# functions
# ── FETCH ─────────────────────────────────────────────────────────────────────
def handle_fetch(p: PipelineState) -> PipelineState:
    """Simulate fetching a document from an external source."""
    doc_id = p["document_id"]
    log.info("[FETCH]   doc_id=%s  attempt=%d", doc_id, p["retry_count"] + 1)

    # Simulate a 25 % transient fetch failure on the first attempt only.
    if random.random() < 0.25 and p["retry_count"] == 0:
        log.warning("[FETCH]   transient failure — will retry")
        return audit({**p, "current_state": State.FETCH.value, "raw_data": None},
                     "fetch FAILED (simulated transient error)")

    raw = {
        "id":             doc_id,
        "content":        f"Full text of document {doc_id}. Lorem ipsum dolor sit amet.",
        "schema_version": "2.1",
        "source":         "document-store-v2",
    }
    return audit({**p, "current_state": State.FETCH.value, "raw_data": raw},
                 f"fetch OK  schema_version={raw['schema_version']}")


# ── VALIDATE ──────────────────────────────────────────────────────────────────
def handle_validate(p: PipelineState) -> PipelineState:
    """LLM validates raw_data; sets validated_data on success."""
    log.info("[VALIDATE] doc_id=%s", p["document_id"])
    try:
        raw_json = str(p["raw_data"])
        result   = VALIDATE_AGENT.run(f"<raw_data>{raw_json}</raw_data>").content

        if result.is_valid:
            validated = {**result.sanitized_data, "_validated": True}
            return audit({**p, "current_state": State.VALIDATE.value, "validated_data": validated},
                         f"validate OK  issues=[]")

        issues_str = "; ".join(result.issues)
        log.warning("[VALIDATE] FAILED  issues=%s", issues_str)
        return audit({**p, "current_state": State.VALIDATE.value, "validated_data": None},
                     f"validate FAILED  issues={issues_str}")

    except Exception as exc:
        log.error("[VALIDATE] exception: %s", exc)
        return audit({**p, "current_state": State.VALIDATE.value,
                      "validated_data": None, "error_message": str(exc)},
                     f"validate EXCEPTION: {exc}")


# ── ENRICH ────────────────────────────────────────────────────────────────────

def handle_enrich(p: PipelineState) -> PipelineState:
    """LLM enriches validated_data with tags, summary, and metadata."""
    log.info("[ENRICH]  doc_id=%s", p["document_id"])
    base = p.get("validated_data") or p.get("raw_data") or {}

    try:
        result = ENRICH_AGENT.run(f"<validated_data>{base}</validated_data>").content
        enriched = {
            **base,
            "tags":       result.tags,
            "summary":    result.summary,
            "word_count": result.word_count,
            "language":   result.language,
            "metadata":   result.metadata,
            "_enriched":  True,
        }
        return audit({**p, "current_state": State.ENRICH.value, "enriched_data": enriched},
                     f"enrich OK  tags={result.tags}  lang={result.language}")

    except Exception as exc:
        log.error("[ENRICH] exception: %s", exc)
        return audit({**p, "current_state": State.ENRICH.value,
                      "enriched_data": None, "error_message": str(exc)},
                     f"enrich EXCEPTION: {exc}")


# ── STORE ─────────────────────────────────────────────────────────────────────
def handle_store(p: PipelineState) -> PipelineState:
    """Simulate persisting enriched_data to the document store."""
    log.info("[STORE]   doc_id=%s", p["document_id"])
    record_id = f"rec-{p['document_id']}-{id(p) & 0xFFFF:04x}"
    return audit({**p, "current_state": State.STORE.value},
                 f"store OK  record_id={record_id}")


# ── COMPLETE ──────────────────────────────────────────────────────────────────
def handle_complete(p: PipelineState) -> PipelineState:
    """Terminal success state."""
    log.info("[COMPLETE] ✅  doc_id=%s", p["document_id"])
    return audit({**p, "current_state": State.COMPLETE.value}, "COMPLETE ✅")


# ── RETRY ─────────────────────────────────────────────────────────────────────
def handle_retry(p: PipelineState) -> PipelineState:
    """Increment retry counter and clear stale payload to force a clean re-fetch."""
    new_count = p["retry_count"] + 1
    log.info("[RETRY]   attempt #%d", new_count)
    return audit({
        **p,
        "current_state": State.RETRY.value,
        "retry_count":   new_count,
        "raw_data":      None,   # clear stale data
    }, f"retry #{new_count} — clearing stale payload")


# ── HUMAN_REVIEW ──────────────────────────────────────────────────────────────
def handle_human_review(p: PipelineState) -> PipelineState:
    """
    Route a failed-validation document to a human (LLM-simulated) reviewer.

    In production: push to a review queue / Slack / ticketing system and
    pause the workflow until a human approves via HITL.  Here we use an LLM
    as a stand-in reviewer for demonstration purposes.
    """
    log.warning("[REVIEW]  🔍  doc_id=%s  routing to human review", p["document_id"])
    raw_json = str(p.get("raw_data", {}))

    try:
        decision = REVIEW_AGENT.run(f"<raw_data>{raw_json}</raw_data>").content

        if decision.approved:
            validated = {
                **decision.fixed_data,
                "_human_approved": True,
                "_validated":      True,
            }
            return audit({**p, "current_state": State.HUMAN_REVIEW.value,
                          "validated_data": validated},
                         f"human_review: APPROVED  note='{decision.reviewer_note}'")

        return audit({**p, "current_state": State.HUMAN_REVIEW.value,
                      "validated_data": None,
                      "error_message":  decision.reviewer_note},
                     f"human_review: REJECTED  note='{decision.reviewer_note}'")

    except Exception as exc:
        log.error("[REVIEW] exception: %s", exc)
        return audit({**p, "current_state": State.HUMAN_REVIEW.value,
                      "validated_data": None, "error_message": str(exc)},
                     f"human_review EXCEPTION: {exc}")


# ── ERROR ─────────────────────────────────────────────────────────────────────
def handle_error(p: PipelineState) -> PipelineState:
    """Terminal error state — log and freeze."""
    reason = p.get("error_message", "unknown error")
    log.error("[ERROR]   🔴  doc_id=%s  reason=%s", p["document_id"], reason)
    return audit({**p, "current_state": State.ERROR.value},
                 f"ERROR 🔴  reason={reason}")


# ── HANDLER_MAP ───────────────────────────────────────────────────────────────
# The single export consumed by the workflow dispatcher.
# Key   = State enum value
# Value = handler function (PipelineState → PipelineState)
HANDLER_MAP: dict[State, Any] = {
    State.FETCH:        handle_fetch,
    State.VALIDATE:     handle_validate,
    State.ENRICH:       handle_enrich,
    State.STORE:        handle_store,
    State.COMPLETE:     handle_complete,
    State.RETRY:        handle_retry,
    State.HUMAN_REVIEW: handle_human_review,
    State.ERROR:        handle_error,
}