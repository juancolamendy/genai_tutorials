"""
main.py
────────────────────────────────────────────────────────────────────────────
Demo: run the document processing pipeline on three different documents,
exercising distinct state machine paths:

  DOC-001  happy path       INIT → FETCH → VALIDATE → ENRICH → STORE → COMPLETE
  DOC-002  fetch retry      FETCH fails once → RETRY → FETCH → … → COMPLETE
  DOC-003  human review     VALIDATE fails → HUMAN_REVIEW → ENRICH → STORE → COMPLETE

After the three runs the script also demonstrates session resume: a new
DocPipelineWorkflow is created with the same session_id as DOC-001 and the
stored pipeline_runs audit history is printed without re-running the pipeline.

Install:
    uv add agno python-dotenv

Run:
    uv run main.py
"""

import random
import uuid
from dotenv import load_dotenv

from workflow import build_doc_pipeline, handlers
from workflow.state_machine import State
from workflow.pipeline_state import audit

load_dotenv()

SEP = "═" * 64


def run_doc(document_id: str, *, seed: int | None = None) -> None:
    """Run a single document through the pipeline and print the audit trail."""
    if seed is not None:
        random.seed(seed)

    session_id = str(uuid.uuid4())
    print(f"\n{SEP}")
    print(f"  PIPELINE  {document_id}   session={session_id[:8]}…")
    print(SEP)

    wf = build_doc_pipeline(session_id)
    wf.process(document_id)


def demo_session_resume(session_id: str) -> None:
    """Show that pipeline_runs history persists across process restarts."""
    print(f"\n{SEP}")
    print(f"  RESUME SESSION  {session_id[:8]}…")
    print(SEP)

    resumed = build_doc_pipeline(session_id)
    if resumed.session_state is None:
        print("  (no session state found)")
        return
    runs = resumed.session_state.get("pipeline_runs", [])
    if not runs:
        print("  (no completed runs found in session)")
        return

    for i, run in enumerate(runs, 1):
        print(f"\n  Run #{i}:")
        print(f"    document_id : {run['document_id']}")
        print(f"    final_state : {run['final_state'].upper()}")
        print(f"    retry_count : {run['retry_count']}")
        print(f"    audit steps : {len(run['audit_trail'])}")


def test_multiturn_workflow() -> None:
    """
    Test multi-turn workflow with pause/resume at UPLOAD_SUPPORT_DOCS.

    Demonstrates the auto-progression feature:
    - Turn 1: INIT → FETCH → VALIDATE → UPLOAD_SUPPORT_DOCS (PAUSE)
    - Turn 2: Continue from UPLOAD_SUPPORT_DOCS → ENRICH → STORE → COMPLETE

    This shows how workflows can pause at specific states (waits_for_input=True)
    and resume with user-provided context in the next turn.
    """
    print(f"\n{SEP}")
    print("  MULTI-TURN WORKFLOW TEST")
    print(f"  Auto-progress: FETCH → VALIDATE → PAUSE at UPLOAD_SUPPORT_DOCS")
    print(SEP)

    session_id = str(uuid.uuid4())
    doc_id = "MULTITURN-DOC-001"

    wf = build_doc_pipeline(session_id)

    # ── TURN 1: Auto-progress INIT → FETCH → VALIDATE → UPLOAD_SUPPORT_DOCS ──
    print(f"\n▶ TURN 1: Starting workflow for {doc_id}")
    print(f"  Expected: INIT → FETCH → VALIDATE → UPLOAD_SUPPORT_DOCS (STOP)")

    response_1 = wf.process_turn(
        user_id="user_123",
        session_id=session_id,
        turn_input="start_processing",
    )

    print(f"\n✅ Turn 1 Complete:")
    print(f"  Current State: {response_1.get('current_state')}")
    print(f"  Waits for Input: {response_1.get('waits_for_input')}")
    print(f"  Turn Number: {response_1.get('turn_number')}")
    confidence = response_1.get('router_confidence')
    if confidence is not None:
        print(f"  Router Confidence: {confidence:.2f}")

    assert response_1.get("current_state") == "upload_support_docs", \
        f"Expected to stop at upload_support_docs, but got {response_1.get('current_state')}"
    assert response_1.get("waits_for_input") is True, \
        "Expected waits_for_input=True at upload_support_docs"

    print(f"\n📋 Audit trail after Turn 1:")
    state = wf.session_state
    for entry in state.get("audit_trail", [])[-3:]:
        print(f"    • {entry}")

    # ── TURN 2: Provide supporting docs and continue ──────────────────────────
    print(f"\n▶ TURN 2: User provides supporting documents")
    print(f"  Expected: UPLOAD_SUPPORT_DOCS → ENRICH → STORE → COMPLETE")

    # Simulate user uploading documents
    response_2 = wf.process_turn(
        user_id="user_123",
        session_id=session_id,
        turn_input='{"supporting_docs": ["report.pdf", "invoice.xlsx"]}',
    )

    print(f"\n✅ Turn 2 Complete:")
    print(f"  Current State: {response_2.get('current_state')}")
    print(f"  Waits for Input: {response_2.get('waits_for_input')}")
    print(f"  Turn Number: {response_2.get('turn_number')}")

    assert response_2.get("current_state") == "complete", \
        f"Expected to complete, but got {response_2.get('current_state')}"
    assert response_2.get("waits_for_input") is False, \
        "Expected waits_for_input=False at complete"

    print(f"\n📋 Complete audit trail:")
    for i, entry in enumerate(state.get("audit_trail", []), 1):
        print(f"    {i:2d}. {entry}")

    print(f"\n🎉 Multi-turn workflow test PASSED!")
    print(f"   Successfully paused at UPLOAD_SUPPORT_DOCS and resumed in next turn.\n")


def main() -> None:
    random.seed(42)

    # ── DOC-001: expect happy path (seed avoids fetch failure) ───────────────
    run_doc("DOC-20240619-001", seed=99)

    # ── DOC-002: force fetch failure on first attempt ─────────────────────────
    run_doc("DOC-20240619-002", seed=0)   # seed=0 → random() < 0.25

    # ── DOC-003: schema_version missing → HUMAN_REVIEW path ──────────────────
    # Patch handle_fetch temporarily to return a malformed doc.
    _original_fetch = handlers.handle_fetch
    def _bad_fetch(p):
        bad = {"id": p["document_id"], "content": "Important report text.", "schema_version": ""}
        return audit({**p, "current_state": State.FETCH.value, "raw_data": bad},
                     "fetch OK  (malformed schema_version)")
    handlers.handle_fetch = _bad_fetch
    handlers.HANDLER_MAP[State.FETCH] = _bad_fetch

    resume_session = str(uuid.uuid4())
    print(f"\n{SEP}")
    print("  PIPELINE  DOC-20240619-003  (bad schema → human_review path)")
    print(f"  session={resume_session[:8]}…")
    print(SEP)
    wf = build_doc_pipeline(resume_session)
    wf.process("DOC-20240619-003")

    handlers.handle_fetch = _original_fetch   # restore
    handlers.HANDLER_MAP[State.FETCH] = _original_fetch

    # ── Session resume (reload DOC-003 session from disk) ────────────────────
    demo_session_resume(resume_session)

    # ── Multi-turn workflow test ──────────────────────────────────────────────
    test_multiturn_workflow()


if __name__ == "__main__":
    main()
