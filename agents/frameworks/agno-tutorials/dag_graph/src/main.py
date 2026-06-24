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
        turn_input="I want to fetch this document",
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
    # ── Multi-turn workflow test ──────────────────────────────────────────────
    test_multiturn_workflow()


if __name__ == "__main__":
    main()
