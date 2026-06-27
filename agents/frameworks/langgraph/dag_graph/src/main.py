"""
main.py
────────────────────────────────────────────────────────────────────────────
Demo: run the document processing pipeline with both one-turn and multi-turn support.

ONE-TURN EXAMPLES (process()):
  DOC-001  happy path       INIT → FETCH → VALIDATE → ENRICH → STORE → COMPLETE
  DOC-002  fetch retry      FETCH fails (30%) → RETRY → FETCH → VALIDATE → COMPLETE
  DOC-003  human review     VALIDATE fails → HUMAN_REVIEW → ENRICH → STORE → COMPLETE

MULTI-TURN EXAMPLES (invoke_turn()):
  Session-001: Multi-turn conversation with pause/resume
  - Turn 1: Start processing
  - Turn 2: Provide feedback
  - Turn 3: Continue to completion

After the runs the script also demonstrates checkpoint resume: retrieve
and print audit history from a previous pipeline execution.

Run:
    python -m src.main
"""

import random
import sys
from pathlib import Path
from uuid import uuid4

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.json_checkpointer import JsonCheckpointer
from src.workflow import run_pipeline
from src.workflow.graph import DocumentPipelineGraph, build_graph

SEP = "═" * 80


def print_state_summary(result: dict, doc_id: str, session_id: str) -> None:
    """Print audit trail and final state summary."""
    print(f"\n{SEP}")
    print(f"  DOCUMENT: {doc_id}  │  SESSION: {session_id[:8]}…")
    print(SEP)

    final_state = result["current_state"].upper()
    retry_count = result.get("retry_count", 0)
    audit_trail = result.get("audit_trail", [])

    print(f"\n  Final State : {final_state}")
    print(f"  Retry Count : {retry_count}")
    print(f"  Audit Steps : {len(audit_trail)}")

    print("\n  ┌─ Audit Trail ──────────────────────────────────────────────┐")
    for i, entry in enumerate(audit_trail, 1):
        print(f"  │ {i:2d}. {entry}")
    print("  └────────────────────────────────────────────────────────────┘")

    if result.get("error_message"):
        print(f"\n  ⚠️  Error: {result['error_message']}")

    print()


def scenario_happy_path(seed: int = 99, sessions_dir: str = ".doc_sessions") -> str:
    """
    Scenario 1: Happy Path
    Expected: INIT → FETCH → VALIDATE → ENRICH → STORE → COMPLETE
    (seed=99 avoids 30% fetch failure chance)
    """
    random.seed(seed)

    session_id = str(uuid4())
    doc_id = "DOC-20240619-001"

    print(f"\n\n{'█' * 80}")
    print("█ SCENARIO 1: HAPPY PATH")
    print("█ Expected: INIT → FETCH → VALIDATE → ENRICH → STORE → COMPLETE")
    print(f"{'█' * 80}")

    result = run_pipeline(
        document_id=doc_id,
        timeout_seconds=300,
        thread_id=session_id,
        sessions_dir=sessions_dir,
    )

    print_state_summary(result, doc_id, session_id)
    return session_id


def scenario_fetch_retry(seed: int = 0, sessions_dir: str = ".doc_sessions") -> str:
    """
    Scenario 2: Fetch Retry
    Expected: FETCH fails (30% chance with seed=0) → RETRY → FETCH → VALIDATE → ...
    """
    random.seed(seed)

    session_id = str(uuid4())
    doc_id = "DOC-20240619-002"

    print(f"\n\n{'█' * 80}")
    print("█ SCENARIO 2: FETCH RETRY")
    print("█ Expected: FETCH fails → RETRY → FETCH → VALIDATE → ENRICH → STORE → COMPLETE")
    print(f"{'█' * 80}")

    result = run_pipeline(
        document_id=doc_id,
        timeout_seconds=300,
        thread_id=session_id,
        sessions_dir=sessions_dir,
    )

    print_state_summary(result, doc_id, session_id)
    return session_id


def scenario_human_review(sessions_dir: str = ".doc_sessions") -> str:
    """
    Scenario 3: Human Review Path
    Expected: VALIDATE fails (schema_version validation) → HUMAN_REVIEW → ENRICH → STORE → COMPLETE
    """
    random.seed(42)

    session_id = str(uuid4())
    doc_id = "DOC-20240619-003"

    print(f"\n\n{'█' * 80}")
    print("█ SCENARIO 3: HUMAN REVIEW PATH")
    print("█ Expected: FETCH (bad schema) → VALIDATE fails → HUMAN_REVIEW")
    print("█           → ENRICH → STORE → COMPLETE")
    print(f"{'█' * 80}")

    # Note: In this version, the bad schema is simulated by handle_validate
    # checking for missing schema_version, which routes to HUMAN_REVIEW
    result = run_pipeline(
        document_id=doc_id,
        timeout_seconds=300,
        thread_id=session_id,
        sessions_dir=sessions_dir,
    )

    print_state_summary(result, doc_id, session_id)
    return session_id


def scenario_checkpoint_resume(thread_id: str, sessions_dir: str = ".doc_sessions") -> None:
    """
    Scenario 4: Checkpoint Resume
    Demonstrate loading and printing audit history from checkpoint.
    """
    print(f"\n\n{'█' * 80}")
    print("█ SCENARIO 4: CHECKPOINT RESUME")
    print(f"█ Load execution history from checkpoint: {thread_id[:8]}…")
    print(f"{'█' * 80}")

    checkpointer = JsonCheckpointer(sessions_dir=sessions_dir)

    # Try to load the checkpoint tuple with metadata
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint_tuple = checkpointer.get_tuple(config)

    if not checkpoint_tuple:
        print(f"\n  (No checkpoint found for thread: {thread_id})")
        return

    resumed_state = checkpoint_tuple.checkpoint.get("values", {})

    if not resumed_state:
        print(f"\n  (Checkpoint found but state is empty: {thread_id})")
        return

    print(f"\n  ✓ Checkpoint loaded for thread: {thread_id[:8]}…\n")

    final_state = resumed_state.get("current_state", "unknown").upper()
    doc_id = resumed_state.get("document_id", "unknown")
    audit_trail = resumed_state.get("audit_trail", [])
    retry_count = resumed_state.get("retry_count", 0)

    print(f"  Document ID  : {doc_id}")
    print(f"  Final State  : {final_state}")
    print(f"  Retry Count  : {retry_count}")
    print(f"  Audit Steps  : {len(audit_trail)}")

    print("\n  ┌─ Audit Trail from Checkpoint ──────────────────────────────┐")
    for i, entry in enumerate(audit_trail, 1):
        print(f"  │ {i:2d}. {entry}")
    print("  └────────────────────────────────────────────────────────────┘\n")


def scenario_multi_turn_example(sessions_dir: str = ".doc_sessions") -> None:
    """
    Scenario 5: Multi-turn conversation with pause/resume at upload_documents

    This scenario demonstrates the key multi-turn workflow feature:
    - Automatic pause at blocking states (waits_for_input=True)
    - Automatic checkpoint save/resume between turns
    - Hidden from user API (same invoke_turn call for both turns)

    Expected flow:
    - Turn 1: INIT → FETCH → UPLOAD_DOCUMENTS (PAUSE and wait for documents)
    - Turn 2: Resume from pause point → handler processes documents → continues to completion
    """
    import json
    from unittest.mock import patch

    print(f"\n\n{'█' * 80}")
    print("█ SCENARIO 5: MULTI-TURN CONVERSATION WITH PAUSE/RESUME")
    print("█ Feature: Automatic checkpoint management at blocking states")
    print("█ User Experience: Same invoke_turn() call for all turns")
    print(f"█ Expected: INIT → FETCH → UPLOAD_DOCUMENTS (pause)")
    print(f"█           UPLOAD_DOCUMENTS → VALIDATE → ENRICH → STORE → COMPLETE")
    print(f"{'█' * 80}")

    # Initialize graph with checkpointer
    graph = build_graph(sessions_dir=sessions_dir)

    session_id = str(uuid4())
    user_id = "user-demo"

    # ──────────────────────────────────────────────────────────────
    # TURN 1: Start processing, pause at upload_documents
    # ──────────────────────────────────────────────────────────────
    print(f"\n  ┌─ TURN 1: Start document processing ────────────────────────┐")

    # Mock random to ensure fetch succeeds (avoid 30% random failure)
    with patch("workflow.handlers.random.random", return_value=0.9):
        response_1 = graph.invoke_turn(
            user_id=user_id,
            session_id=session_id,
            turn_input="Please process document",
            timeout_sec=10.0,
        )

    print(f"  │ Turn Number     : {response_1.get('turn_number')}")
    print(f"  │ Current State   : {response_1.get('current_state').upper()}")
    print(f"  │ Waits for Input : {response_1.get('waits_for_input')}")
    print(f"  │ Status          : {'✓ Paused at upload_documents' if response_1.get('waits_for_input') else '✗ Not paused'}")
    print(f"  │ Note            : Checkpoint automatically saved for resumption")
    print(f"  └──────────────────────────────────────────────────────────┘")

    # ──────────────────────────────────────────────────────────────
    # TURN 2: Upload documents and continue to completion
    # ──────────────────────────────────────────────────────────────
    print(f"\n  ┌─ TURN 2: Upload supporting documents and continue ───────┐")

    supporting_docs = [
        {"name": "attachment1.pdf", "content": "Supporting document 1"},
        {"name": "attachment2.pdf", "content": "Supporting document 2"},
    ]

    with patch("workflow.handlers.random.random", return_value=0.9):
        response_2 = graph.invoke_turn(
            user_id=user_id,
            session_id=session_id,
            turn_input=json.dumps(supporting_docs),
            timeout_sec=10.0,
        )

    print(f"  │ Turn Number     : {response_2.get('turn_number')}")
    print(f"  │ Current State   : {response_2.get('current_state').upper()}")
    print(f"  │ Waits for Input : {response_2.get('waits_for_input')}")
    print(f"  │ Status          : {'✓ Complete' if response_2.get('current_state') == 'complete' else '✗ In progress'}")
    print(f"  │ Note            : Resumed from checkpoint, documents processed, flow continued")
    print(f"  └──────────────────────────────────────────────────────────┘")

    # Print conversation history from Turn 2
    print(f"\n  Conversation History (Turn 2 summary):")
    history = response_2.get("conversation_history", [])
    if history:
        # Show last 2 entries (user input and assistant response from Turn 2)
        for i, entry in enumerate(history[-2:], 1):
            role = entry.get("role", "?").upper()
            content = entry.get("content", "")[:60]
            turn = entry.get("turn_number", "?")
            print(f"    {i}. [Turn {turn} - {role}] {content}...")
    else:
        print(f"    (No conversation history)")

    print()

def main() -> None:
    """Run all scenarios."""
    print(f"\n\n{'▓' * 80}")
    print("▓ LANGGRAPH STATE MACHINE DEMO - Document Processing Pipeline")
    print("▓ One-turn + Multi-turn scenarios with checkpoint resume")
    print(f"{'▓' * 80}")

    # Use .doc_sessions directory for checkpointing (matches Agno pattern)
    sessions_dir = ".doc_sessions"

    # # Scenario 1: Happy path (seed avoids fetch failure)
    # session_1 = scenario_happy_path(seed=99, sessions_dir=sessions_dir)

    # # Scenario 2: Fetch retry (seed triggers fetch failure)
    # session_2 = scenario_fetch_retry(seed=0, sessions_dir=sessions_dir)

    # # Scenario 3: Human review (bad schema triggers review path)
    # session_3 = scenario_human_review(sessions_dir=sessions_dir)

    # # Scenario 4: Resume from checkpoint
    # scenario_checkpoint_resume(session_3, sessions_dir=sessions_dir)

    # Scenario 5: Multi-turn conversation
    scenario_multi_turn_example(sessions_dir=sessions_dir)


if __name__ == "__main__":
    main()
