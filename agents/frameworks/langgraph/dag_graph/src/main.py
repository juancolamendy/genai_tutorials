"""
main.py
────────────────────────────────────────────────────────────────────────────
Demo: run the document processing pipeline on three different documents,
exercising distinct state machine paths:

  DOC-001  happy path       INIT → FETCH → VALIDATE → ENRICH → STORE → COMPLETE
  DOC-002  fetch retry      FETCH fails (30%) → RETRY → FETCH → VALIDATE → COMPLETE
  DOC-003  human review     VALIDATE fails → HUMAN_REVIEW → ENRICH → STORE → COMPLETE

After the three runs the script also demonstrates checkpoint resume: retrieve
and print audit history from a previous pipeline execution.

Run:
    python -m src.main
"""

import json
import random
import sys
from pathlib import Path
from uuid import uuid4

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.engine.checkpointing import SqliteCheckpointer
from src.workflow import run_pipeline

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


def scenario_happy_path(seed: int = 99, db_path: str = ":memory:") -> str:
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
        db_path=db_path,
    )

    print_state_summary(result, doc_id, session_id)
    return session_id


def scenario_fetch_retry(seed: int = 0, db_path: str = ":memory:") -> str:
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
        db_path=db_path,
    )

    print_state_summary(result, doc_id, session_id)
    return session_id


def scenario_human_review(db_path: str = ":memory:") -> str:
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
        db_path=db_path,
    )

    print_state_summary(result, doc_id, session_id)
    return session_id


def scenario_checkpoint_resume(thread_id: str, db_path: str = ":memory:") -> None:
    """
    Scenario 4: Checkpoint Resume
    Demonstrate loading and printing audit history from checkpoint.
    """
    print(f"\n\n{'█' * 80}")
    print("█ SCENARIO 4: CHECKPOINT RESUME")
    print(f"█ Load execution history from checkpoint: {thread_id[:8]}…")
    print(f"{'█' * 80}")

    checkpointer = SqliteCheckpointer(db_path)

    # Try to load the checkpoint tuple with metadata
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint_tuple = checkpointer.get_tuple(config)

    if not checkpoint_tuple:
        print(f"\n  (No checkpoint found for thread: {thread_id})")
        return

    config_out, checkpoint, metadata = checkpoint_tuple
    resumed_state = checkpoint.get("values", {})

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


def main() -> None:
    """Run all scenarios."""
    print(f"\n\n{'▓' * 80}")
    print("▓ LANGGRAPH STATE MACHINE DEMO - Document Processing Pipeline")
    print("▓ Three scenarios exercising different state paths + checkpoint resume")
    print(f"{'▓' * 80}")

    # Use in-memory checkpointing for demo (checkpoints lost on exit)
    # For production, use: db_path = str(Path.home() / ".cache" / "langgraph.db")
    db_path = ":memory:"

    # Scenario 1: Happy path (seed avoids fetch failure)
    session_1 = scenario_happy_path(seed=99, db_path=db_path)

    # Scenario 2: Fetch retry (seed triggers fetch failure)
    session_2 = scenario_fetch_retry(seed=0, db_path=db_path)

    # Scenario 3: Human review (bad schema triggers review path)
    session_3 = scenario_human_review(db_path=db_path)

    # Scenario 4: Resume from checkpoint
    scenario_checkpoint_resume(session_3, db_path=db_path)

    print(f"\n{'▓' * 80}")
    print("▓ DEMO COMPLETE - All scenarios executed successfully")
    print(f"▓ Sessions: {session_1[:8]}… {session_2[:8]}… {session_3[:8]}…")
    print(f"{'▓' * 80}\n")


if __name__ == "__main__":
    main()
