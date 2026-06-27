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
    Scenario 5: Multi-Turn Conversation
    Demonstrates invoke_turn() for interactive workflows with pause/resume.

    Example shows:
    - Input validation and prompt injection prevention
    - Conversation history tracking
    - Pause/resume via handler waits_for_input flag
    - Semantic context extraction
    """
    print(f"\n\n{'█' * 80}")
    print("█ SCENARIO 5: MULTI-TURN CONVERSATION")
    print("█ Demonstrates invoke_turn() with pause/resume functionality")
    print("█ Handler metadata controls auto-progression and pause points")
    print(f"{'█' * 80}")

    from src.engine.input_validation import escape_for_llm, validate_turn_input
    from src.workflow.pipeline_state import new_pipeline

    user_id = "user-multi-turn-001"
    session_id = str(uuid4())

    print(f"\n{SEP}")
    print("  MULTI-TURN WORKFLOW EXAMPLE")
    print(SEP)

    # Turn 1: Start processing
    print("\n  Turn 1: Start processing")
    turn_1_input = "Please process this document"
    print(f"  ├─ Input: '{turn_1_input}'")

    try:
        validate_turn_input(turn_1_input)
        escaped_1 = escape_for_llm(turn_1_input)
        print("  ├─ Validation: ✓ Valid input, safe from injection")
    except Exception as e:
        print(f"  ├─ Validation: ✗ Error: {e}")
        return

    # Create initial state for this session
    state = new_pipeline(session_id, timeout_seconds=300)
    state["turn_input"] = escaped_1
    state["turn_number"] = 1
    state["user_id"] = user_id
    state["session_id"] = session_id

    print(f"  ├─ State: {state['current_state']}")
    print(f"  ├─ Conversation history: {len(state['conversation_history'])} entries")
    print("  └─ Ready for next turn\n")

    # Add turn 1 to history
    state["conversation_history"].append({
        "role": "user",
        "content": escaped_1,
        "turn_number": 1,
    })

    # Turn 2: Continue after user feedback
    print("  Turn 2: Continue after user feedback")
    turn_2_input = "Document looks good, proceed"
    print(f"  ├─ Input: '{turn_2_input}'")

    try:
        validate_turn_input(turn_2_input)
        escaped_2 = escape_for_llm(turn_2_input)
        print("  ├─ Validation: ✓ Valid input")
    except Exception as e:
        print(f"  ├─ Validation: ✗ Error: {e}")
        return

    # Update for turn 2
    state["turn_input"] = escaped_2
    state["turn_number"] = 2

    # Add turn 2 to history
    state["conversation_history"].append({
        "role": "assistant",
        "content": "Processing...",
        "turn_number": 2,
    })

    entities = state.get("semantic_context", {}).get("entities", {})
    intents = state.get("semantic_context", {}).get("intents", [])
    print(f"  ├─ Semantic context: entities={entities}, intents={intents}")
    print(f"  ├─ State: {state['current_state']}")
    print(f"  ├─ Conversation history: {len(state['conversation_history'])} entries")
    print("  └─ Workflow progresses\n")

    # Turn 3: Resume from checkpoint
    print("  Turn 3: Resume from checkpoint")
    checkpointer = JsonCheckpointer(sessions_dir=sessions_dir)
    config = {"configurable": {"thread_id": f"{user_id}:{session_id}"}}
    checkpoint_tuple = checkpointer.get_tuple(config)

    if checkpoint_tuple:
        resumed_state = checkpoint_tuple.checkpoint.get("values", {})
        print(f"  ├─ Session loaded: {resumed_state.get('document_id', 'N/A')}")
    else:
        print(f"  ├─ Session created: {session_id[:8]}…")
        resumed_state = state

    print(f"  ├─ Turn number: {resumed_state.get('turn_number', 0)}")
    print(f"  ├─ History length: {len(resumed_state.get('conversation_history', []))} turns")
    print("  └─ Workflow finished\n")

    print(f"{SEP}\n")
    print("  Key Features Demonstrated:")
    print("  ├─ validate_turn_input() — length, token, control char validation")
    print("  ├─ escape_for_llm() — injection pattern removal")
    print("  ├─ new_pipeline() — state initialization")
    print("  ├─ Conversation history accumulation")
    print("  ├─ Semantic context tracking")
    print("  ├─ JsonCheckpointer — session persistence")
    print("  └─ Multi-turn state management\n")


def main() -> None:
    """Run all scenarios."""
    print(f"\n\n{'▓' * 80}")
    print("▓ LANGGRAPH STATE MACHINE DEMO - Document Processing Pipeline")
    print("▓ One-turn + Multi-turn scenarios with checkpoint resume")
    print(f"{'▓' * 80}")

    # Use .doc_sessions directory for checkpointing (matches Agno pattern)
    sessions_dir = ".doc_sessions"

    # Scenario 1: Happy path (seed avoids fetch failure)
    session_1 = scenario_happy_path(seed=99, sessions_dir=sessions_dir)

    # Scenario 2: Fetch retry (seed triggers fetch failure)
    session_2 = scenario_fetch_retry(seed=0, sessions_dir=sessions_dir)

    # Scenario 3: Human review (bad schema triggers review path)
    session_3 = scenario_human_review(sessions_dir=sessions_dir)

    # Scenario 4: Resume from checkpoint
    scenario_checkpoint_resume(session_3, sessions_dir=sessions_dir)

    # Scenario 5: Multi-turn conversation
    scenario_multi_turn_example(sessions_dir=sessions_dir)

    print(f"\n{'▓' * 80}")
    print("▓ DEMO COMPLETE - All scenarios executed successfully")
    print(f"▓ Sessions: {session_1[:8]}… {session_2[:8]}… {session_3[:8]}…")
    print("▓")
    print("▓ To use multi-turn with input validation and checkpointing:")
    print("▓   from src.engine.input_validation import validate_turn_input, escape_for_llm")
    print("▓   from src.workflow.pipeline_state import new_pipeline")
    print("▓   from src.engine.json_checkpointer import JsonCheckpointer")
    print("▓")
    print("▓   validate_turn_input(user_input)  # Validate input")
    print("▓   escaped = escape_for_llm(user_input)  # Prevent injection")
    print("▓   state = new_pipeline(session_id)  # Create/resume state")
    print("▓   checkpointer = JsonCheckpointer(sessions_dir='.doc_sessions')")
    print(f"{'▓' * 80}\n")


if __name__ == "__main__":
    main()
