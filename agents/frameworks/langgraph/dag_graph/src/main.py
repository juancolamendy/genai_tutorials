"""
main.py
────────────────────────────────────────────────────────────────────────────
Demo: run the document processing pipeline's multi-turn support.

MULTI-TURN EXAMPLE (invoke_turn()):
  Multi-turn conversation with pause/resume at a blocking state.
  - Turn 1: Start processing, pause at upload_documents
  - Turn 2: Upload documents, continue to completion

Run:
    python -m src.main
"""

import sys
from pathlib import Path
from uuid import uuid4

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.docprocessing.graph import build_graph


def scenario_multi_turn_example(sessions_dir: str = ".doc_sessions") -> None:
    """
    Multi-turn conversation with pause/resume at upload_documents

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
    print("█ Expected: INIT → FETCH → UPLOAD_DOCUMENTS (pause)")
    print("█           UPLOAD_DOCUMENTS → VALIDATE → ENRICH → STORE → COMPLETE")
    print(f"{'█' * 80}")

    # Initialize graph with checkpointer
    graph = build_graph(sessions_dir=sessions_dir)

    session_id = str(uuid4())
    user_id = "user-demo"

    # ──────────────────────────────────────────────────────────────
    # TURN 1: Start processing, pause at upload_documents
    # ──────────────────────────────────────────────────────────────
    print("\n  ┌─ TURN 1: Start document processing ────────────────────────┐")

    # Mock random to ensure fetch succeeds (avoid 30% random failure)
    with patch("src.docprocessing.handlers.random.random", return_value=0.9):
        response_1 = graph.invoke_turn(
            user_id=user_id,
            session_id=session_id,
            turn_input="Please process document",
            timeout_sec=10.0,
        )

    print(f"  │ Turn Number     : {response_1.get('turn_number')}")
    print(f"  │ Current State   : {response_1.get('current_state').upper()}")
    print(f"  │ Waits for Input : {response_1.get('waits_for_input')}")
    paused = response_1.get("waits_for_input")
    print(f"  │ Status          : {'✓ Paused at upload_documents' if paused else '✗ Not paused'}")
    print("  │ Note            : Checkpoint automatically saved for resumption")
    print("  └──────────────────────────────────────────────────────────┘")

    # ──────────────────────────────────────────────────────────────
    # TURN 2: Upload documents and continue to completion
    # ──────────────────────────────────────────────────────────────
    print("\n  ┌─ TURN 2: Upload supporting documents and continue ───────┐")

    supporting_docs = [
        {"name": "attachment1.pdf", "content": "Supporting document 1"},
        {"name": "attachment2.pdf", "content": "Supporting document 2"},
    ]

    with patch("src.docprocessing.handlers.random.random", return_value=0.9):
        response_2 = graph.invoke_turn(
            user_id=user_id,
            session_id=session_id,
            turn_input=json.dumps(supporting_docs),
            timeout_sec=10.0,
        )

    print(f"  │ Turn Number     : {response_2.get('turn_number')}")
    print(f"  │ Current State   : {response_2.get('current_state').upper()}")
    print(f"  │ Waits for Input : {response_2.get('waits_for_input')}")
    completed = response_2.get("current_state") == "complete"
    print(f"  │ Status          : {'✓ Complete' if completed else '✗ In progress'}")
    print("  │ Note            : Resumed from checkpoint, documents processed, flow continued")
    print("  └──────────────────────────────────────────────────────────┘")

    # Print conversation history from Turn 2
    print("\n  Conversation History (Turn 2 summary):")
    history = response_2.get("conversation_history", [])
    if history:
        # Show last 2 entries (user input and assistant response from Turn 2)
        for i, entry in enumerate(history[-2:], 1):
            role = entry.get("role", "?").upper()
            content = entry.get("content", "")[:60]
            turn = entry.get("turn_number", "?")
            print(f"    {i}. [Turn {turn} - {role}] {content}...")
    else:
        print("    (No conversation history)")

    print()

def main() -> None:
    """Run the multi-turn demo scenario."""
    print(f"\n\n{'▓' * 80}")
    print("▓ LANGGRAPH STATE MACHINE DEMO - Document Processing Pipeline")
    print("▓ Multi-turn conversation with pause/resume")
    print(f"{'▓' * 80}")

    # Use .doc_sessions directory for checkpointing (matches Agno pattern)
    sessions_dir = ".doc_sessions"

    scenario_multi_turn_example(sessions_dir=sessions_dir)


if __name__ == "__main__":
    main()
