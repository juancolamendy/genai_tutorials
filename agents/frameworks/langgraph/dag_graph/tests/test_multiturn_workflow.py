"""Multi-turn workflow test demonstrating pause/resume at blocking states.

This test mirrors the Agno multi-turn workflow pattern:
- Turn 1: Auto-progress INIT → FETCH → VALIDATE → HUMAN_REVIEW (PAUSE)
- Turn 2: Continue from HUMAN_REVIEW → ENRICH → STORE → COMPLETE

The workflow pauses at HUMAN_REVIEW because @handler(waits_for_input=True),
allowing the user to provide feedback before continuing.
"""

import sys
from pathlib import Path
from uuid import uuid4

# Ensure src is in path for imports
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import handlers FIRST to populate the metadata registry via @handler decorators
from workflow import handlers  # noqa: F401
from engine.handler_registry import does_state_wait_for_input
from workflow.graph import DocumentPipelineGraph, build_graph
from workflow.pipeline_state import new_pipeline
from workflow.state_machine import State


def test_multiturn_workflow_pause_at_upload_documents() -> None:
    """Test multi-turn workflow with pause/resume at UPLOAD_DOCUMENTS.

    Demonstrates the auto-progression feature with document upload:
    - Turn 1: INIT → FETCH → UPLOAD_DOCUMENTS (PAUSE)
    - Turn 2: Continue from UPLOAD_DOCUMENTS with uploaded docs → VALIDATE → ENRICH → STORE → COMPLETE

    This shows how workflows can pause at specific states (waits_for_input=True)
    and resume with user-provided context in the next turn.
    """
    from unittest.mock import patch

    sep = "═" * 80

    print(f"\n{sep}")
    print("  MULTI-TURN WORKFLOW TEST")
    print("  Auto-progress: FETCH → PAUSE at UPLOAD_DOCUMENTS")
    print(sep)

    # Setup
    session_id = str(uuid4())
    doc_id = "MULTITURN-DOC-001"
    user_id = "user-123"

    # Build graph with default configuration
    compiled_graph = build_graph()
    graph = DocumentPipelineGraph()
    graph.compiled_graph = compiled_graph

    # ── TURN 1: Auto-progress INIT → FETCH → UPLOAD_DOCUMENTS ──
    print(f"\n▶ TURN 1: Starting workflow for {doc_id}")
    print(f"  Expected: INIT → FETCH → UPLOAD_DOCUMENTS (STOP)")

    # Create initial state
    state_1 = new_pipeline(doc_id)

    # Invoke turn with mocked fetch to ensure it succeeds
    with patch("workflow.handlers.random.random", return_value=0.9):  # Mock random to skip failure
        response_1 = graph.invoke_turn(
            user_id=user_id,
            session_id=session_id,
            turn_input="Please process this document for me",
            timeout_sec=10.0,
        )

    print(f"\n✅ Turn 1 Complete:")
    print(f"  Current State: {response_1.get('current_state')}")
    print(f"  Waits for Input: {response_1.get('waits_for_input')}")
    print(f"  Turn Number: {response_1.get('turn_number')}")
    confidence = response_1.get("router_confidence")
    if confidence is not None:
        print(f"  Router Confidence: {confidence:.2f}")

    # Verify Turn 1 results
    assert response_1.get("current_state") == "upload_documents", (
        f"Expected to pause at upload_documents, "
        f"but got {response_1.get('current_state')}"
    )
    assert response_1.get("waits_for_input") is True, (
        "Expected waits_for_input=True at upload_documents"
    )
    assert response_1.get("turn_number") == 1, (
        f"Expected turn_number=1, got {response_1.get('turn_number')}"
    )

    # Verify that UPLOAD_DOCUMENTS is indeed a blocking state
    assert does_state_wait_for_input("upload_documents"), (
        "UPLOAD_DOCUMENTS should have waits_for_input=True"
    )

    print("\n📋 Conversation history after Turn 1:")
    history = response_1.get("conversation_history", [])
    for i, turn in enumerate(history, 1):
        role = turn.get("role", "?").upper()
        content = turn.get("content", "")[:50]
        print(f"    {i}. [{role}] {content}...")

    # ── TURN 2: Upload supporting documents and continue ────────────────────
    print(f"\n▶ TURN 2: User uploads supporting documents")
    print(f"  Expected: UPLOAD_DOCUMENTS → VALIDATE → ENRICH → STORE → COMPLETE")

    # Upload supporting documents with metadata
    import json

    supporting_docs = [
        {
            "name": "attachment1.pdf",
            "content": "Supporting document 1 content",
            "type": "reference",
        },
        {
            "name": "attachment2.pdf",
            "content": "Supporting document 2 content",
            "type": "reference",
        },
    ]

    # Continue with uploaded documents and mocked handlers
    with patch("workflow.handlers.random.random", return_value=0.9):  # Skip fetch failure
        response_2 = graph.invoke_turn(
            user_id=user_id,
            session_id=session_id,
            turn_input=json.dumps(supporting_docs),
            timeout_sec=10.0,
        )

    print(f"\n✅ Turn 2 Complete:")
    print(f"  Current State: {response_2.get('current_state')}")
    print(f"  Waits for Input: {response_2.get('waits_for_input')}")
    print(f"  Turn Number: {response_2.get('turn_number')}")

    # Verify Turn 2 results
    assert response_2.get("current_state") == "complete", (
        f"Expected to complete, got {response_2.get('current_state')}"
    )
    assert response_2.get("waits_for_input") is False, (
        "Expected waits_for_input=False at complete"
    )
    assert response_2.get("turn_number") == 2, (
        f"Expected turn_number=2, got {response_2.get('turn_number')}"
    )

    # Verify conversation history accumulated across both turns
    history_2 = response_2.get("conversation_history", [])
    assert len(history_2) >= 2, (
        f"Expected at least 2 turns in history, got {len(history_2)}"
    )

    print("\n📋 Complete conversation history:")
    for i, turn in enumerate(history_2, 1):
        role = turn.get("role", "?").upper()
        content = turn.get("content", "")[:60]
        state = turn.get("state", "?")
        print(f"    {i:2d}. [{role}] {content}... (state: {state})")

    # Verify semantic context was captured
    semantic_context = response_2.get("semantic_context", {})
    if semantic_context:
        print("\n🧠 Semantic Context Extracted:")
        entities = semantic_context.get("entities", {})
        intents = semantic_context.get("intents", [])
        if entities:
            print(f"    Entities: {entities}")
        if intents:
            print(f"    Intents: {intents}")

    print(f"\n🎉 Multi-turn workflow test PASSED!")
    print(
        f"   Successfully paused at HUMAN_REVIEW "
        f"and resumed in next turn.\n"
    )


def test_multiturn_auto_progression() -> None:
    """Test that non-blocking states auto-progress within a single turn.

    Verifies that a turn can progress through multiple non-blocking states
    (INIT → FETCH) and only pause at the first blocking state (UPLOAD_DOCUMENTS).
    """
    from unittest.mock import patch

    sep = "═" * 80

    print(f"\n{sep}")
    print("  AUTO-PROGRESSION TEST")
    print("  Verify non-blocking states auto-progress in single turn")
    print(sep)

    # Build graph
    compiled_graph = build_graph()
    graph = DocumentPipelineGraph()
    graph.compiled_graph = compiled_graph

    session_id = str(uuid4())
    doc_id = "AUTO-PROGRESS-001"

    print(f"\n▶ Single turn with auto-progression")
    print(f"  Input: Process this document")

    with patch("workflow.handlers.random.random", return_value=0.9):  # Skip fetch failure
        response = graph.invoke_turn(
            user_id="user-456",
            session_id=session_id,
            turn_input="Process this document",
        )

    print(f"\n✅ Single turn result:")
    print(f"  Starting state: {State.INIT.value}")
    print(f"  Ending state: {response.get('current_state')}")
    print(f"  Waits for input: {response.get('waits_for_input')}")

    # Verify it progressed through non-blocking states
    final_state = response.get("current_state")
    assert final_state == "upload_documents", (
        f"Expected to auto-progress to upload_documents (first blocking state), "
        f"got {final_state}"
    )

    # Verify it paused at the blocking state
    assert response.get("waits_for_input") is True, (
        "Should pause at blocking state (upload_documents)"
    )

    print(f"\n✅ Auto-progression verified:")
    print(f"   INIT → FETCH → UPLOAD_DOCUMENTS (blocked)")


def test_turn_semantics() -> None:
    """Test that turn metadata and conversation tracking work correctly."""
    from unittest.mock import patch

    graph = DocumentPipelineGraph()
    graph.compiled_graph = build_graph()

    session_id = str(uuid4())

    # Turn 1 - mock fetch to ensure success
    with patch("workflow.handlers.random.random", return_value=0.9):
        response_1 = graph.invoke_turn(
            user_id="user-789",
            session_id=session_id,
            turn_input="Start workflow",
        )

    assert response_1.get("turn_number") == 1
    assert response_1.get("current_state") == "upload_documents"
    assert len(response_1.get("conversation_history", [])) >= 2  # User + assistant entries

    # Turn 2 - provide document data to progress from upload_documents
    import json
    docs = [{"name": "doc1.pdf", "content": "test"}]
    with patch("workflow.handlers.random.random", return_value=0.9):
        response_2 = graph.invoke_turn(
            user_id="user-789",
            session_id=session_id,
            turn_input=json.dumps(docs),
        )

    assert response_2.get("turn_number") == 2
    # After uploading docs, should progress to complete
    assert response_2.get("current_state") == "complete"
    assert len(response_2.get("conversation_history", [])) >= 2  # At least user + assistant for this turn

    # Verify turn numbers are properly set in history
    hist = response_2.get("conversation_history", [])
    if len(hist) > 0:
        assert hist[0].get("turn_number") in [1, 2], "Turn numbers should be set"

    print("✅ Turn semantics test passed")


if __name__ == "__main__":
    test_multiturn_workflow_pause_at_human_review()
    test_multiturn_auto_progression()
    test_turn_semantics()
    print("\n🎉 All multi-turn tests passed!")
