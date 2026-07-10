"""Multi-turn workflow test demonstrating pause/resume at blocking states.

This test mirrors the Agno multi-turn workflow pattern:
- Turn 1: Auto-progress INIT → FETCH → VALIDATE → HUMAN_REVIEW (PAUSE)
- Turn 2: Continue from HUMAN_REVIEW → ENRICH → STORE → COMPLETE

The workflow pauses at HUMAN_REVIEW because @handler(waits_for_input=True),
allowing the user to provide feedback before continuing.
"""

from uuid import uuid4

# Import handlers FIRST to populate the metadata registry via @handler decorators
from src.docprocessing import handlers  # noqa: F401
from src.docprocessing.graph import build_graph
from src.docprocessing.state_transitions import State
from src.engine.handler_registry import does_state_wait_for_input


def test_multiturn_workflow_pause_at_upload_documents() -> None:
    """Test multi-turn workflow with pause/resume at UPLOAD_DOCUMENTS.

    Demonstrates the auto-progression feature with document upload:
    - Turn 1: INIT → FETCH → UPLOAD_DOCUMENTS (PAUSE)
    - Turn 2: Continue from UPLOAD_DOCUMENTS with uploaded docs
      → VALIDATE → ENRICH → STORE → COMPLETE

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
    graph = build_graph()

    # ── TURN 1: Auto-progress INIT → FETCH → UPLOAD_DOCUMENTS ──
    print(f"\n▶ TURN 1: Starting workflow for {doc_id}")
    print("  Expected: INIT → FETCH → UPLOAD_DOCUMENTS (STOP)")

    # Invoke turn with mocked fetch to ensure it succeeds (mock skips random failure)
    with patch("src.docprocessing.handlers.random.random", return_value=0.9):
        response_1 = graph.invoke(
            user_id=user_id,
            session_id=session_id,
            input_message="Please process this document for me",
            timeout_sec=10.0,
        )

    print("\n✅ Turn 1 Complete:")
    print(f"  Current State: {response_1.get('current_state')}")
    print(f"  Waits for Input: {does_state_wait_for_input(response_1.get('current_state'))}")
    print(f"  Turn Number: {response_1.get('turn_number')}")
    confidence = response_1.get("router_confidence")
    if confidence is not None:
        print(f"  Router Confidence: {confidence:.2f}")

    # Verify Turn 1 results
    assert response_1.get("current_state") == "upload_documents", (
        f"Expected to pause at upload_documents, "
        f"but got {response_1.get('current_state')}"
    )
    assert does_state_wait_for_input(response_1.get("current_state")) is True, (
        "Expected waits_for_input=True at upload_documents"
    )
    assert response_1.get("turn_number") == 1, (
        f"Expected turn_number=1, got {response_1.get('turn_number')}"
    )

    # Verify that UPLOAD_DOCUMENTS is indeed a blocking state
    assert does_state_wait_for_input("upload_documents"), (
        "UPLOAD_DOCUMENTS should have waits_for_input=True"
    )

    print("\n📋 Message history after Turn 1:")
    history = response_1.get("messages", [])
    for i, msg in enumerate(history, 1):
        content = str(msg.content)[:50]
        print(f"    {i}. [{msg.type.upper()}] {content}...")

    # ── TURN 2: Upload supporting documents and continue ────────────────────
    print("\n▶ TURN 2: User uploads supporting documents")
    print("  Expected: UPLOAD_DOCUMENTS → VALIDATE → ENRICH → STORE → COMPLETE")

    # Upload supporting documents with metadata
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
    with patch("src.docprocessing.handlers.random.random", return_value=0.9):  # Skip fetch failure
        response_2 = graph.invoke(
            user_id=user_id,
            session_id=session_id,
            input_message="Here are the supporting documents",
            state_delta={"supporting_docs": supporting_docs},
            timeout_sec=10.0,
        )

    print("\n✅ Turn 2 Complete:")
    print(f"  Current State: {response_2.get('current_state')}")
    print(f"  Waits for Input: {does_state_wait_for_input(response_2.get('current_state'))}")
    print(f"  Turn Number: {response_2.get('turn_number')}")

    # Verify Turn 2 results
    assert response_2.get("current_state") == "complete", (
        f"Expected to complete, got {response_2.get('current_state')}"
    )
    assert does_state_wait_for_input(response_2.get("current_state")) is False, (
        "Expected waits_for_input=False at complete"
    )
    assert response_2.get("turn_number") == 2, (
        f"Expected turn_number=2, got {response_2.get('turn_number')}"
    )

    # Verify message history accumulated for this turn
    history_2 = response_2.get("messages", [])
    assert len(history_2) >= 2, (
        f"Expected at least 2 messages in history, got {len(history_2)}"
    )

    print("\n📋 Message history (Turn 2):")
    for i, msg in enumerate(history_2, 1):
        content = str(msg.content)[:60]
        state = msg.additional_kwargs.get("state", "?")
        print(f"    {i:2d}. [{msg.type.upper()}] {content}... (state: {state})")

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

    print("\n🎉 Multi-turn workflow test PASSED!")
    print(
        "   Successfully paused at HUMAN_REVIEW "
        "and resumed in next turn.\n"
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
    graph = build_graph()

    session_id = str(uuid4())

    print("\n▶ Single turn with auto-progression")
    print("  Input: Process this document")

    with patch("src.docprocessing.handlers.random.random", return_value=0.9):  # Skip fetch failure
        response = graph.invoke(
            user_id="user-456",
            session_id=session_id,
            input_message="Process this document",
        )

    print("\n✅ Single turn result:")
    print(f"  Starting state: {State.INIT.value}")
    print(f"  Ending state: {response.get('current_state')}")
    print(f"  Waits for input: {does_state_wait_for_input(response.get('current_state'))}")

    # Verify it progressed through non-blocking states
    final_state = response.get("current_state")
    assert final_state == "upload_documents", (
        f"Expected to auto-progress to upload_documents (first blocking state), "
        f"got {final_state}"
    )

    # Verify it paused at the blocking state
    assert does_state_wait_for_input(final_state) is True, (
        "Should pause at blocking state (upload_documents)"
    )

    print("\n✅ Auto-progression verified:")
    print("   INIT → FETCH → UPLOAD_DOCUMENTS (blocked)")


def test_turn_semantics() -> None:
    """Test that turn metadata and conversation tracking work correctly."""
    from unittest.mock import patch

    graph = build_graph()

    session_id = str(uuid4())

    # Turn 1 - mock fetch to ensure success
    with patch("src.docprocessing.handlers.random.random", return_value=0.9):
        response_1 = graph.invoke(
            user_id="user-789",
            session_id=session_id,
            input_message="Start workflow",
        )

    assert response_1.get("turn_number") == 1
    assert response_1.get("current_state") == "upload_documents"
    assert len(response_1.get("messages", [])) >= 2  # User + assistant entries

    # Turn 2 - provide document data to progress from upload_documents
    docs = [{"name": "doc1.pdf", "content": "test"}]
    with patch("src.docprocessing.handlers.random.random", return_value=0.9):
        response_2 = graph.invoke(
            user_id="user-789",
            session_id=session_id,
            input_message="Here's the document",
            state_delta={"supporting_docs": docs},
        )

    assert response_2.get("turn_number") == 2
    # After uploading docs, should progress to complete
    assert response_2.get("current_state") == "complete"

    # Guard: message history must accumulate ACROSS turns, not just contain
    # this turn's own entries. Regression this guards against: messages are
    # appended in invoke() via plain Python after the graph has already
    # finished running (and checkpointing) for the turn, so without an
    # explicit persist step, the next turn's checkpoint load never sees
    # them and history silently resets to just the current turn every time.
    hist = response_2.get("messages", [])
    assert len(hist) == 4, (
        f"Expected 4 messages (2 from turn 1 + 2 from turn 2), got {len(hist)}: "
        f"{[m.content for m in hist]}"
    )
    assert hist[0].additional_kwargs.get("turn_number") == 1, (
        "First message should be turn 1's — message history did not persist across turns"
    )
    assert hist[0].content == "Start workflow", (
        "First message should be turn 1's own input, not overwritten by turn 2"
    )
    assert hist[2].additional_kwargs.get("turn_number") == 2, (
        "Third message should be turn 2's — message history did not persist across turns"
    )

    print("✅ Turn semantics test passed")


if __name__ == "__main__":
    test_multiturn_workflow_pause_at_upload_documents()
    test_multiturn_auto_progression()
    test_turn_semantics()
    print("\n🎉 All multi-turn tests passed!")
