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


def test_multiturn_workflow_pause_at_human_review() -> None:
    """Test multi-turn workflow with pause/resume at HUMAN_REVIEW.

    Demonstrates the auto-progression feature:
    - Turn 1: INIT → FETCH → VALIDATE → HUMAN_REVIEW (PAUSE)
    - Turn 2: Continue from HUMAN_REVIEW → ENRICH → STORE → COMPLETE

    This shows how workflows can pause at specific states (waits_for_input=True)
    and resume with user-provided context in the next turn.
    """
    from unittest.mock import patch

    sep = "═" * 80

    print(f"\n{sep}")
    print("  MULTI-TURN WORKFLOW TEST")
    print("  Auto-progress: FETCH → VALIDATE → PAUSE at HUMAN_REVIEW")
    print(sep)


    # Setup
    session_id = str(uuid4())
    doc_id = "MULTITURN-DOC-001"
    user_id = "user-123"

    # Build graph with default configuration
    compiled_graph = build_graph()
    graph = DocumentPipelineGraph()
    graph.compiled_graph = compiled_graph

    # ── TURN 1: Auto-progress INIT → FETCH → VALIDATE → HUMAN_REVIEW ──
    print(f"\n▶ TURN 1: Starting workflow for {doc_id}")
    print(f"  Expected: INIT → FETCH → VALIDATE → HUMAN_REVIEW (STOP)")

    # Mock validation to fail so validated_data is empty,
    # triggering HUMAN_REVIEW from ENRICH guardrail
    mock_validate = lambda x: {"is_valid": False, "sanitized_data": {}, "issues": ["schema mismatch"]}

    # Create initial state
    state_1 = new_pipeline(doc_id)

    # Invoke turn with mocked validation
    with patch("workflow.chains.VALIDATE_CHAIN") as mock_chain:
        mock_chain.invoke.return_value = mock_validate({})
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
    assert response_1.get("current_state") == "human_review", (
        f"Expected to pause at human_review, "
        f"but got {response_1.get('current_state')}"
    )
    assert response_1.get("waits_for_input") is True, (
        "Expected waits_for_input=True at human_review"
    )
    assert response_1.get("turn_number") == 1, (
        f"Expected turn_number=1, got {response_1.get('turn_number')}"
    )

    # Verify that HUMAN_REVIEW is indeed a blocking state
    assert does_state_wait_for_input("human_review"), (
        "HUMAN_REVIEW should have waits_for_input=True"
    )

    print("\n📋 Conversation history after Turn 1:")
    history = response_1.get("conversation_history", [])
    for i, turn in enumerate(history, 1):
        role = turn.get("role", "?").upper()
        content = turn.get("content", "")[:50]
        print(f"    {i}. [{role}] {content}...")

    # ── TURN 2: Provide feedback and continue ──────────────────────────────
    print(f"\n▶ TURN 2: User provides feedback and approval")
    print(f"  Expected: HUMAN_REVIEW → ENRICH → STORE → COMPLETE")

    # Continue with user feedback (approved by reviewer)
    response_2 = graph.invoke_turn(
        user_id=user_id,
        session_id=session_id,
        turn_input="Document looks correct, approved for enrichment",
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
    (INIT → FETCH → VALIDATE) and only pause at a blocking state (HUMAN_REVIEW).
    """
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
    assert final_state == "human_review", (
        f"Expected to auto-progress to human_review (first blocking state), "
        f"got {final_state}"
    )

    # Verify it paused at the blocking state
    assert response.get("waits_for_input") is True, (
        "Should pause at blocking state (human_review)"
    )

    print(f"\n✅ Auto-progression verified:")
    print(f"   INIT → FETCH → VALIDATE → HUMAN_REVIEW (blocked)")


def test_turn_semantics() -> None:
    """Test that turn metadata and conversation tracking work correctly."""
    graph = DocumentPipelineGraph()
    graph.compiled_graph = build_graph()

    session_id = str(uuid4())

    # Turn 1
    response_1 = graph.invoke_turn(
        user_id="user-789",
        session_id=session_id,
        turn_input="Start workflow",
    )

    assert response_1.get("turn_number") == 1
    assert len(response_1.get("conversation_history", [])) >= 1

    # Turn 2
    response_2 = graph.invoke_turn(
        user_id="user-789",
        session_id=session_id,
        turn_input="Continue workflow",
    )

    assert response_2.get("turn_number") == 2
    assert len(response_2.get("conversation_history", [])) > len(
        response_1.get("conversation_history", [])
    )

    print("✅ Turn semantics test passed")


if __name__ == "__main__":
    test_multiturn_workflow_pause_at_human_review()
    test_multiturn_auto_progression()
    test_turn_semantics()
    print("\n🎉 All multi-turn tests passed!")
