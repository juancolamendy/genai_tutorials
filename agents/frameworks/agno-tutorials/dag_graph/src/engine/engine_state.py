"""
engine/engine_state.py
────────────────────────────────────────────────────────────────────────────
EngineState TypedDict — control plane fields for multi-turn conversations.

This module defines the shared, reusable state structure used across all
workflows. It includes fields for:
  • Turn management (turn_input, turn_number, conversation_history)
  • Semantic context (entities, intents extracted by router)
  • State machine control (current_state, proposed_next, retry_count)
  • Error tracking (error_message, guardrail_ok)
  • Audit trail (immutable append-only log)

Workflow-specific business state (document_id, raw_data, etc.) is defined
in workflow/pipeline_state.py, which inherits from EngineState.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class EngineState(TypedDict, total=False):
    """
    Control plane state for multi-turn conversation workflows.

    All fields are optional (total=False) for backward compatibility with
    one-turn workflows that ignore multi-turn fields.

    Fields:
        turn_input: User input text for this turn (validated, escaped)
        turn_number: 0-indexed turn count
        conversation_history: List of {input, output, state_from, state_to, ...} dicts
        semantic_context: {entities: dict, intents: list[str]} from router
        conversation_id: UUID for multi-turn session grouping
        max_history_turns: Keep last N turns in memory (default 10)
        current_state: Current active state
        proposed_next: Router's candidate for next state
        retry_count: Number of retries attempted
        error_message: Error text if in ERROR state
        guardrail_ok: True if last guardrail passed
        audit_trail: Append-only chronological log
        output: Output of the last turn
    """

    turn_input: Optional[str]
    turn_number: int
    conversation_history: list
    semantic_context: dict
    conversation_id: str
    max_history_turns: int
    current_state: str
    proposed_next: str
    retry_count: int
    error_message: Optional[str]
    guardrail_ok: bool
    audit_trail: list
    output: Optional[dict]


def init_engine_state() -> EngineState:
    """
    Return a fresh EngineState with sensible defaults.

    Used to initialize a new multi-turn conversation or one-turn workflow.
    """
    return EngineState(
        turn_input=None,
        turn_number=0,
        conversation_history=[],
        semantic_context={"entities": {}, "intents": []},
        conversation_id="",
        max_history_turns=10,
        current_state="init",
        proposed_next="init",
        retry_count=0,
        error_message=None,
        guardrail_ok=True,
        audit_trail=[],
        output=None,
    )


def audit(state: EngineState, entry: str) -> EngineState:
    """
    Append entry to audit_trail and return updated state.

    Uses immutable-style update: original state is unchanged.

    Args:
        state: Current EngineState
        entry: Text to append to audit trail

    Returns:
        New EngineState with entry appended to audit_trail
    """
    return {
        **state,
        "audit_trail": state.get("audit_trail", []) + [entry],
    }
