"""
lib/session.py
────────────────────────────────────────────────────────────────────────────
Session-state helpers shared by all Agno workflow implementations.

Provides:
  • init_control_state_defaults() — initialize multi-turn control fields
  • append_turn()         — add a turn to session_state["conversation_history"]
  • build_history_prompt()  — format last N turns as XML for LLM injection
  • get_agno_history()      — pull Agno's built-in run-summary history
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from agno.workflow.types import StepInput


# functions
# ── Multi-turn initialization ────────────────────────────────────────────────
def init_control_state_defaults(session_state: dict[str, Any]) -> None:
    """
    Initialize control plane fields for multi-turn workflows.

    Sets defaults for turn management, semantic context, state machine control.
    Called from _init_session_defaults() in subclass.

    Args:
        session_state: The Agno session_state dict (modified in-place)
    """
    defaults = {
        "turn_input": None,
        "turn_number": 0,
        "conversation_history": [],
        "semantic_context": {"entities": {}, "intents": []},
        "conversation_id": "",
        "max_history_turns": 10,
        "router_timeout_sec": 10.0,
        "current_state": "init",
        "proposed_next": "init",
        "retry_count": 0,
        "error_message": None,
        "guardrail_ok": True,
        "audit_trail": [],
    }
    for k, v in defaults.items():
        session_state.setdefault(k, v)


# ── Conversation history helpers ──────────────────────────────────────────────
def append_turn(
    session_state: dict[str, Any],
    role:          str,
    content:       str,
    intent:        Optional[str] = None,
) -> None:
    """
    Append a turn to session_state["conversation_history"] in-place.

    session_state is the Agno-persisted dict, so this is automatically
    checkpointed to the db backend after the run completes.
    """
    turn = {
        "role":      role,
        "content":   content,
        "intent":    intent,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    session_state.setdefault("conversation_history", []).append(turn)
    session_state["turn_count"] = session_state.get("turn_count", 0) + 1


def build_history_prompt(
    session_state: dict[str, Any],
    max_turns:     int = 8,
) -> str:
    """
    Render the last N conversation turns as a <history> XML block.

    Returns an empty string when there are no turns yet.
    """
    turns = session_state.get("conversation_history", [])[-max_turns:]
    if not turns:
        return ""
    lines = []
    for t in turns:
        intent_attr = f' intent="{t["intent"]}"' if t.get("intent") else ""
        lines.append(f'  <turn role="{t["role"]}"{intent_attr}>{t["content"]}</turn>')
    return "<history>\n" + "\n".join(lines) + "\n</history>"


def get_agno_history(step_input: StepInput, max_runs: int = 6) -> str:
    """
    Return Agno's built-in workflow run history as a formatted string.

    This is separate from `build_history_prompt` — Agno history captures
    full run-level summaries while turns track raw message text.
    Returns empty string when no history is available.
    """
    return step_input.get_workflow_history_context(num_runs=max_runs) or ""