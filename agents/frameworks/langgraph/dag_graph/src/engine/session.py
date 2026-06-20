"""Session state helpers for managing conversation and execution context.

Provides:
  • append_turn() — add a conversation turn to session state
  • init_session_defaults() — ensure all required keys exist
  • build_history_prompt() — format conversation turns as XML for LLM injection
  • get_execution_context() — extract execution metadata
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional


def append_turn(
    session_state: dict[str, Any],
    role: str,
    content: str,
    intent: Optional[str] = None,
) -> None:
    """Append a turn to session_state["turns"] in-place.

    session_state is the LangGraph-persisted dict, so this is automatically
    checkpointed to the db backend after the run completes.

    Args:
        session_state: The session state dict
        role: Turn role (e.g., "user", "assistant", "system")
        content: Turn content/message
        intent: Optional intent/category for the turn
    """
    turn = {
        "role": role,
        "content": content,
        "intent": intent,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }
    session_state.setdefault("turns", []).append(turn)
    session_state["turn_count"] = session_state.get("turn_count", 0) + 1


def build_history_prompt(
    session_state: dict[str, Any],
    max_turns: int = 8,
) -> str:
    """Render the last N conversation turns as a <history> XML block.

    Returns an empty string when there are no turns yet.

    Args:
        session_state: The session state dict with "turns" key
        max_turns: Maximum number of recent turns to include

    Returns:
        XML-formatted history block for LLM injection
    """
    turns = session_state.get("turns", [])[-max_turns:]
    if not turns:
        return ""

    lines = []
    for t in turns:
        intent_attr = f' intent="{t["intent"]}"' if t.get("intent") else ""
        lines.append(f'  <turn role="{t["role"]}"{intent_attr}>{t["content"]}</turn>')

    return "<history>\n" + "\n".join(lines) + "\n</history>"


def get_execution_context(state: dict[str, Any], max_turns: int = 5) -> dict[str, str]:
    """Extract execution context for LLM prompts.

    Combines conversation history, current state info, and metadata into
    a dict suitable for prompt templating.

    Args:
        state: The LangGraph state dict
        max_turns: Maximum turns to include in history

    Returns:
        Dict with keys: history, current_state, error_message, etc.
    """
    history = build_history_prompt(state, max_turns)
    return {
        "history": history,
        "current_state": str(state.get("current_state", "unknown")),
        "proposed_next": str(state.get("proposed_next", "unknown")),
        "error_message": state.get("error_message", ""),
        "turn_count": str(state.get("turn_count", 0)),
    }


def init_session_defaults(
    session_state: dict[str, Any],
    defaults: Optional[dict[str, Any]] = None,
) -> None:
    """Ensure all required keys exist in session_state.

    Args:
        session_state: The session state dict to initialize (mutated in-place)
        defaults: Optional dict of {key: default_value} to apply
    """
    defaults = defaults or {}

    # Common session keys
    base_defaults = {
        "turns": [],
        "turn_count": 0,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    for key, value in {**base_defaults, **defaults}.items():
        session_state.setdefault(key, value)
