"""Conversational message helpers for chatbot workflows.

Generic — no topic/helpdesk vocabulary. Domains call these when closing a
sticky conversation segment or trimming context for prompts.
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage


def trim_messages(
    messages: Sequence[BaseMessage],
    max_n: int = 12,
) -> list[BaseMessage]:
    """Return the last ``max_n`` messages (prompt window)."""
    if max_n <= 0:
        return []
    return list(messages[-max_n:])


def segment_reset_messages(
    messages: Sequence[BaseMessage],
    summary: str,
) -> list[Any]:
    """Build a messages-channel update that clears prior turns and keeps a summary.

    Returns a list suitable for the ``messages`` reducer: one ``RemoveMessage``
    per existing message id, then a single summary ``AIMessage``. Messages
    without an id are skipped for removal (cannot target them safely).
    """
    removals: list[Any] = []
    for msg in messages:
        msg_id = getattr(msg, "id", None)
        if msg_id:
            removals.append(RemoveMessage(id=msg_id))
    summary_msg = AIMessage(content=summary)
    return [*removals, summary_msg]


def close_topic_delta(
    messages: Sequence[BaseMessage],
    summary: str,
) -> dict[str, Any]:
    """State delta for closing a sticky lane: clear topic fields + reset messages.

    Domains may merge additional fields; this returns only the shared chat keys.
    """
    return {
        "active_topic": None,
        "topic_started_at": None,
        "topic_data": {},
        "messages": segment_reset_messages(messages, summary),
    }
