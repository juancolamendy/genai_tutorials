"""Generic helpers for chatbot workflows and handler tooling.

Includes message hygiene (trim / segment reset) plus small utilities shared
across domain handlers (ledger sync checks, tool-call inspection, logging).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage, ToolMessage

log = logging.getLogger(__name__)


# ── Message hygiene ───────────────────────────────────────────────────────────

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

    The summary ``AIMessage`` is marked ``additional_kwargs.segment_reset=True``
    so streaming entry points can skip it — it is transcript hygiene, not the
    user-facing reply (that lives in ``output_messages``).
    """
    removals: list[Any] = []
    for msg in messages:
        msg_id = getattr(msg, "id", None)
        if msg_id:
            removals.append(RemoveMessage(id=msg_id))
    summary_msg = AIMessage(
        content=summary,
        additional_kwargs={"segment_reset": True},
    )
    return [*removals, summary_msg]


def close_topic_delta(
    messages: Sequence[BaseMessage],
    summary: str,
) -> dict[str, Any]:
    """State delta for closing a sticky lane: clear topic fields + reset messages.
    """
    return {
        "active_topic": None,
        "topic_started_at": None,
        "topic_data": {},
        "messages": segment_reset_messages(messages, summary),
    }


# ── Ledger (sync wrappers for sync handlers) ──────────────────────────────────

def ledger_is_processed_sync(ledger: Any, key: str) -> bool:
    """Sync check whether a keyed ledger marker exists."""
    return ledger._marker_path(key).exists()


def ledger_mark_processed_sync(ledger: Any, key: str) -> None:
    """Sync write of a keyed ledger marker."""
    ledger._mark_processed_sync(key)


# ── Hashing ───────────────────────────────────────────────────────────────────

def content_hash(*parts: str) -> str:
    """Short stable hash for idempotency keys (first 16 hex chars of sha256)."""
    return hashlib.sha256(":".join(parts).encode()).hexdigest()[:16]


# ── Handler logging ───────────────────────────────────────────────────────────

def log_handler_enter(handler_name: str, state: dict[str, Any]) -> None:
    """Trace handler entry with common control-plane fields."""
    log.info(
        "[HANDLER] ▶ %s  current_state=%s  active_topic=%s  event=%s",
        handler_name,
        state.get("current_state"),
        state.get("active_topic"),
        state.get("current_event_type"),
    )


def log_handler_exit(handler_name: str, delta: dict[str, Any]) -> dict[str, Any]:
    """Trace handler exit and return the delta unchanged (for return chaining)."""
    log.info("[HANDLER] ◀ %s  delta_keys=%s", handler_name, sorted(delta.keys()))
    return delta


# ── Agent message inspection ──────────────────────────────────────────────────

def last_ai_content(messages: Sequence[Any]) -> str:
    """Return plain text from the most recent AIMessage, or empty string.

    Normalizes provider content shapes: plain ``str`` or a list of content
    blocks (e.g. ``[{"type": "text", "text": "..."}]``).
    """
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return _normalize_ai_content(msg.content)
    return ""


def _normalize_ai_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content) if content else ""


def find_tool_message(
    messages: Sequence[Any], tool_name: str
) -> Optional[ToolMessage]:
    """Return the most recent ToolMessage with the given tool name, if any."""
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and msg.name == tool_name:
            return msg
    return None


def find_tool_call(
    messages: Sequence[Any], tool_name: str
) -> Optional[dict[str, Any]]:
    """Return the most recent tool_call dict with the given name, if any."""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                if call.get("name") == tool_name:
                    return call
    return None
