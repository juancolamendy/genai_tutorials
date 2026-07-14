"""Session state for hub + sticky-topic conversational (chatbot) workflows."""

from __future__ import annotations

from typing import Any, Dict, Optional

from src.engine.engine_session_state import EngineSessionState, new_engine_session_state


class ChatEngineSessionState(EngineSessionState):
    """Base session state for hub + sticky-topic conversational workflows.

    Linear pipelines keep using EngineSessionState. Chatbot domains
    (e.g. hrhelpdesk) inherit this and add business payload fields.

    Fields:
      • schema_version: Durable schema revision for migration shims
      • active_topic: Sticky lane id while a topic is open; None at hub
      • topic_started_at: ISO timestamp when the sticky topic opened (sweep)
      • topic_data: Lean per-lane scratch (slots, etc.) — not document blobs
    """

    schema_version: int
    active_topic: Optional[str]
    topic_started_at: Optional[str]
    topic_data: Dict[str, Any]


def new_chat_session_state(
    *,
    current_state: str = "idle",
    proposed_next: str = "idle",
) -> ChatEngineSessionState:
    """Create fresh chat session state with sticky-lane fields initialized."""
    base = new_engine_session_state()
    chat_state: ChatEngineSessionState = {
        **base,
        "current_state": current_state,
        "proposed_next": proposed_next,
        "schema_version": 1,
        "active_topic": None,
        "topic_started_at": None,
        "topic_data": {},
    }
    return chat_state


__all__ = [
    "ChatEngineSessionState",
    "new_chat_session_state",
]
