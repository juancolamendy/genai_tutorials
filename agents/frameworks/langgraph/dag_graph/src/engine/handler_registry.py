"""
Handler registration and @handler decorator.

Provides metadata registry that allows handlers to declare:
- state: Which state this handler processes
- waits_for_input: If True, workflow pauses at this state
- wait_kind: Who may resume this state ("human", "system_event", "either")
- expected_events: Legal system event types that may resume this state
- description: Human-readable description
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Optional


@dataclass
class HandlerMetadata:
    """Metadata configuration for a state handler."""

    state: str
    waits_for_input: bool = False
    wait_kind: Literal["human", "system_event", "either"] = "either"
    expected_events: Optional[list[str]] = None
    description: Optional[str] = None


# Global registry: populated by @handler decorator
handler_metadata_map: dict[str, HandlerMetadata] = {}


def handler(
    state: str,
    waits_for_input: bool = False,
    wait_kind: Literal["human", "system_event", "either"] = "either",
    expected_events: Optional[list[str]] = None,
    description: Optional[str] = None,
) -> Callable:
    """
    Decorator that registers a handler function with metadata.

    Usage:
        @handler(state="validate", waits_for_input=False)
        def handle_validate(state: SessionState) -> SessionState:
            # Validate document
            state["current_state"] = "enrich"
            return state

    Args:
        state: State enum value (e.g., "validate", "enrich")
        waits_for_input: If True, workflow pauses and waits for next turn
        wait_kind: Who may resume this park state — "human" (chat turn only),
            "system_event" (webhook/timeout only), or "either" (default,
            matches today's behavior for every existing call site)
        expected_events: Legal system event types that may resume this state
            (ignored for wait_kind="human" states)
        description: Human-readable description (optional)

    Returns:
        Decorator function that registers metadata and returns original function
    """
    def decorator(func: Callable) -> Callable:
        handler_metadata_map[state] = HandlerMetadata(
            state=state,
            waits_for_input=waits_for_input,
            wait_kind=wait_kind,
            expected_events=expected_events,
            description=description,
        )
        return func

    return decorator


def get_handler_metadata(state: str) -> Optional[HandlerMetadata]:
    """
    Retrieve metadata for a registered state.

    Args:
        state: State to look up (e.g., "validate")

    Returns:
        HandlerMetadata if registered, None otherwise
    """
    return handler_metadata_map.get(state)


def does_state_wait_for_input(state: str) -> bool:
    """
    Check if a state pauses and waits for user input.

    Args:
        state: State to check (e.g., "human_review")

    Returns:
        True if state has waits_for_input=True, False otherwise
    """
    meta = get_handler_metadata(state)
    return meta.waits_for_input if meta else False


def clear_metadata() -> None:
    """Clear registry (useful for testing). Use with caution."""
    handler_metadata_map.clear()
