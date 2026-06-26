"""
Handler registration and @handler decorator.

Provides metadata registry that allows handlers to declare:
- state: Which state this handler processes
- waits_for_input: If True, workflow pauses at this state
- description: Human-readable description
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class HandlerMetadata:
    """Metadata configuration for a state handler."""

    state: str
    waits_for_input: bool = False
    description: Optional[str] = None

    def __repr__(self) -> str:
        return (
            f"HandlerMetadata(state={self.state!r}, "
            f"waits_for_input={self.waits_for_input}, "
            f"description={self.description!r})"
        )


# Global registry: populated by @handler decorator
HANDLER_MAP_METADATA: dict[str, HandlerMetadata] = {}


def handler(
    state: str,
    waits_for_input: bool = False,
    description: Optional[str] = None,
) -> Callable:
    """
    Decorator that registers a handler function with metadata.

    Usage:
        @handler(state="validate", waits_for_input=False)
        def handle_validate(state: PipelineState) -> PipelineState:
            # Validate document
            state["current_state"] = "enrich"
            return state

    Args:
        state: State enum value (e.g., "validate", "enrich")
        waits_for_input: If True, workflow pauses and waits for next turn
        description: Human-readable description (optional)

    Returns:
        Decorator function that registers metadata and returns original function
    """
    def decorator(func: Callable) -> Callable:
        HANDLER_MAP_METADATA[state] = HandlerMetadata(
            state=state,
            waits_for_input=waits_for_input,
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
    return HANDLER_MAP_METADATA.get(state)


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


def list_handler_metadata() -> list[HandlerMetadata]:
    """List all registered handler metadata."""
    return list(HANDLER_MAP_METADATA.values())


def clear_metadata() -> None:
    """Clear registry (useful for testing). Use with caution."""
    HANDLER_MAP_METADATA.clear()
