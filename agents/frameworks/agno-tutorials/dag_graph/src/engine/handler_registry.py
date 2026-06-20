"""
engine/handler_registry.py
────────────────────────────────────────────────────────────────────────────
Handler registration and @handler decorator for state machine workflows.

Provides:
  • HandlerMetadata dataclass to store handler configuration
  • HANDLER_MAP_METADATA global registry
  • @handler decorator to register handlers with metadata
  • Helper functions to query metadata

Metadata allows handlers to declare:
  • state: Which state this handler processes
  • waits_for_input: If True, workflow pauses and waits for next turn
  • description: Human-readable description (optional)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class HandlerMetadata:
    """Metadata for a state handler."""

    state: str
    waits_for_input: bool = False
    description: Optional[str] = None


# Global registry populated by @handler decorator
HANDLER_MAP_METADATA: dict[str, HandlerMetadata] = {}


def handler(state: str,
            waits_for_input: bool = False,
            description: Optional[str] = None) -> Callable:
    """
    Decorator that registers a handler function with metadata.

    Usage:
        @handler(state="validate", waits_for_input=False, description="Validate doc")
        def handle_validate(state: PipelineState) -> PipelineState:
            ...

    Args:
        state: State enum value (e.g., "validate")
        waits_for_input: If True, workflow pauses and waits for next turn
        description: Human-readable description (optional)

    Returns:
        Decorator function that registers metadata and returns original function
    """
    def decorator(func: Callable) -> Callable:
        HANDLER_MAP_METADATA[state] = HandlerMetadata(
            state=state,
            waits_for_input=waits_for_input,
            description=description
        )
        return func

    return decorator


def get_handler_metadata(state: str) -> Optional[HandlerMetadata]:
    """
    Retrieve metadata for a registered state.

    Args:
        state: State to look up

    Returns:
        HandlerMetadata if registered, None otherwise
    """
    return HANDLER_MAP_METADATA.get(state)


def does_state_wait_for_input(state: str) -> bool:
    """
    Check if a state pauses and waits for user input.

    Args:
        state: State to check

    Returns:
        True if state has waits_for_input=True, False otherwise
    """
    meta = get_handler_metadata(state)
    return meta.waits_for_input if meta else False
