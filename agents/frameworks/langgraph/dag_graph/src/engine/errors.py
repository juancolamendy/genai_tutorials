"""Exception types for the engine's async event-extension entry points.

Raised by EngineGraph.arun_to_completion() (added in a later phase of this
design) to distinguish the specific ways a bg/batch run can fail to reach a
clean terminal state, so callers can branch on which one occurred rather
than pattern-matching an error string.
"""

from __future__ import annotations


class GraphIncompleteError(Exception):
    """A run stopped without reaching a terminal state and isn't parked at a
    waits_for_input state either (e.g. max_auto_iters exhausted)."""


class GraphAlreadyInteractiveError(Exception):
    """arun_to_completion() was called against a thread already parked
    mid-flow at a waits_for_input state — use aemit_event() instead."""


class GraphAlreadyCompleteError(Exception):
    """arun_to_completion() was called against a thread already at a
    terminal state — call it again with a new session_id for a new run."""


class GraphRunError(Exception):
    """A run ended with status="error"."""
