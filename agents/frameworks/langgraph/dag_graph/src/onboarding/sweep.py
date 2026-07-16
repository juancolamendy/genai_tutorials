"""Timeout sweep for the onboarding pipeline (design spec §13, corrected).

Enumerates active sessions via EngineGraph.get_active_sessions() (which
hides the checkpointer's internal file/row shape — resolves what was
Open Question 3), and emits a timeout_escalation system event for any
thread that's been sitting at a park state past its threshold.
"""

from __future__ import annotations

import time
from typing import Any


def _stale(state: dict[str, Any], threshold_seconds: float) -> bool:
    """A session is stale once threshold_seconds have elapsed since
    started_at. Approximates "time spent at the current park state" with
    "time since the workflow started" — the state schema has no per-state
    entry timestamp, so this is the best available signal without adding
    one; acceptable for a sweep whose thresholds are measured in
    days (§14's onboarding timeouts), not something a few extra states'
    worth of elapsed time would meaningfully skew.
    """
    started_at = state.get("started_at")
    if started_at is None:
        return False
    return bool((time.time() - started_at) > threshold_seconds)


async def sweep(graph: Any, thresholds: dict[str, float]) -> list[dict[str, Any]]:
    """Emit a timeout_escalation event for every active session sitting
    past its threshold.

    Args:
        graph: An onboarding Graph (or any EngineGraph subclass) instance
        thresholds: {current_state: max_seconds_before_stale} — only
            states present here are ever considered for escalation

    Returns:
        One aemit_event() result per thread that was swept
    """
    results = []
    for session in graph.get_active_sessions():
        state = session["state"]
        current = state.get("current_state")
        if current in thresholds and _stale(state, thresholds[current]):
            results.append(
                await graph.aemit_event(
                    thread_id=session["thread_id"],
                    event_source="system",
                    event_type="timeout_escalation",
                    # Deterministic, not timestamp-based: overlapping sweep
                    # runs correctly dedupe against each other via the
                    # file-backed ledger.
                    event_id=f"sweep:{session['thread_id']}:{current}",
                )
            )
    return results


__all__ = ["sweep"]
