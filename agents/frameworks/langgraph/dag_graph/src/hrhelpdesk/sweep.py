"""Timeout sweep for sticky booking sessions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _booking_stale(state: dict[str, Any], threshold_seconds: float) -> bool:
    """True when an active booking topic exceeded the inactivity threshold."""
    if state.get("active_topic") != "booking":
        return False
    started = state.get("topic_started_at")
    if not started:
        return False
    try:
        started_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
    except ValueError:
        return False
    elapsed = (datetime.now(timezone.utc) - started_dt).total_seconds()
    return elapsed > threshold_seconds


async def sweep(graph: Any, thresholds: dict[str, float]) -> list[dict[str, Any]]:
    """Emit topic_timeout for stale booking lanes; log open-ticket reminders."""
    results: list[dict[str, Any]] = []
    booking_threshold = thresholds.get("topic_booking", 48 * 3600.0)

    for session in graph.get_active_sessions():
        state = session["state"]
        thread_id = session["thread_id"]

        if _booking_stale(state, booking_threshold):
            results.append(
                await graph.aemit_event(
                    thread_id=thread_id,
                    source="system",
                    event_type="topic_timeout",
                    event_id=f"sweep:{thread_id}:topic_timeout",
                )
            )

        if state.get("open_tickets"):
            # Reminder stub — log only in v1
            pass

    return results


__all__ = ["sweep"]
