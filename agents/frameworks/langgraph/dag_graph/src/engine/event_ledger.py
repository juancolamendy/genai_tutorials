"""File-backed keyed ledger for once-only application of durable keys.

Used for:
  • system-event delivery dedupe (``event:{event_id}``)
  • write-effect idempotency (``effect:{thread}:{kind}:{...}``)

Must be file-backed, not in-memory: the CLI is a fresh process per one-shot
command, so an in-memory ledger would never actually dedupe anything across
separate invocations — the decisive scenario is two EventLedger instances
pointed at the same directory (simulating two separate process runs)
agreeing on what's already been processed.

``EventLedger`` remains the public name (alias for the keyed store). Callers
may pass bare ids (legacy) or namespaced keys from ``event_key`` / ``effect_key``.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path

DEFAULT_LEDGER_DIR = ".event_ledger"


def event_key(event_id: str) -> str:
    """Namespace a delivery id so it never collides with effect keys."""
    if event_id.startswith("event:") or event_id.startswith("effect:"):
        return event_id
    return f"event:{event_id}"


def effect_key(thread_id: str, kind: str, *parts: str) -> str:
    """Build an effect idempotency key: ``effect:{thread}:{kind}:{parts...}``."""
    safe_parts = [thread_id, kind, *parts]
    return "effect:" + ":".join(safe_parts)


class EventLedger:
    """One marker file per event_id — avoids any read-modify-write race on
    a shared blob, since each event_id's dedupe state is fully independent.
    """

    def __init__(self, ledger_dir: str = DEFAULT_LEDGER_DIR) -> None:
        self.ledger_dir = Path(ledger_dir)
        self.ledger_dir.mkdir(parents=True, exist_ok=True)

    def _marker_path(self, event_id: str) -> Path:
        safe_id = event_id.replace(":", "_").replace("/", "_")
        return self.ledger_dir / f"{safe_id}.json"

    async def is_processed(self, event_id: str) -> bool:
        return await asyncio.to_thread(self._marker_path(event_id).exists)

    async def mark_processed(self, event_id: str) -> None:
        await asyncio.to_thread(self._mark_processed_sync, event_id)

    def _mark_processed_sync(self, event_id: str) -> None:
        path = self._marker_path(event_id)
        data = {"event_id": event_id, "processed_at": datetime.now().isoformat()}
        # Atomic write: temp file in the same directory, then rename — the
        # same pattern JsonCheckpointer already uses for its session files.
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False, suffix=".tmp"
        ) as tmp:
            json.dump(data, tmp)
            tmp_path = tmp.name
        Path(tmp_path).replace(path)
