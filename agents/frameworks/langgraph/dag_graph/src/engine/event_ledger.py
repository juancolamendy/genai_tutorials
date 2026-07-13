"""File-backed dedupe store for system-sourced event delivery.

Used by aemit_event (a later phase) to detect a retried/duplicate delivery
of the same event_id and avoid double-applying its state transition. Must
be file-backed, not in-memory: the CLI is a fresh process per one-shot
command, so an in-memory ledger would never actually dedupe anything across
separate invocations — the decisive scenario is two EventLedger instances
pointed at the same directory (simulating two separate process runs)
agreeing on what's already been processed.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path

DEFAULT_LEDGER_DIR = ".event_ledger"


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
