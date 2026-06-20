"""SQLite checkpointing for LangGraph workflows.

Enables resumable execution and state persistence.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# Global connection for in-memory database
_memory_conn: Optional[sqlite3.Connection] = None


class SqliteCheckpointer:
    """SQLite-based checkpoint storage for workflow state."""

    def __init__(self, db_path: str = ":memory:"):
        """Initialize checkpointer.

        Args:
            db_path: Path to SQLite database file (":memory:" for in-memory)
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

        if db_path == ":memory:":
            # Keep global connection for in-memory databases
            global _memory_conn
            if _memory_conn is None:
                _memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
                self._init_db_on_conn(_memory_conn)
            self._conn = _memory_conn
        else:
            # File-based database
            self._init_db()

    def _init_db_on_conn(self, conn: sqlite3.Connection) -> None:
        """Initialize database tables on a specific connection."""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                state TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (thread_id, checkpoint_id)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                thread_id TEXT PRIMARY KEY,
                latest_checkpoint_id TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        conn.commit()

    def _init_db(self) -> None:
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        try:
            self._init_db_on_conn(conn)
        finally:
            conn.close()

    def save(self, thread_id: str, checkpoint_id: str, state: dict[str, Any]) -> None:
        """Save state checkpoint.

        Args:
            thread_id: Unique thread/execution identifier
            checkpoint_id: Unique checkpoint identifier
            state: State dict to persist
        """
        try:
            state_json = json.dumps(state, default=str)
            now = datetime.now().isoformat()

            if self._conn is not None:
                conn = self._conn
            else:
                conn = sqlite3.connect(self.db_path)

            try:
                # Save checkpoint
                conn.execute(
                    """
                    INSERT OR REPLACE INTO checkpoints (thread_id, checkpoint_id, state, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (thread_id, checkpoint_id, state_json, now),
                )

                # Update metadata
                conn.execute(
                    """
                    INSERT OR REPLACE INTO metadata (thread_id, latest_checkpoint_id, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (thread_id, checkpoint_id, now),
                )
                conn.commit()

                log.debug(f"Checkpoint saved: {thread_id}/{checkpoint_id}")
            finally:
                if self._conn is None:
                    conn.close()
        except Exception as e:
            log.error(f"Failed to save checkpoint: {e}", exc_info=True)
            raise

    def load(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Load state checkpoint.

        Args:
            thread_id: Unique thread/execution identifier
            checkpoint_id: Specific checkpoint ID (latest if None)

        Returns:
            Loaded state dict, or None if not found
        """
        try:
            if self._conn is not None:
                conn = self._conn
            else:
                conn = sqlite3.connect(self.db_path)

            try:
                # Get checkpoint ID
                if checkpoint_id is None:
                    cursor = conn.execute(
                        "SELECT latest_checkpoint_id FROM metadata WHERE thread_id = ?",
                        (thread_id,),
                    )
                    row = cursor.fetchone()
                    if not row:
                        log.debug(f"No checkpoint found for thread: {thread_id}")
                        return None
                    checkpoint_id = row[0]

                # Load checkpoint
                cursor = conn.execute(
                    "SELECT state FROM checkpoints WHERE thread_id = ? AND checkpoint_id = ?",
                    (thread_id, checkpoint_id),
                )
                row = cursor.fetchone()
                if not row:
                    log.debug(f"Checkpoint not found: {thread_id}/{checkpoint_id}")
                    return None

                state = json.loads(row[0])
                log.debug(f"Checkpoint loaded: {thread_id}/{checkpoint_id}")
                return state
            finally:
                if self._conn is None:
                    conn.close()
        except Exception as e:
            log.error(f"Failed to load checkpoint: {e}", exc_info=True)
            return None

    def delete(self, thread_id: str) -> None:
        """Delete all checkpoints for a thread.

        Args:
            thread_id: Unique thread/execution identifier
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
                conn.execute("DELETE FROM metadata WHERE thread_id = ?", (thread_id,))
                conn.commit()

            log.debug(f"Checkpoints deleted: {thread_id}")
        except Exception as e:
            log.error(f"Failed to delete checkpoints: {e}", exc_info=True)
            raise

    def list_checkpoints(self, thread_id: str) -> list[str]:
        """List all checkpoint IDs for a thread.

        Args:
            thread_id: Unique thread/execution identifier

        Returns:
            List of checkpoint IDs
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT checkpoint_id FROM checkpoints WHERE thread_id = ? ORDER BY created_at DESC",
                    (thread_id,),
                )
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            log.error(f"Failed to list checkpoints: {e}", exc_info=True)
            return []


# Global checkpointer instance
_checkpointer: Optional[SqliteCheckpointer] = None


def init_checkpointer(db_path: str = ":memory:") -> SqliteCheckpointer:
    """Initialize global checkpointer.

    Args:
        db_path: Path to SQLite database file

    Returns:
        SqliteCheckpointer instance
    """
    global _checkpointer
    _checkpointer = SqliteCheckpointer(db_path)
    return _checkpointer


def get_checkpointer() -> Optional[SqliteCheckpointer]:
    """Get global checkpointer instance."""
    return _checkpointer
