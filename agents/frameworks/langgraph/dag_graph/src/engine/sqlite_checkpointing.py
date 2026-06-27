"""SQLite checkpointing for LangGraph workflows.

Implements BaseCheckpointSaver protocol for seamless integration with LangGraph.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from typing import Any, AsyncIterator, Iterator, Optional, Sequence

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.types import RunnableConfig

log = logging.getLogger(__name__)

# Global connection for in-memory database
_memory_conn: Optional[sqlite3.Connection] = None


class SqliteCheckpointer(BaseCheckpointSaver):
    """SQLite-based checkpoint storage implementing LangGraph's BaseCheckpointSaver."""

    def __init__(self, db_path: str = ":memory:"):
        """Initialize checkpointer.

        Args:
            db_path: Path to SQLite database file (":memory:" for in-memory)
        """
        super().__init__()
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
                ts_created TEXT NOT NULL,
                metadata TEXT NOT NULL,
                checkpoint_values TEXT NOT NULL,
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

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._conn is not None:
            return self._conn
        return sqlite3.connect(self.db_path)

    def _release_conn(self, conn: sqlite3.Connection) -> None:
        """Release database connection if it's not the global one."""
        if self._conn is None:
            conn.close()

    def _extract_thread_id(self, config: RunnableConfig) -> str:
        """Extract thread_id from config."""
        if config is None:
            raise ValueError("Config must contain thread_id")
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        if not thread_id:
            raise ValueError("Config must contain configurable.thread_id")
        return thread_id

    # ─────────────────────────────────────────────────────────────────────────
    # BaseCheckpointSaver Protocol Implementation
    # ─────────────────────────────────────────────────────────────────────────

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any,
    ) -> RunnableConfig:
        """Save a checkpoint.

        Args:
            config: RunnableConfig with thread_id in configurable
            checkpoint: Checkpoint dict to save
            metadata: CheckpointMetadata with ts and other info
            new_versions: Channel version info (unused for simple SQLite storage)

        Returns:
            Updated config with checkpoint_id
        """
        thread_id = self._extract_thread_id(config)
        checkpoint_id = metadata.get("id", str(uuid.uuid4()))

        try:
            conn = self._get_conn()
            try:
                ts_created = metadata.get("ts_created", datetime.now().isoformat())
                metadata_json = json.dumps(metadata, default=str)

                # LangGraph's checkpoint structure varies - handle both cases
                if "values" in checkpoint:
                    # Standard case: checkpoint = {"values": {...state...}}
                    values_json = json.dumps(checkpoint["values"], default=str)
                else:
                    # Alternative: checkpoint IS the values dict
                    values_json = json.dumps(checkpoint, default=str)

                conn.execute(
                    """
                    INSERT OR REPLACE INTO checkpoints
                    (thread_id, checkpoint_id, ts_created, metadata, checkpoint_values)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (thread_id, checkpoint_id, ts_created, metadata_json, values_json),
                )

                # Update metadata
                conn.execute(
                    """
                    INSERT OR REPLACE INTO metadata (thread_id, latest_checkpoint_id, updated_at)
                    VALUES (?, ?, ?)
                    """,
                    (thread_id, checkpoint_id, datetime.now().isoformat()),
                )
                conn.commit()

                log.debug(f"Checkpoint saved: {thread_id}/{checkpoint_id}")

                # Return config with checkpoint_id
                return {
                    **config,
                    "checkpoint_id": checkpoint_id,
                }
            finally:
                self._release_conn(conn)
        except Exception as e:
            log.error(f"Failed to save checkpoint: {e}", exc_info=True)
            raise

    def get(self, config: RunnableConfig) -> Checkpoint | None:
        """Load the latest checkpoint for a thread.

        Args:
            config: RunnableConfig with thread_id in configurable

        Returns:
            Checkpoint dict or None if not found
        """
        thread_id = self._extract_thread_id(config)

        try:
            conn = self._get_conn()
            try:
                # Get latest checkpoint ID from metadata
                cursor = conn.execute(
                    "SELECT latest_checkpoint_id FROM metadata WHERE thread_id = ?",
                    (thread_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return None

                checkpoint_id = row[0]

                # Load checkpoint
                cursor = conn.execute(
                    (
                        "SELECT checkpoint_values FROM checkpoints "
                        "WHERE thread_id = ? AND checkpoint_id = ?"
                    ),
                    (thread_id, checkpoint_id),
                )
                row = cursor.fetchone()
                if not row:
                    return None

                values = json.loads(row[0])
                return {"values": values}
            finally:
                self._release_conn(conn)
        except Exception as e:
            log.error(f"Failed to load checkpoint: {e}", exc_info=True)
            return None

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        """Load the latest checkpoint tuple for a thread.

        Args:
            config: RunnableConfig with thread_id in configurable

        Returns:
            CheckpointTuple (config, checkpoint, metadata) or None
        """
        thread_id = self._extract_thread_id(config)

        try:
            conn = self._get_conn()
            try:
                # Get latest checkpoint ID from metadata
                cursor = conn.execute(
                    "SELECT latest_checkpoint_id FROM metadata WHERE thread_id = ?",
                    (thread_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return None

                checkpoint_id = row[0]

                # Load checkpoint with metadata
                cursor = conn.execute(
                    """
                    SELECT checkpoint_values, metadata, ts_created
                    FROM checkpoints
                    WHERE thread_id = ? AND checkpoint_id = ?
                    """,
                    (thread_id, checkpoint_id),
                )
                row = cursor.fetchone()
                if not row:
                    return None

                values_json, metadata_json, ts_created = row
                values = json.loads(values_json)
                metadata = json.loads(metadata_json)

                return (
                    {**config, "checkpoint_id": checkpoint_id},
                    {"values": values},
                    metadata,
                )
            finally:
                self._release_conn(conn)
        except Exception as e:
            log.error(f"Failed to load checkpoint tuple: {e}", exc_info=True)
            return None

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints for a thread.

        Args:
            config: RunnableConfig with thread_id in configurable
            filter: Optional filter dict (unused)
            before: Optional config to list checkpoints before
            limit: Optional limit on number of results

        Yields:
            CheckpointTuple objects
        """
        if config is None:
            return

        thread_id = self._extract_thread_id(config)

        try:
            conn = self._get_conn()
            try:
                query = """
                    SELECT checkpoint_id, ts_created, metadata, checkpoint_values
                    FROM checkpoints
                    WHERE thread_id = ?
                    ORDER BY ts_created DESC
                """
                params = [thread_id]

                if limit:
                    query += " LIMIT ?"
                    params.append(limit)

                cursor = conn.execute(query, params)
                for row in cursor.fetchall():
                    checkpoint_id, ts_created, metadata_json, values_json = row
                    metadata = json.loads(metadata_json)
                    values = json.loads(values_json)
                    yield (
                        {**config, "checkpoint_id": checkpoint_id},
                        {"values": values},
                        metadata,
                    )
            finally:
                self._release_conn(conn)
        except Exception as e:
            log.error(f"Failed to list checkpoints: {e}", exc_info=True)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """Async version of list.

        Args:
            config: RunnableConfig with thread_id in configurable
            filter: Optional filter dict (unused)
            before: Optional config to list checkpoints before
            limit: Optional limit on number of results

        Yields:
            CheckpointTuple objects
        """
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Save intermediate writes/channel updates.

        For simple use cases, we store these as part of the checkpoint.
        This is called by LangGraph during execution for channel updates.

        Args:
            config: RunnableConfig with thread_id
            writes: List of (channel_name, value) tuples
            task_id: Task identifier
            task_path: Task path
        """
        # For a simple implementation, we don't store intermediate writes separately
        # They will be included in the full checkpoint via put()
        pass

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        """Async version of put_writes."""
        self.put_writes(config, writes, task_id, task_path)

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoints for a thread.

        Args:
            thread_id: Unique thread/execution identifier
        """
        try:
            conn = self._get_conn()
            try:
                conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
                conn.execute("DELETE FROM metadata WHERE thread_id = ?", (thread_id,))
                conn.commit()
                log.debug(f"Checkpoints deleted: {thread_id}")
            finally:
                self._release_conn(conn)
        except Exception as e:
            log.error(f"Failed to delete checkpoints: {e}", exc_info=True)
            raise

    async def adelete_thread(self, thread_id: str) -> None:
        """Async version of delete_thread."""
        self.delete_thread(thread_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Legacy Methods (for backward compatibility)
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, thread_id: str, checkpoint_id: str, state: dict[str, Any]) -> None:
        """Legacy save method for backward compatibility.

        Args:
            thread_id: Unique thread/execution identifier
            checkpoint_id: Checkpoint identifier
            state: State dict to persist
        """
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = {"values": state}
        metadata = {"id": checkpoint_id, "ts_created": datetime.now().isoformat()}
        self.put(config, checkpoint, metadata, None)

    def load(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Legacy load method for backward compatibility.

        Args:
            thread_id: Unique thread/execution identifier
            checkpoint_id: Specific checkpoint ID (latest if None)

        Returns:
            Loaded state dict, or None if not found
        """
        config = {"configurable": {"thread_id": thread_id}}

        if checkpoint_id:
            # Load specific checkpoint
            conn = self._get_conn()
            try:
                cursor = conn.execute(
                    (
                        "SELECT checkpoint_values FROM checkpoints "
                        "WHERE thread_id = ? AND checkpoint_id = ?"
                    ),
                    (thread_id, checkpoint_id),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                return json.loads(row[0])
            finally:
                self._release_conn(conn)
        else:
            # Load latest
            checkpoint = self.get(config)
            return checkpoint["values"] if checkpoint else None

    def list_checkpoints(self, thread_id: str) -> list[str]:
        """Legacy list method for backward compatibility.

        Args:
            thread_id: Unique thread/execution identifier

        Returns:
            List of checkpoint IDs
        """
        config = {"configurable": {"thread_id": thread_id}}
        try:
            return [item[0]["checkpoint_id"] for item in self.list(config)]
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
