"""Directory-based session checkpointer for .doc_sessions storage.

Stores each session as a separate JSON file in .doc_sessions directory,
similar to Agno's session storage pattern.
"""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.types import RunnableConfig

log = logging.getLogger(__name__)

# Default sessions directory
DEFAULT_SESSIONS_DIR = ".doc_sessions"


class JsonCheckpointer(BaseCheckpointSaver):
    """
    Directory-based checkpointer that stores sessions in .doc_sessions directory.

    Each session is stored as a JSON file with thread_id as the filename.
    This matches the Agno session storage pattern while maintaining LangGraph
    checkpointer interface compatibility.
    """

    def __init__(self, sessions_dir: str = DEFAULT_SESSIONS_DIR) -> None:
        """
        Initialize JSON checkpointer.

        Args:
            sessions_dir: Directory to store session files (default: .doc_sessions)
        """
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"[JsonCheckpointer] Initialized with sessions_dir={self.sessions_dir}")

    def save_pause_point(self, config: RunnableConfig, checkpoint_id: str) -> None:
        """Mark a checkpoint as a pause point for resumption.

        Args:
            config: Config with thread_id in configurable
            checkpoint_id: The checkpoint_id to mark as pause point
        """
        if not config or "configurable" not in config:
            return

        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return

        path = self._get_session_path(thread_id)
        session_data = self._load_session_file(path)

        if not session_data:
            return

        # Store pause point at session level
        session_data["pause_checkpoint_id"] = checkpoint_id
        session_data["pause_timestamp"] = datetime.now().isoformat()

        self._save_session_file(path, session_data)
        log.info(f"[JsonCheckpointer] Saved pause point {checkpoint_id} for thread {thread_id}")

    def get_pause_point(self, config: RunnableConfig) -> Optional[str]:
        """Get the pause point checkpoint_id for resumption.

        Returns the pause point if explicitly marked, or the latest COMPLETE checkpoint
        (one with 'v' field for LangGraph compatibility).

        Args:
            config: Config with thread_id in configurable

        Returns:
            checkpoint_id of the pause point, or None if not found
        """
        if not config or "configurable" not in config:
            return None

        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return None

        path = self._get_session_path(thread_id)
        session_data = self._load_session_file(path)

        if not session_data:
            return None

        # Check for explicitly marked pause point first
        pause_id = session_data.get("pause_checkpoint_id")
        if pause_id:
            return pause_id

        # Fallback: find latest complete checkpoint (has "v" field for LangGraph)
        checkpoints = session_data.get("checkpoints", {})
        complete_checkpoints = [
            (cid, data) for cid, data in checkpoints.items()
            if isinstance(data.get("checkpoint"), dict) and "v" in data["checkpoint"]
        ]

        if complete_checkpoints:
            # Return the latest (by checkpoint_id which is timestamp-based)
            return sorted(complete_checkpoints, key=lambda x: x[0])[-1][0]

        return None

    def _get_session_path(self, thread_id: str) -> Path:
        """Get file path for a given thread_id."""
        # Sanitize thread_id for filesystem
        safe_id = thread_id.replace(":", "_").replace("/", "_")
        return self.sessions_dir / f"{safe_id}.json"

    def _load_session_file(self, path: Path) -> Optional[dict[str, Any]]:
        """Load session from JSON file."""
        if not path.exists():
            return None

        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.error(f"Error loading session from {path}: {e}")
            return None

    def _save_session_file(self, path: Path, data: dict[str, Any]) -> None:
        """Save session to JSON file with atomic writes."""
        try:
            # Ensure directory exists
            path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temporary file in same directory (ensures atomicity)
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=path.parent,
                delete=False,
                suffix=".tmp",
            ) as tmp:
                json.dump(data, tmp, indent=2, default=str)
                tmp_path = tmp.name

            # Atomic rename (works across platforms)
            Path(tmp_path).replace(path)
        except OSError as e:
            log.error(f"Error saving session to {path}: {e}")
            raise

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any,
    ) -> RunnableConfig:
        """Save a checkpoint.

        Args:
            config: Config with thread_id
            checkpoint: Checkpoint data to save
            metadata: Checkpoint metadata
            new_versions: Channel version info (unused for JSON storage)

        Returns:
            Config with checkpoint_id set
        """
        if not config or "configurable" not in config:
            raise ValueError("Config must have configurable dict with thread_id")

        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            raise ValueError("thread_id must be in config.configurable")

        path = self._get_session_path(thread_id)

        # Load existing session or create new
        session_data = self._load_session_file(path) or {
            "thread_id": thread_id,
            "checkpoints": {},
            "metadata": {},
            "created_at": datetime.now().isoformat(),
        }

        # Handle metadata safely (could be dict or other type)
        if isinstance(metadata, dict):
            checkpoint_id = metadata.get("checkpoint_id", str(datetime.now().timestamp()))
            metadata_to_store = metadata
        else:
            checkpoint_id = str(datetime.now().timestamp())
            metadata_to_store = {"id": str(metadata)} if metadata else {}

        # Store all checkpoints (both complete and intermediate)
        session_data["checkpoints"][checkpoint_id] = {
            "checkpoint": checkpoint,
            "metadata": metadata_to_store,
            "ts_created": datetime.now().isoformat(),
        }

        # Update metadata
        session_data["metadata"] = metadata_to_store
        session_data["updated_at"] = datetime.now().isoformat()
        session_data["latest_checkpoint_id"] = checkpoint_id

        # Save session file
        self._save_session_file(path, session_data)

        log.info(
            f"[JsonCheckpointer] Saved checkpoint {checkpoint_id} for thread {thread_id}"
        )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Load a checkpoint tuple.

        Args:
            config: Config with thread_id and optional checkpoint_id

        Returns:
            CheckpointTuple or None if not found
        """
        if not config or "configurable" not in config:
            return None

        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return None

        path = self._get_session_path(thread_id)
        session_data = self._load_session_file(path)

        if not session_data or "checkpoints" not in session_data:
            return None

        # Get requested checkpoint_id or latest
        checkpoint_id = config["configurable"].get("checkpoint_id")
        if not checkpoint_id:
            checkpoint_id = session_data.get("latest_checkpoint_id")

        if not checkpoint_id or checkpoint_id not in session_data["checkpoints"]:
            return None

        cp_data = session_data["checkpoints"][checkpoint_id]

        # Return stored checkpoint as-is (includes "v", values, channels, etc.)
        stored_checkpoint = cp_data.get("checkpoint", {})
        log.debug(f"[JsonCheckpointer.get_tuple] Loaded checkpoint keys: {stored_checkpoint.keys() if isinstance(stored_checkpoint, dict) else 'not a dict'}")

        return CheckpointTuple(
            config={
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                }
            },
            checkpoint=stored_checkpoint,
            metadata=cp_data.get("metadata", {}),
        )

    def get(self, config: RunnableConfig) -> Optional[Checkpoint]:
        """Load a checkpoint.

        Args:
            config: Config with thread_id

        Returns:
            Checkpoint or None if not found
        """
        tuple_result = self.get_tuple(config)
        return tuple_result.checkpoint if tuple_result else None

    def list(self, config: RunnableConfig) -> list[CheckpointTuple]:
        """List all checkpoints for a thread.

        Args:
            config: Config with thread_id

        Returns:
            List of CheckpointTuple objects
        """
        if not config or "configurable" not in config:
            return []

        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return []

        path = self._get_session_path(thread_id)
        session_data = self._load_session_file(path)

        if not session_data or "checkpoints" not in session_data:
            return []

        result = []
        for checkpoint_id, cp_data in session_data["checkpoints"].items():
            stored_checkpoint = cp_data.get("checkpoint", {})
            result.append(
                CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_id": checkpoint_id,
                        }
                    },
                    checkpoint=stored_checkpoint,
                    metadata=cp_data.get("metadata", {}),
                )
            )

        return result

    def delete_thread(self, config: RunnableConfig) -> None:
        """Delete all checkpoints for a thread.

        Args:
            config: Config with thread_id
        """
        if not config or "configurable" not in config:
            return

        thread_id = config["configurable"].get("thread_id")
        if not thread_id:
            return

        path = self._get_session_path(thread_id)
        try:
            if path.exists():
                path.unlink()
                log.info(f"[JsonCheckpointer] Deleted session for thread {thread_id}")
        except OSError as e:
            log.error(f"Error deleting session file {path}: {e}")

    # Legacy compatibility methods
    def put_writes(
        self,
        config: RunnableConfig,
        writes: list[tuple[str, Any]],
        metadata: CheckpointMetadata,
    ) -> None:
        """Legacy method for compatibility."""
        # Convert writes to checkpoint format
        checkpoint_values = {}
        for key, value in writes:
            checkpoint_values[key] = value

        self.put(
            config,
            Checkpoint(values=checkpoint_values),
            metadata,
            None,
        )

    async def alist(self, config: RunnableConfig):
        """Async version of list."""
        return self.list(config)

    # Helper methods for testing/inspection
    def get_sessions(self) -> list[dict[str, Any]]:
        """Get all sessions in the checkpointer."""
        sessions = []
        for file_path in self.sessions_dir.glob("*.json"):
            session_data = self._load_session_file(file_path)
            if session_data:
                sessions.append(session_data)
        return sessions

    def export_session(self, thread_id: str) -> Optional[dict[str, Any]]:
        """Export a session as a dict for inspection."""
        path = self._get_session_path(thread_id)
        return self._load_session_file(path)

    def import_session(self, session_data: dict[str, Any]) -> None:
        """Import a session from a dict."""
        if "thread_id" not in session_data:
            raise ValueError("session_data must have thread_id")

        path = self._get_session_path(session_data["thread_id"])
        self._save_session_file(path, session_data)
