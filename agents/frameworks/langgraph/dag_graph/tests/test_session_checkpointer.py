"""Tests for session checkpointer with .doc_sessions directory storage."""

import json
import tempfile
from pathlib import Path

import pytest

from src.engine.session_checkpointer import SessionCheckpointer


def test_session_checkpointer_initialization():
    """Test that checkpointer initializes .doc_sessions directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpointer = SessionCheckpointer(sessions_dir=tmpdir)
        assert Path(tmpdir).exists()
        assert Path(tmpdir).is_dir()


def test_session_checkpointer_put_and_get():
    """Test saving and retrieving a checkpoint."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpointer = SessionCheckpointer(sessions_dir=tmpdir)

        config = {
            "configurable": {
                "thread_id": "user-123:session-456",
                "checkpoint_id": "cp-001",
            }
        }

        checkpoint = {
            "values": {
                "current_state": "validate",
                "turn_number": 1,
                "conversation_history": [],
            }
        }

        metadata = {
            "checkpoint_id": "cp-001",
            "ts_created": "2024-01-01T00:00:00",
        }

        # Save checkpoint
        result = checkpointer.put(config, checkpoint, metadata)
        assert result["configurable"]["thread_id"] == "user-123:session-456"

        # Retrieve checkpoint
        retrieved = checkpointer.get(config)
        assert retrieved is not None
        assert retrieved["values"]["current_state"] == "validate"
        assert retrieved["values"]["turn_number"] == 1


def test_session_checkpointer_get_tuple():
    """Test retrieving a checkpoint tuple."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpointer = SessionCheckpointer(sessions_dir=tmpdir)

        config = {"configurable": {"thread_id": "test-thread-1"}}
        checkpoint = {"values": {"data": "test_value"}}
        metadata = {"checkpoint_id": "cp-001"}

        checkpointer.put(config, checkpoint, metadata)

        # Get tuple
        tuple_result = checkpointer.get_tuple(config)
        assert tuple_result is not None
        assert tuple_result.checkpoint["values"]["data"] == "test_value"
        assert tuple_result.config["configurable"]["thread_id"] == "test-thread-1"


def test_session_checkpointer_list():
    """Test listing all checkpoints for a thread."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpointer = SessionCheckpointer(sessions_dir=tmpdir)

        config = {"configurable": {"thread_id": "test-thread-2"}}

        # Save multiple checkpoints
        for i in range(3):
            checkpoint = {"values": {"iteration": i}}
            metadata = {"checkpoint_id": f"cp-{i:03d}"}
            checkpointer.put(config, checkpoint, metadata)

        # List all
        checkpoints = checkpointer.list(config)
        assert len(checkpoints) == 3


def test_session_checkpointer_delete_thread():
    """Test deleting a thread's session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpointer = SessionCheckpointer(sessions_dir=tmpdir)

        config = {"configurable": {"thread_id": "test-thread-3"}}
        checkpoint = {"values": {"data": "to_delete"}}
        metadata = {"checkpoint_id": "cp-001"}

        # Save
        checkpointer.put(config, checkpoint, metadata)
        assert checkpointer.get(config) is not None

        # Delete
        checkpointer.delete_thread(config)
        assert checkpointer.get(config) is None


def test_session_checkpointer_thread_id_sanitization():
    """Test that thread_id with special characters is sanitized."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpointer = SessionCheckpointer(sessions_dir=tmpdir)

        # Thread ID with colons and special chars
        config = {"configurable": {"thread_id": "user:123/session:456"}}
        checkpoint = {"values": {"test": "value"}}
        metadata = {"checkpoint_id": "cp-001"}

        checkpointer.put(config, checkpoint, metadata)

        # Should be able to retrieve
        retrieved = checkpointer.get(config)
        assert retrieved is not None
        assert retrieved["values"]["test"] == "value"

        # Check file was created with sanitized name
        files = list(Path(tmpdir).glob("*.json"))
        assert len(files) == 1
        assert "user_123_session_456" in str(files[0])


def test_session_checkpointer_export_import():
    """Test exporting and importing sessions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpointer = SessionCheckpointer(sessions_dir=tmpdir)

        config = {"configurable": {"thread_id": "test-thread-4"}}
        checkpoint = {"values": {"important": "data"}}
        metadata = {"checkpoint_id": "cp-001"}

        checkpointer.put(config, checkpoint, metadata)

        # Export
        exported = checkpointer.export_session("test-thread-4")
        assert exported is not None
        assert exported["thread_id"] == "test-thread-4"

        # Create new checkpointer and import
        checkpointer2 = SessionCheckpointer(sessions_dir=Path(tmpdir) / "new")
        checkpointer2.import_session(exported)

        # Verify imported data
        retrieved = checkpointer2.get(config)
        assert retrieved["values"]["important"] == "data"


def test_session_checkpointer_get_sessions():
    """Test retrieving all sessions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpointer = SessionCheckpointer(sessions_dir=tmpdir)

        # Create multiple sessions
        for i in range(3):
            config = {"configurable": {"thread_id": f"thread-{i}"}}
            checkpoint = {"values": {"index": i}}
            metadata = {"checkpoint_id": "cp-001"}
            checkpointer.put(config, checkpoint, metadata)

        # Get all sessions
        sessions = checkpointer.get_sessions()
        assert len(sessions) == 3
        thread_ids = {s["thread_id"] for s in sessions}
        assert thread_ids == {"thread-0", "thread-1", "thread-2"}


def test_session_checkpointer_persistence():
    """Test that sessions persist across checkpointer instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # First instance
        checkpointer1 = SessionCheckpointer(sessions_dir=tmpdir)
        config = {"configurable": {"thread_id": "persist-test"}}
        checkpoint = {"values": {"persisted": True}}
        metadata = {"checkpoint_id": "cp-001"}
        checkpointer1.put(config, checkpoint, metadata)

        # Second instance
        checkpointer2 = SessionCheckpointer(sessions_dir=tmpdir)
        retrieved = checkpointer2.get(config)

        # Should retrieve the data saved by first instance
        assert retrieved is not None
        assert retrieved["values"]["persisted"] is True


def test_session_checkpointer_json_format():
    """Test that sessions are stored as readable JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpointer = SessionCheckpointer(sessions_dir=tmpdir)

        config = {"configurable": {"thread_id": "json-test"}}
        checkpoint = {"values": {"key": "value"}}
        metadata = {"checkpoint_id": "cp-001"}

        checkpointer.put(config, checkpoint, metadata)

        # Find the JSON file (thread_id gets sanitized)
        files = list(Path(tmpdir).glob("*.json"))
        assert len(files) == 1

        with open(files[0], "r") as f:
            data = json.load(f)

        # Verify structure
        assert "thread_id" in data
        assert "checkpoints" in data
        assert "cp-001" in data["checkpoints"]
        assert data["checkpoints"]["cp-001"]["values"]["key"] == "value"
