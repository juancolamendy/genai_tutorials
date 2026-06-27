# JsonCheckpointer: Directory-Based Session Storage

The `JsonCheckpointer` stores LangGraph sessions as JSON files in the `.doc_sessions` directory, matching Agno's session storage pattern while maintaining full LangGraph `BaseCheckpointSaver` interface compatibility.

## Overview

- **Storage**: JSON files in `.doc_sessions/` directory
- **One file per thread**: Thread ID becomes sanitized filename
- **Human-readable format**: JSON for easy inspection and debugging
- **Checkpoint history**: Multiple checkpoints per session
- **Metadata tracking**: Created/updated timestamps, checkpoint IDs

## Quick Start

### Basic Usage

```python
from src.workflow.workflow import run_pipeline

# Sessions automatically stored in .doc_sessions/
result = run_pipeline(
    document_id="doc-001",
    timeout_seconds=300,
    sessions_dir=".doc_sessions"  # Default
)
```

### Resume a Session

```python
from src.engine.session_checkpointer import JsonCheckpointer

checkpointer = JsonCheckpointer(sessions_dir=".doc_sessions")

# Load latest checkpoint for a session
config = {"configurable": {"thread_id": "user-123:session-456"}}
checkpoint_tuple = checkpointer.get_tuple(config)

if checkpoint_tuple:
    state = checkpoint_tuple.checkpoint["values"]
    print(f"Resumed at state: {state['current_state']}")
    print(f"Turn number: {state.get('turn_number', 0)}")
```

### List All Sessions

```python
checkpointer = JsonCheckpointer(sessions_dir=".doc_sessions")

sessions = checkpointer.get_sessions()
for session in sessions:
    print(f"Thread: {session['thread_id']}")
    print(f"Created: {session['created_at']}")
    print(f"Latest checkpoint: {session['latest_checkpoint_id']}")
```

## File Structure

### Directory Layout

```
project_root/
└── .doc_sessions/
    ├── user_123_session_456.json
    ├── user_789_session_012.json
    ├── process_doc_001.json
    └── ...
```

### Session File Format

```json
{
  "thread_id": "user-123:session-456",
  "created_at": "2024-01-01T10:30:00",
  "updated_at": "2024-01-01T10:35:42",
  "latest_checkpoint_id": "1704110142.5678",
  "checkpoints": {
    "1704110142.5678": {
      "values": {
        "current_state": "validate",
        "turn_number": 1,
        "conversation_history": [
          {
            "role": "user",
            "content": "Process this document",
            "turn_number": 1
          }
        ],
        "document_id": "doc-001",
        "retry_count": 0,
        "audit_trail": ["init", "fetch OK"],
        "raw_data": {
          "id": "doc-001",
          "content": "Lorem ipsum...",
          "schema_version": "2.1"
        },
        "validated_data": null,
        "enriched_data": null
      },
      "metadata": {
        "checkpoint_id": "1704110142.5678"
      },
      "ts_created": "2024-01-01T10:30:15"
    }
  },
  "metadata": {
    "checkpoint_id": "1704110142.5678"
  }
}
```

## API Reference

### JsonCheckpointer Class

#### `__init__(sessions_dir: str = ".doc_sessions")`

Initialize checkpointer with a directory.

```python
checkpointer = JsonCheckpointer(sessions_dir=".doc_sessions")
```

#### `put(config, checkpoint, metadata) -> RunnableConfig`

Save a checkpoint for a thread.

```python
config = {"configurable": {"thread_id": "user-123:session-456"}}
checkpoint = {"values": {"current_state": "validate", ...}}
metadata = {"checkpoint_id": "cp-001"}

result = checkpointer.put(config, checkpoint, metadata)
# Returns: {"configurable": {"thread_id": "...", "checkpoint_id": "..."}}
```

#### `get(config) -> Checkpoint | None`

Load the latest checkpoint for a thread.

```python
config = {"configurable": {"thread_id": "user-123:session-456"}}
checkpoint = checkpointer.get(config)

if checkpoint:
    state = checkpoint["values"]
```

#### `get_tuple(config) -> CheckpointTuple | None`

Load checkpoint with metadata.

```python
tuple_result = checkpointer.get_tuple(config)

if tuple_result:
    config, checkpoint, metadata = tuple_result.config, tuple_result.checkpoint, tuple_result.metadata
```

#### `list(config) -> list[CheckpointTuple]`

List all checkpoints for a thread (in order).

```python
checkpoints = checkpointer.list(config)
for cp in checkpoints:
    print(f"Checkpoint ID: {cp.config['configurable']['checkpoint_id']}")
    print(f"State: {cp.checkpoint['values']['current_state']}")
```

#### `delete_thread(config) -> None`

Delete all checkpoints for a thread.

```python
checkpointer.delete_thread(config)  # Removes the JSON file
```

#### `get_sessions() -> list[dict]`

Get all sessions in the checkpointer.

```python
sessions = checkpointer.get_sessions()
for session in sessions:
    print(f"Thread: {session['thread_id']}")
    print(f"Checkpoints: {len(session['checkpoints'])}")
```

#### `export_session(thread_id: str) -> dict | None`

Export a session as a dictionary.

```python
session_dict = checkpointer.export_session("user-123:session-456")
if session_dict:
    # Do something with the dict (backup, analyze, etc)
    print(json.dumps(session_dict, indent=2))
```

#### `import_session(session_data: dict) -> None`

Import a session from a dictionary.

```python
# Load from backup or another source
session_dict = load_backup()

checkpointer.import_session(session_dict)
```

## Examples

### Multi-Turn Conversation with Resume

```python
from src.workflow.workflow import run_pipeline
from src.engine.session_checkpointer import JsonCheckpointer

user_id = "user-123"
session_id = "session-456"
sessions_dir = ".doc_sessions"

# Turn 1: Start processing
print("Turn 1: Start processing")
result1 = run_pipeline(
    document_id="doc-001",
    thread_id=f"{user_id}:{session_id}",
    sessions_dir=sessions_dir
)
print(f"State: {result1['current_state']}")

# Inspect session file
checkpointer = JsonCheckpointer(sessions_dir=sessions_dir)
session = checkpointer.export_session(f"{user_id}:{session_id}")
print(f"Session saved with {len(session['checkpoints'])} checkpoints")

# Turn 2: Resume and continue
print("\nTurn 2: Resume and continue")
config = {"configurable": {"thread_id": f"{user_id}:{session_id}"}}
checkpoint_tuple = checkpointer.get_tuple(config)

if checkpoint_tuple:
    previous_state = checkpoint_tuple.checkpoint["values"]
    print(f"Resumed at state: {previous_state['current_state']}")
    
    # Continue processing with the same session_id
    result2 = run_pipeline(
        document_id=previous_state["document_id"],
        thread_id=f"{user_id}:{session_id}",
        sessions_dir=sessions_dir
    )
    print(f"New state: {result2['current_state']}")
```

### Backup and Restore Sessions

```python
import json
from pathlib import Path
from src.engine.session_checkpointer import JsonCheckpointer

checkpointer = JsonCheckpointer(sessions_dir=".doc_sessions")

# Backup all sessions
backup_dir = Path("backups/2024-01-01")
backup_dir.mkdir(parents=True, exist_ok=True)

for session in checkpointer.get_sessions():
    thread_id = session["thread_id"]
    backup_path = backup_dir / f"{thread_id}.json"
    
    with open(backup_path, "w") as f:
        json.dump(session, f, indent=2)
    
    print(f"Backed up: {thread_id}")

# Restore from backup
for backup_file in backup_dir.glob("*.json"):
    with open(backup_file, "r") as f:
        session_data = json.load(f)
    
    checkpointer.import_session(session_data)
    print(f"Restored: {session_data['thread_id']}")
```

### Thread ID Sanitization

Thread IDs can contain special characters that are sanitized for filesystem compatibility:

```python
checkpointer = JsonCheckpointer(sessions_dir=".doc_sessions")

# These thread IDs all work:
thread_ids = [
    "user:123:session:456",      # Colons become underscores
    "user/123/session/456",      # Slashes become underscores
    "user@domain.com:session-1", # @ and . are preserved
]

for thread_id in thread_ids:
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint = {"values": {"test": "data"}}
    metadata = {"checkpoint_id": "cp-001"}
    
    checkpointer.put(config, checkpoint, metadata)

# Check resulting filenames
files = list(Path(".doc_sessions").glob("*.json"))
for f in files:
    print(f.name)
    # Output:
    # user_123_session_456.json
    # user_123_session_456.json (same as above)
    # user@domain.com_session_1.json
```

## Compatibility

### With LangGraph

The `JsonCheckpointer` implements the `BaseCheckpointSaver` interface from LangGraph, making it fully compatible with LangGraph's state persistence:

```python
from langgraph.graph import StateGraph
from src.engine.session_checkpointer import JsonCheckpointer

# Use with StateGraph
graph = StateGraph(MyState)
# ... build graph ...

checkpointer = JsonCheckpointer(sessions_dir=".doc_sessions")
compiled = graph.compile(checkpointer=checkpointer)

# Invoke with thread_id
result = compiled.invoke(
    initial_state,
    config={"configurable": {"thread_id": "user-123:session-456"}}
)
```

### With Agno

Session format is compatible with Agno's `.doc_sessions/` directory pattern:

```
.doc_sessions/
├── agno_sessions.json (Agno format)
├── user_123_session_456.json (LangGraph format)
└── ...
```

## Performance Considerations

- **File I/O**: Each checkpoint triggers a file write (consider batch operations for high throughput)
- **File Size**: Session files grow with conversation history (consider archival policies)
- **Directory Access**: All sessions in one directory (OK for typical workloads, consider sharding for massive scale)

## Troubleshooting

### Session file not created

Check that the sessions directory is writable:

```python
from pathlib import Path
sessions_dir = Path(".doc_sessions")
print(f"Writable: {sessions_dir.is_dir() and os.access(sessions_dir, os.W_OK)}")
```

### Thread ID not found

Thread IDs are case-sensitive and undergo sanitization:

```python
# Original
thread_id = "user:123:session:456"

# Becomes filename
# user_123_session_456.json

# When retrieving, use the original thread_id (not the filename)
config = {"configurable": {"thread_id": "user:123:session:456"}}
```

### Large session files

Trim conversation history or archive old sessions:

```python
# Manually trim history in state before saving
state["conversation_history"] = state["conversation_history"][-10:]  # Keep last 10
```

## Migration

### From SqliteCheckpointer

To migrate from SQLite to directory-based sessions:

```python
from src.engine.checkpointing import SqliteCheckpointer
from src.engine.session_checkpointer import JsonCheckpointer

# Export from SQLite
sqlite_cp = SqliteCheckpointer("old.db")
threads = ["user-1:session-1", "user-2:session-2"]

# Import into JsonCheckpointer
dir_cp = JsonCheckpointer(sessions_dir=".doc_sessions")

for thread_id in threads:
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint = sqlite_cp.get(config)
    if checkpoint:
        # Note: metadata may be lost; see SqliteCheckpointer docs
        dir_cp.put(config, checkpoint, {})
```

## References

- [LangGraph Checkpointing](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [BaseCheckpointSaver API](https://langchain-ai.github.io/langgraph/reference/checkpoint/)
- [Agno Documentation](https://docs.agno.com/)
