"""Markdown-file-backed memory store compatible with agno's MemoryTools.

Stores each UserMemory as a JSON-serialised .md file under memory_dir/.
Compatible with bot1's memory/*.md directory layout for long-term memories.

Note: 'memory_dir' is for long-term agent memory (save_memory / memory_search).
This is separate from workspace/memory/ daily logs that are read into the
system prompt at startup.

Interface required by MemoryTools(db=...):
    - upsert_user_memory(memory: UserMemory) -> Optional[UserMemory]
    - get_user_memories(user_id, ...) -> List[UserMemory]
    - get_user_memory(memory_id, ...) -> Optional[UserMemory]
    - delete_user_memory(memory_id, ...) -> None
"""

import json
import os
from typing import List, Optional, Any, Dict, Union

from agno.memory import UserMemory


class MarkdownMemoryDb:
    """Duck-typed memory store backed by Markdown (JSON) files.

    Passed to ``MemoryTools(db=MarkdownMemoryDb(memory_dir))``.
    Each memory is persisted as ``<memory_dir>/<memory_id>.md`` containing
    the JSON representation of the ``UserMemory`` dataclass.
    """

    def __init__(self, memory_dir: str = './memory') -> None:
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filepath(self, memory_id: str) -> str:
        return os.path.join(self.memory_dir, f'{memory_id}.md')

    def _write(self, memory: UserMemory) -> None:
        """Serialise a UserMemory to its .md file."""
        filepath = self._filepath(memory.memory_id)  # type: ignore[arg-type]
        with open(filepath, 'w', encoding='utf-8') as fh:
            json.dump(memory.to_dict(), fh, ensure_ascii=False, indent=2)

    def _read(self, memory_id: str) -> Optional[UserMemory]:
        """Deserialise a UserMemory from its .md file, or None if absent."""
        filepath = self._filepath(memory_id)
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, 'r', encoding='utf-8') as fh:
                data: Dict[str, Any] = json.load(fh)
            # Ensure memory_id is present (may have been omitted in to_dict)
            if 'memory_id' not in data:
                data['memory_id'] = memory_id
            return UserMemory.from_dict(data)
        except Exception:
            return None

    def _all_memory_ids(self) -> List[str]:
        """Return all memory IDs from .md files in memory_dir."""
        try:
            return [
                fname[:-3]  # strip '.md'
                for fname in os.listdir(self.memory_dir)
                if fname.endswith('.md')
            ]
        except OSError:
            return []

    # ------------------------------------------------------------------
    # Convenience save/search API (mirrors bot1 helper pattern)
    # ------------------------------------------------------------------

    def save(self, key: str, content: str) -> str:
        """Save raw text content to memory/<key>.md. Returns the key."""
        filepath = self._filepath(key)
        with open(filepath, 'w', encoding='utf-8') as fh:
            fh.write(content)
        return key

    def search(self, query: str) -> List[str]:
        """Keyword search across all memory/*.md files.

        Returns a list of ``'--- key.md ---\\ncontent'`` strings for files
        whose text content contains any word in *query* (case-insensitive).
        """
        q = query.lower()
        results: List[str] = []
        for fname in os.listdir(self.memory_dir):
            if not fname.endswith('.md'):
                continue
            try:
                with open(os.path.join(self.memory_dir, fname), 'r', encoding='utf-8') as fh:
                    content = fh.read()
                if any(word in content.lower() for word in q.split()):
                    results.append(f'--- {fname} ---\n{content}')
            except Exception:
                continue
        return results

    # ------------------------------------------------------------------
    # MemoryTools adapter methods  (actual names from MemoryTools source)
    # ------------------------------------------------------------------

    def upsert_user_memory(
        self,
        memory: UserMemory,
        deserialize: Optional[bool] = True,
    ) -> Optional[Union[UserMemory, Dict[str, Any]]]:
        """Persist or overwrite a UserMemory. Returns the stored UserMemory."""
        if memory.memory_id is None:
            from uuid import uuid4
            memory.memory_id = str(uuid4())
        self._write(memory)
        return memory

    def get_user_memory(
        self,
        memory_id: str,
        deserialize: Optional[bool] = True,
        user_id: Optional[str] = None,
    ) -> Optional[Union[UserMemory, Dict[str, Any]]]:
        """Return a UserMemory by ID, or None if not found."""
        return self._read(memory_id)

    def get_user_memories(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        topics: Optional[List[str]] = None,
        search_content: Optional[str] = None,
        limit: Optional[int] = None,
        page: Optional[int] = None,
        deserialize: Optional[bool] = True,
    ) -> List[Union[UserMemory, Dict[str, Any]]]:
        """Return all stored UserMemory objects, optionally filtered by user_id."""
        memories: List[UserMemory] = []
        for memory_id in self._all_memory_ids():
            mem = self._read(memory_id)
            if mem is None:
                continue
            if user_id is not None and mem.user_id != user_id:
                continue
            memories.append(mem)
        if limit is not None:
            offset = ((page or 1) - 1) * limit
            memories = memories[offset: offset + limit]
        return memories  # type: ignore[return-value]

    def delete_user_memory(
        self,
        memory_id: str,
        user_id: Optional[str] = None,
    ) -> None:
        """Delete the .md file for the given memory_id. Silent if not found."""
        filepath = self._filepath(memory_id)
        try:
            os.remove(filepath)
        except FileNotFoundError:
            pass
