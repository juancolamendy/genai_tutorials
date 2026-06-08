"""Tests for MarkdownMemoryDb — a duck-typed memory store compatible
with agno's MemoryTools(db=...) interface.
"""

import pytest
from memory_db import MarkdownMemoryDb
from agno.memory import UserMemory


@pytest.fixture
def db(tmp_path):
    return MarkdownMemoryDb(memory_dir=str(tmp_path))


def _make_memory(memory_id: str, content: str, user_id: str = 'u1') -> UserMemory:
    return UserMemory(memory=content, memory_id=memory_id, user_id=user_id)


class TestMarkdownMemoryDbSaveLoad:
    """Core save/load behaviour."""

    def test_save_creates_md_file(self, db, tmp_path):
        db.save('user-prefs', 'I like Python')
        assert (tmp_path / 'user-prefs.md').exists()

    def test_save_and_load_roundtrip(self, db, tmp_path):
        db.save('user-prefs', 'I like Python')
        content = (tmp_path / 'user-prefs.md').read_text(encoding='utf-8')
        assert 'I like Python' in content

    def test_save_overwrites_existing(self, db, tmp_path):
        db.save('key', 'old content')
        db.save('key', 'new content')
        content = (tmp_path / 'key.md').read_text(encoding='utf-8')
        assert 'new content' in content
        assert 'old content' not in content


class TestMarkdownMemoryDbSearch:
    """Keyword search across .md files."""

    def test_search_returns_matching_file(self, db):
        db.save('python-notes', 'I like Python and pip')
        db.save('java-notes', 'I use Java at work')
        results = db.search('Python')
        assert any('python-notes' in r for r in results)

    def test_search_case_insensitive(self, db):
        db.save('notes', 'I like PYTHON')
        results = db.search('python')
        assert len(results) > 0

    def test_search_returns_empty_list_for_no_match(self, db):
        db.save('notes', 'I like Python')
        results = db.search('ruby')
        assert results == []

    def test_search_matches_multiple_words(self, db):
        db.save('notes', 'Python and Django')
        results = db.search('Python Django')
        assert len(results) > 0

    def test_empty_memory_dir_search_returns_empty(self, db):
        results = db.search('anything')
        assert results == []


class TestMarkdownMemoryDbAdapter:
    """Adapter methods matching MemoryTools' self.db call sites:
      - upsert_user_memory(memory: UserMemory) -> UserMemory
      - get_user_memories(user_id=...) -> List[UserMemory]
      - get_user_memory(memory_id) -> UserMemory | None
      - delete_user_memory(memory_id) -> None
    """

    def test_upsert_user_memory_creates_file(self, db, tmp_path):
        mem = _make_memory('mem-001', 'remember this fact')
        result = db.upsert_user_memory(mem)
        assert (tmp_path / 'mem-001.md').exists()
        assert result is not None

    def test_upsert_user_memory_returns_user_memory(self, db):
        mem = _make_memory('mem-002', 'test content')
        result = db.upsert_user_memory(mem)
        assert isinstance(result, UserMemory)
        assert result.memory == 'test content'
        assert result.memory_id == 'mem-002'

    def test_upsert_user_memory_overwrites_existing(self, db):
        db.upsert_user_memory(_make_memory('mem-003', 'original'))
        updated = _make_memory('mem-003', 'updated')
        result = db.upsert_user_memory(updated)
        assert result.memory == 'updated'

    def test_get_user_memories_returns_all(self, db):
        db.upsert_user_memory(_make_memory('m1', 'fact one', user_id='alice'))
        db.upsert_user_memory(_make_memory('m2', 'fact two', user_id='alice'))
        memories = db.get_user_memories(user_id='alice')
        assert len(memories) >= 2
        assert all(isinstance(m, UserMemory) for m in memories)

    def test_get_user_memories_empty_dir_returns_list(self, db):
        memories = db.get_user_memories(user_id='nobody')
        assert isinstance(memories, list)
        assert memories == []

    def test_get_user_memory_returns_correct_memory(self, db):
        db.upsert_user_memory(_make_memory('find-me', 'specific content'))
        result = db.get_user_memory('find-me')
        assert result is not None
        assert isinstance(result, UserMemory)
        assert result.memory == 'specific content'
        assert result.memory_id == 'find-me'

    def test_get_user_memory_returns_none_for_missing(self, db):
        result = db.get_user_memory('does-not-exist')
        assert result is None

    def test_delete_user_memory_removes_file(self, db, tmp_path):
        db.upsert_user_memory(_make_memory('to-delete', 'ephemeral'))
        assert (tmp_path / 'to-delete.md').exists()
        db.delete_user_memory('to-delete')
        assert not (tmp_path / 'to-delete.md').exists()

    def test_delete_user_memory_nonexistent_does_not_raise(self, db):
        # Should not raise even if memory does not exist
        db.delete_user_memory('ghost-id')

    def test_get_user_memories_no_filter_returns_all(self, db):
        db.upsert_user_memory(_make_memory('x1', 'content', user_id='bob'))
        db.upsert_user_memory(_make_memory('x2', 'content', user_id='carol'))
        all_memories = db.get_user_memories()
        assert len(all_memories) >= 2
