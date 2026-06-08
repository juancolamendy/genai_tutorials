"""Tests for JsonlAgentDb storage.

Uses the actual Agno API:
  - get_session(session_id, session_type, user_id) -> Optional[AgentSession]
  - upsert_session(session) -> Optional[AgentSession]
  - delete_session(session_id, user_id) -> bool

AgentSession is a dataclass from agno.session.agent (NOT agno.db.session).
"""


import pytest

from storage import JsonlAgentDb


@pytest.fixture
def db(tmp_path):
    return JsonlAgentDb(sessions_dir=str(tmp_path))


class TestJsonlAgentDb:

    def test_get_session_returns_none_for_missing_session(self, db):
        from agno.db.base import SessionType

        result = db.get_session(
            session_id='s1', session_type=SessionType.AGENT, user_id='u1'
        )
        assert result is None

    def test_upsert_and_get_session_roundtrip(self, db):
        from agno.db.base import SessionType
        from agno.session.agent import AgentSession

        session = AgentSession(session_id='s1', user_id='u1')
        db.upsert_session(session)
        loaded = db.get_session(
            session_id='s1', session_type=SessionType.AGENT, user_id='u1'
        )
        assert loaded is not None
        assert loaded.session_id == 's1'
        assert loaded.user_id == 'u1'

    def test_upsert_overwrites_existing(self, db, tmp_path):
        from agno.db.base import SessionType
        from agno.session.agent import AgentSession

        s1 = AgentSession(session_id='s1', user_id='u1')
        db.upsert_session(s1)
        s2 = AgentSession(session_id='s1', user_id='u1')
        db.upsert_session(s2)

        second_content = (tmp_path / 'u1_s1.jsonl').read_text()
        loaded = db.get_session(
            session_id='s1', session_type=SessionType.AGENT, user_id='u1'
        )
        assert loaded is not None
        assert loaded.session_id == 's1'
        lines = [line for line in second_content.splitlines() if line.strip()]
        assert len(lines) == 1, (
            f'Expected single-line JSON, got {len(lines)} lines'
        )

    def test_get_sessions_returns_sessions_for_user(self, db):
        from agno.db.base import SessionType
        from agno.session.agent import AgentSession

        db.upsert_session(AgentSession(session_id='s1', user_id='u1'))
        db.upsert_session(AgentSession(session_id='s2', user_id='u1'))
        db.upsert_session(AgentSession(session_id='s3', user_id='u2'))

        sessions = db.get_sessions(
            session_type=SessionType.AGENT, user_id='u1'
        )
        ids = {s.session_id for s in sessions}
        assert ids == {'s1', 's2'}

    def test_get_sessions_none_user_returns_all(self, db):
        from agno.db.base import SessionType
        from agno.session.agent import AgentSession

        db.upsert_session(AgentSession(session_id='s1', user_id='u1'))
        db.upsert_session(AgentSession(session_id='s2', user_id='u2'))

        sessions = db.get_sessions(
            session_type=SessionType.AGENT, user_id=None
        )
        ids = {s.session_id for s in sessions}
        assert ids == {'s1', 's2'}

    def test_delete_session_removes_file(self, db, tmp_path):
        from agno.session.agent import AgentSession

        db.upsert_session(AgentSession(session_id='s1', user_id='u1'))
        db.delete_session(session_id='s1', user_id='u1')
        path = tmp_path / 'u1_s1.jsonl'
        assert not path.exists()

    def test_delete_nonexistent_session_does_not_raise(self, db):
        db.delete_session(session_id='doesnt-exist', user_id=None)

    def test_session_file_named_userid_sessionid(self, db, tmp_path):
        from agno.session.agent import AgentSession

        db.upsert_session(AgentSession(session_id='mysession', user_id='myuser'))
        assert (tmp_path / 'myuser_mysession.jsonl').exists()

    def test_get_session_with_no_user_id(self, db):
        from agno.db.base import SessionType
        from agno.session.agent import AgentSession

        db.upsert_session(AgentSession(session_id='s1', user_id=None))
        result = db.get_session(
            session_id='s1', session_type=SessionType.AGENT, user_id=None
        )
        assert result is not None

    def test_upsert_returns_session(self, db):
        from agno.session.agent import AgentSession

        session = AgentSession(session_id='s1', user_id='u1')
        result = db.upsert_session(session)
        assert result is not None
        assert result.session_id == 's1'

    def test_delete_session_returns_true_when_found(self, db):
        from agno.session.agent import AgentSession

        db.upsert_session(AgentSession(session_id='s1', user_id='u1'))
        result = db.delete_session(session_id='s1', user_id='u1')
        assert result is True

    def test_delete_session_returns_false_when_not_found(self, db):
        result = db.delete_session(session_id='no-such', user_id=None)
        assert result is False

    def test_table_exists_returns_bool(self, db):
        result = db.table_exists('sessions')
        assert isinstance(result, bool)

    def test_get_session_cross_user_lookup_by_session_id(self, db):
        """get_session should find a session even when user_id differs from file prefix."""
        from agno.db.base import SessionType
        from agno.session.agent import AgentSession

        db.upsert_session(AgentSession(session_id='s1', user_id='u1'))
        # Lookup without user_id — should still find via content scan
        result = db.get_session(
            session_id='s1', session_type=SessionType.AGENT, user_id=None
        )
        assert result is not None
        assert result.session_id == 's1'
