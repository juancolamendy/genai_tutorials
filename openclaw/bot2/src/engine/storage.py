"""JSONL-backed session storage for bot2.

Design decisions
----------------
- Does NOT inherit from ``agno.db.base.BaseDb``.

  BaseDb has 47 abstract methods spanning sessions, memories, traces, spans,
  evals, schedules, approvals, learnings, and more. Implementing all of them
  for a CLI bot that only needs session persistence would be impractical and
  would obscure the intent of this class. The Agent only calls three methods
  at runtime: ``get_session``, ``upsert_session``, and ``delete_session``.

- File naming: ``{user_id or 'anonymous'}_{session_id}.jsonl``.
  The ``.jsonl`` suffix matches the bot1 filename convention. The file
  contains exactly ONE JSON line (the latest AgentSession state). It is NOT
  an append-only log — each upsert overwrites the file.

- To swap backends, replace ``db=JsonlAgentDb(...)`` with
  ``db=SqliteDb(...)`` or ``db=PostgresDb(...)`` in main.py.

Lookup strategy
---------------
``get_session`` first tries the deterministic path
``{prefix}_{session_id}.jsonl``. If that file is absent it falls back to
scanning all ``.jsonl`` files and reading their JSON content to find a
matching ``session_id``. This handles the case where ``user_id`` is unknown
at lookup time. Scan is O(n) over session files — acceptable for a CLI bot.
"""

import json
import os
from typing import Optional, Union

from agno.db.base import SessionType
from agno.session.agent import AgentSession
from agno.session.team import TeamSession
from agno.session.workflow import WorkflowSession

_Session = Union[AgentSession, TeamSession, WorkflowSession]

_SESSION_CLS = {
    SessionType.AGENT: AgentSession,
    SessionType.TEAM: TeamSession,
    SessionType.WORKFLOW: WorkflowSession,
}


class JsonlAgentDb:
    """Single-JSON-per-session file store.

    Each session is persisted as a single-line JSON file:
    ``<sessions_dir>/<user_id or 'anonymous'>_<session_id>.jsonl``

    Args:
        sessions_dir: Directory where session files are stored. Created
            automatically if it does not exist.
    """

    def __init__(self, sessions_dir: str = './sessions') -> None:
        self.sessions_dir = sessions_dir
        os.makedirs(sessions_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prefix(self, user_id: Optional[str]) -> str:
        return user_id if user_id else 'anonymous'

    def _path(self, session_id: str, user_id: Optional[str]) -> str:
        return os.path.join(
            self.sessions_dir,
            f'{self._prefix(user_id)}_{session_id}.jsonl',
        )

    def _find_path(self, session_id: str) -> Optional[str]:
        """Locate the file for *session_id* regardless of user_id prefix.

        Reads JSON content rather than parsing filenames to avoid ambiguity
        when user_id or session_id contains underscores.
        """
        for fname in os.listdir(self.sessions_dir):
            if not fname.endswith('.jsonl'):
                continue
            path = os.path.join(self.sessions_dir, fname)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    line = f.readline().strip()
                if line and json.loads(line).get('session_id') == session_id:
                    return path
            except Exception:
                continue
        return None

    def _load(self, path: str, session_type: SessionType) -> Optional[_Session]:
        """Read and deserialize a session file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                line = f.readline().strip()
        except OSError:
            return None
        if not line:
            return None
        data = json.loads(line)
        cls = _SESSION_CLS.get(session_type, AgentSession)
        if hasattr(cls, 'from_dict'):
            return cls.from_dict(data)
        return cls(**data)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Public API (methods called by the Agno Agent at runtime)
    # ------------------------------------------------------------------

    def get_session(
        self,
        session_id: str,
        session_type: SessionType,
        user_id: Optional[str] = None,
        deserialize: Optional[bool] = True,
    ) -> Optional[_Session]:
        """Return the session for *session_id*, or ``None`` if not found.

        Args:
            session_id: The session identifier.
            session_type: ``SessionType.AGENT``, ``TEAM``, or ``WORKFLOW``.
            user_id: Optional owner of the session. When ``None`` the method
                falls back to a content-scan of all session files.
            deserialize: Kept for API parity with BaseDb; always honoured
                (session objects are always returned, never raw dicts).
        """
        path = self._path(session_id, user_id)
        if not os.path.exists(path):
            path = self._find_path(session_id)  # type: ignore[assignment]
        if not path or not os.path.exists(path):
            return None
        return self._load(path, session_type)

    def upsert_session(
        self,
        session: _Session,
        deserialize: Optional[bool] = True,
    ) -> Optional[_Session]:
        """Persist *session* to disk, overwriting any existing file.

        Args:
            session: The session object to store.
            deserialize: Kept for API parity with BaseDb; ignored.

        Returns:
            The same session that was passed in.
        """
        path = self._path(session.session_id, session.user_id)
        if hasattr(session, 'to_dict'):
            data = session.to_dict()
        else:
            from dataclasses import asdict
            data = asdict(session)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data) + '\n')
        return session

    def delete_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """Delete the session file for *session_id*.

        Args:
            session_id: The session identifier.
            user_id: Optional owner; used to build the deterministic path
                first. Falls back to a content-scan when ``None``.

        Returns:
            ``True`` if the file was deleted, ``False`` if it was not found.
        """
        path = self._path(session_id, user_id)
        if not os.path.exists(path):
            path = self._find_path(session_id)  # type: ignore[assignment]
        if path and os.path.exists(path):
            os.remove(path)
            return True
        return False

    # ------------------------------------------------------------------
    # Additional helpers used by agent CLI commands (list sessions, etc.)
    # ------------------------------------------------------------------

    def get_sessions(
        self,
        session_type: SessionType,
        user_id: Optional[str] = None,
        **_kwargs,
    ) -> list[_Session]:
        """Return all sessions, optionally filtered by *user_id*.

        Args:
            session_type: Used to pick the correct deserialiser.
            user_id: When provided, only sessions owned by this user are
                returned. When ``None``, all sessions are returned.
        """
        results: list[_Session] = []
        for fname in os.listdir(self.sessions_dir):
            if not fname.endswith('.jsonl'):
                continue
            path = os.path.join(self.sessions_dir, fname)
            session = self._load(path, session_type)
            if session is None:
                continue
            if user_id is None or session.user_id == user_id:
                results.append(session)
        return results

    # ------------------------------------------------------------------
    # Minimal stubs required so the Agent does not crash on start-up
    # ------------------------------------------------------------------

    def table_exists(self, table_name: str) -> bool:
        """Always returns ``True`` — file-based storage has no tables."""
        return True

    def to_dict(self) -> dict:
        """Serialise this db instance (used when Agent serialises itself)."""
        return {
            'type': 'JsonlAgentDb',
            'sessions_dir': self.sessions_dir,
        }
