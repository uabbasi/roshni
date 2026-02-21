"""Session persistence — append-only JSONL storage for agent conversations.

Tracks conversation turns (user/assistant messages) with metadata and
provides a simple file-based store for session history.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from loguru import logger


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Turn:
    """A single conversational turn."""

    role: str
    content: str
    timestamp: str = field(default_factory=_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """An agent conversation session."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    agent_name: str = ""
    channel: str = ""
    started: str = field(default_factory=_now_iso)
    ended: str | None = None
    turns: list[Turn] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SessionStore(Protocol):
    """Storage backend for agent sessions."""

    def save_turn(self, session_id: str, turn: Turn) -> None: ...

    def create_session(self, session: Session) -> None: ...

    def load_session(self, session_id: str) -> Session | None: ...

    def list_sessions(
        self,
        *,
        agent_name: str | None = None,
        channel: str | None = None,
        since: str | None = None,
        limit: int = 50,
    ) -> list[Session]: ...

    def close_session(self, session_id: str) -> None: ...


class JSONLSessionStore:
    """Append-only JSONL file store for sessions.

    Layout::

        base_dir/
            _sessions.jsonl     # index — one JSON line per session (no turns)
            {session_id}.jsonl  # per-session file — header + turns
    """

    _INDEX = "_sessions.jsonl"
    _LOCKS_GUARD = threading.Lock()
    _PATH_LOCKS: dict[str, threading.RLock] = {}

    def __init__(self, base_dir: str | Path) -> None:
        self._base = Path(base_dir)
        os.makedirs(self._base, exist_ok=True)

    # -- public API ----------------------------------------------------------

    def create_session(self, session: Session) -> None:
        header = self._session_header(session)
        session_path = self._session_path(session.id)
        with self._lock_paths(self._index_path, session_path):
            self._append_unlocked(self._index_path, header)
            self._append_unlocked(session_path, header)

    def save_turn(self, session_id: str, turn: Turn) -> None:
        path = self._session_path(session_id)
        with self._lock_paths(path):
            self._append_unlocked(path, asdict(turn))

    def load_session(self, session_id: str) -> Session | None:
        path = self._session_path(session_id)
        if not path.exists():
            return None
        with self._lock_paths(path):
            lines = self._read_lines_unlocked(path)
        if not lines:
            return None
        header = lines[0]
        turns: list[Turn] = []
        for i, line in enumerate(lines[1:], 2):
            try:
                turns.append(Turn(**line))
            except (TypeError, KeyError) as e:
                logger.warning(f"Skipping malformed turn at line {i} in session {session_id}: {e}")
        try:
            return Session(**header, turns=turns)
        except (TypeError, KeyError) as e:
            logger.error(f"Corrupted session header for {session_id}: {e}")
            return None

    def list_sessions(
        self,
        *,
        agent_name: str | None = None,
        channel: str | None = None,
        since: str | None = None,
        limit: int = 50,
    ) -> list[Session]:
        if not self._index_path.exists():
            return []
        with self._lock_paths(self._index_path):
            entries = self._read_lines_unlocked(self._index_path)
        if agent_name is not None:
            entries = [e for e in entries if e.get("agent_name") == agent_name]
        if channel is not None:
            entries = [e for e in entries if e.get("channel") == channel]
        if since is not None:
            entries = [e for e in entries if e.get("started", "") >= since]
        sessions = [Session(**e) for e in entries[-limit:]]
        return sessions

    def close_session(self, session_id: str) -> None:
        path = self._session_path(session_id)
        idx_path = self._index_path
        with self._lock_paths(path, idx_path):
            if not path.exists():
                return
            lines = self._read_lines_unlocked(path)
            if not lines:
                return
            header = lines[0]
            header["ended"] = _now_iso()
            lines[0] = header
            self._write_lines_unlocked(path, lines)

            if idx_path.exists():
                idx_lines = self._read_lines_unlocked(idx_path)
                for i, entry in enumerate(idx_lines):
                    if entry.get("id") == session_id:
                        entry["ended"] = header["ended"]
                        idx_lines[i] = entry
                        break
                self._write_lines_unlocked(idx_path, idx_lines)

    # -- internal helpers ----------------------------------------------------

    @property
    def _index_path(self) -> Path:
        return self._base / self._INDEX

    def _session_path(self, session_id: str) -> Path:
        return self._base / f"{session_id}.jsonl"

    @staticmethod
    def _session_header(session: Session) -> dict[str, Any]:
        """Session metadata dict (without turns)."""
        d = asdict(session)
        d.pop("turns", None)
        return d

    @staticmethod
    def _append_unlocked(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    @staticmethod
    def _read_lines_unlocked(path: Path) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        with open(path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping corrupted JSONL line {line_num} in {path}")
        return results

    @staticmethod
    def _write_lines_unlocked(path: Path, data: list[dict[str, Any]]) -> None:
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    @classmethod
    def _path_lock(cls, path: Path) -> threading.RLock:
        key = str(path.resolve())
        with cls._LOCKS_GUARD:
            lock = cls._PATH_LOCKS.get(key)
            if lock is None:
                lock = threading.RLock()
                cls._PATH_LOCKS[key] = lock
            return lock

    @classmethod
    @contextmanager
    def _lock_paths(cls, *paths: Path):
        unique_paths = sorted({str(p.resolve()) for p in paths})
        locks = [cls._path_lock(Path(p)) for p in unique_paths]
        for lock in locks:
            lock.acquire()
        try:
            yield
        finally:
            for lock in reversed(locks):
                lock.release()
