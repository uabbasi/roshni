"""Tests for agent session persistence."""

from __future__ import annotations

import threading
import time

import pytest

from roshni.agent.session import JSONLSessionStore, Session, SessionStore, Turn


@pytest.mark.smoke
def test_turn_defaults():
    """Turn auto-populates timestamp and metadata."""
    turn = Turn(role="user", content="hello")
    assert turn.role == "user"
    assert turn.content == "hello"
    assert turn.timestamp  # non-empty ISO string
    assert "T" in turn.timestamp
    assert turn.metadata == {}


@pytest.mark.smoke
def test_session_defaults():
    """Session auto-populates id and started."""
    session = Session()
    assert len(session.id) == 8
    assert all(c in "0123456789abcdef" for c in session.id)
    assert "T" in session.started
    assert session.ended is None
    assert session.turns == []


@pytest.mark.smoke
def test_jsonl_store_implements_protocol():
    """JSONLSessionStore satisfies the SessionStore protocol."""
    assert isinstance(JSONLSessionStore, type)
    assert issubclass(JSONLSessionStore, SessionStore) or isinstance(JSONLSessionStore("/tmp/fake"), SessionStore)


@pytest.mark.smoke
def test_create_save_load_roundtrip(tmp_path):
    """Create a session, save turns, load back — everything matches."""
    store = JSONLSessionStore(tmp_path / "sessions")
    session = Session(agent_name="test-agent", channel="cli")

    store.create_session(session)
    t1 = Turn(role="user", content="hello")
    t2 = Turn(role="assistant", content="hi there", metadata={"model": "gpt-4"})
    store.save_turn(session.id, t1)
    store.save_turn(session.id, t2)

    loaded = store.load_session(session.id)
    assert loaded is not None
    assert loaded.id == session.id
    assert loaded.agent_name == "test-agent"
    assert loaded.channel == "cli"
    assert len(loaded.turns) == 2
    assert loaded.turns[0].role == "user"
    assert loaded.turns[0].content == "hello"
    assert loaded.turns[1].role == "assistant"
    assert loaded.turns[1].metadata["model"] == "gpt-4"


@pytest.mark.smoke
def test_list_sessions_filter_agent_name(tmp_path):
    """list_sessions filters by agent_name."""
    store = JSONLSessionStore(tmp_path / "sessions")

    s1 = Session(agent_name="alice", channel="telegram")
    s2 = Session(agent_name="bob", channel="telegram")
    s3 = Session(agent_name="alice", channel="cli")
    store.create_session(s1)
    store.create_session(s2)
    store.create_session(s3)

    alice_sessions = store.list_sessions(agent_name="alice")
    assert len(alice_sessions) == 2
    assert all(s.agent_name == "alice" for s in alice_sessions)

    bob_sessions = store.list_sessions(agent_name="bob")
    assert len(bob_sessions) == 1
    assert bob_sessions[0].agent_name == "bob"


@pytest.mark.smoke
def test_list_sessions_filter_channel(tmp_path):
    """list_sessions filters by channel."""
    store = JSONLSessionStore(tmp_path / "sessions")

    s1 = Session(agent_name="a", channel="telegram")
    s2 = Session(agent_name="b", channel="cli")
    store.create_session(s1)
    store.create_session(s2)

    tg = store.list_sessions(channel="telegram")
    assert len(tg) == 1
    assert tg[0].channel == "telegram"


@pytest.mark.smoke
def test_list_sessions_since_filter(tmp_path):
    """list_sessions filters by since timestamp."""
    store = JSONLSessionStore(tmp_path / "sessions")

    s1 = Session(agent_name="a", started="2024-01-01T00:00:00+00:00")
    s2 = Session(agent_name="b", started="2024-06-01T00:00:00+00:00")
    s3 = Session(agent_name="c", started="2025-01-01T00:00:00+00:00")
    store.create_session(s1)
    store.create_session(s2)
    store.create_session(s3)

    recent = store.list_sessions(since="2024-06-01T00:00:00+00:00")
    assert len(recent) == 2
    names = {s.agent_name for s in recent}
    assert names == {"b", "c"}


@pytest.mark.smoke
def test_close_session(tmp_path):
    """close_session sets the ended timestamp."""
    store = JSONLSessionStore(tmp_path / "sessions")
    session = Session(agent_name="closer")
    store.create_session(session)
    store.save_turn(session.id, Turn(role="user", content="bye"))

    assert store.load_session(session.id).ended is None
    store.close_session(session.id)

    reloaded = store.load_session(session.id)
    assert reloaded is not None
    assert reloaded.ended is not None
    assert "T" in reloaded.ended
    # Turns should survive the rewrite
    assert len(reloaded.turns) == 1

    # Index should also be updated
    sessions = store.list_sessions(agent_name="closer")
    assert len(sessions) == 1
    assert sessions[0].ended is not None


@pytest.mark.smoke
def test_load_missing_session(tmp_path):
    """Loading a non-existent session returns None."""
    store = JSONLSessionStore(tmp_path / "sessions")
    assert store.load_session("nonexistent") is None


@pytest.mark.smoke
def test_concurrent_writes(tmp_path):
    """Concurrent turn writes don't lose data."""
    store = JSONLSessionStore(tmp_path / "sessions")
    session = Session(agent_name="concurrent")
    store.create_session(session)

    num_turns = 20
    errors: list[Exception] = []

    def write_turn(idx: int) -> None:
        try:
            store.save_turn(session.id, Turn(role="user", content=f"msg-{idx}"))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=write_turn, args=(i,)) for i in range(num_turns)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Errors during concurrent writes: {errors}"
    loaded = store.load_session(session.id)
    assert loaded is not None
    assert len(loaded.turns) == num_turns
    contents = {t.content for t in loaded.turns}
    assert contents == {f"msg-{i}" for i in range(num_turns)}


@pytest.mark.smoke
def test_concurrent_close_and_writes(tmp_path):
    """close_session should not corrupt files while turns are being appended."""
    store = JSONLSessionStore(tmp_path / "sessions")
    session = Session(agent_name="close-race")
    store.create_session(session)

    writes = 50
    done = threading.Event()
    errors: list[Exception] = []

    def writer() -> None:
        try:
            for i in range(writes):
                store.save_turn(session.id, Turn(role="user", content=f"turn-{i}"))
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)
        finally:
            done.set()

    thread = threading.Thread(target=writer)
    thread.start()
    time.sleep(0.01)
    store.close_session(session.id)
    thread.join(timeout=5)

    assert done.is_set()
    assert not errors

    loaded = store.load_session(session.id)
    assert loaded is not None
    assert loaded.ended is not None
    assert len(loaded.turns) == writes
