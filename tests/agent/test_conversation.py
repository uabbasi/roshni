"""Tests for ConversationManager — per-chat history isolation."""

import threading

import pytest

from roshni.agent.conversation import ConversationManager


@pytest.mark.smoke
class TestConversationManager:
    def test_new_chat_returns_empty_history(self):
        cm = ConversationManager()
        history = cm.get_history("chat-1")
        assert history == []

    def test_separate_histories_per_chat(self):
        cm = ConversationManager()
        h1 = cm.get_history("chat-1")
        h2 = cm.get_history("chat-2")
        h1.append({"role": "user", "content": "hello from chat 1"})
        h2.append({"role": "user", "content": "hello from chat 2"})

        assert len(cm.get_history("chat-1")) == 1
        assert len(cm.get_history("chat-2")) == 1
        assert cm.get_history("chat-1")[0]["content"] == "hello from chat 1"
        assert cm.get_history("chat-2")[0]["content"] == "hello from chat 2"

    def test_save_history_persists(self):
        cm = ConversationManager()
        cm.save_history("chat-1", [{"role": "user", "content": "saved"}])
        assert cm.get_history("chat-1") == [{"role": "user", "content": "saved"}]

    def test_clear_history_single_chat(self):
        cm = ConversationManager()
        cm.save_history("chat-1", [{"role": "user", "content": "msg"}])
        cm.save_history("chat-2", [{"role": "user", "content": "msg"}])
        cm.clear_history("chat-1")

        assert cm.get_history("chat-1") == []
        assert len(cm.get_history("chat-2")) == 1

    def test_clear_nonexistent_chat_is_noop(self):
        cm = ConversationManager()
        cm.clear_history("nonexistent")  # Should not raise

    def test_lru_eviction(self):
        cm = ConversationManager(max_conversations=2)
        cm.save_history("chat-1", [{"role": "user", "content": "one"}])
        cm.save_history("chat-2", [{"role": "user", "content": "two"}])
        # Adding a third should evict the oldest (chat-1)
        cm.save_history("chat-3", [{"role": "user", "content": "three"}])

        assert cm.active_conversations == 2
        # chat-1 was evicted, get_history creates a fresh empty one
        assert cm.get_history("chat-1") == []

    def test_get_history_touches_lru(self):
        cm = ConversationManager(max_conversations=2)
        cm.save_history("chat-1", [{"role": "user", "content": "one"}])
        cm.save_history("chat-2", [{"role": "user", "content": "two"}])
        # Touch chat-1 so chat-2 becomes LRU
        cm.get_history("chat-1")
        # Adding chat-3 should evict chat-2 (not chat-1)
        cm.save_history("chat-3", [{"role": "user", "content": "three"}])

        assert cm.get_history("chat-1") == [{"role": "user", "content": "one"}]
        assert cm.get_history("chat-2") == []  # Evicted

    def test_session_id_tracking(self):
        cm = ConversationManager()
        assert cm.get_session_id("chat-1") is None
        cm.set_session_id("chat-1", "sess-abc")
        assert cm.get_session_id("chat-1") == "sess-abc"

    def test_clear_history_removes_session_id(self):
        cm = ConversationManager()
        cm.set_session_id("chat-1", "sess-abc")
        cm.clear_history("chat-1")
        assert cm.get_session_id("chat-1") is None

    def test_eviction_removes_session_id(self):
        cm = ConversationManager(max_conversations=1)
        cm.save_history("chat-1", [{"role": "user", "content": "one"}])
        cm.set_session_id("chat-1", "sess-abc")
        # Evict chat-1
        cm.save_history("chat-2", [{"role": "user", "content": "two"}])
        assert cm.get_session_id("chat-1") is None

    def test_active_conversations_count(self):
        cm = ConversationManager()
        assert cm.active_conversations == 0
        cm.get_history("chat-1")
        assert cm.active_conversations == 1
        cm.get_history("chat-2")
        assert cm.active_conversations == 2

    def test_unlimited_conversations(self):
        cm = ConversationManager(max_conversations=0)
        for i in range(200):
            cm.save_history(f"chat-{i}", [{"role": "user", "content": str(i)}])
        assert cm.active_conversations == 200

    def test_thread_safety(self):
        cm = ConversationManager()
        errors = []

        def writer(chat_id: str):
            try:
                for i in range(50):
                    h = cm.get_history(chat_id)
                    h.append({"role": "user", "content": f"msg-{i}"})
                    cm.save_history(chat_id, h)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(f"chat-{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        for i in range(5):
            assert len(cm.get_history(f"chat-{i}")) == 50
