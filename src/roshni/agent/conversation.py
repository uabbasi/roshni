"""Conversation manager — per-chat conversation history isolation.

Maintains separate message histories keyed by chat_id, allowing a single
agent instance to serve multiple concurrent conversations (e.g. different
Telegram chats) without context bleeding.

Thread-safe: all history access is guarded by per-chat locks since
:meth:`DefaultAgent.chat` runs in a thread-pool executor.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

from loguru import logger


class ConversationManager:
    """Manages per-chat conversation histories for multi-conversation agents.

    Each unique ``chat_id`` gets its own message history list.  When the
    number of tracked conversations exceeds ``max_conversations``, the
    least-recently-used conversation is evicted.

    Args:
        max_conversations: Maximum number of concurrent conversations
            to keep in memory.  Oldest (LRU) conversations are evicted
            when this limit is reached.  Set to ``0`` for unlimited.
    """

    def __init__(self, max_conversations: int = 100):
        self._max = max_conversations
        self._histories: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._session_ids: dict[str, str] = {}
        self._lock = threading.Lock()

    def get_history(self, chat_id: str) -> list[dict[str, Any]]:
        """Return the message history for *chat_id* (creates empty if new).

        Touches the entry for LRU tracking.
        """
        with self._lock:
            if chat_id in self._histories:
                self._histories.move_to_end(chat_id)
                return self._histories[chat_id]
            # New conversation
            history: list[dict[str, Any]] = []
            self._histories[chat_id] = history
            self._evict_if_needed()
            return history

    def save_history(self, chat_id: str, history: list[dict[str, Any]]) -> None:
        """Persist the history list back for *chat_id*."""
        with self._lock:
            self._histories[chat_id] = history
            self._histories.move_to_end(chat_id)
            self._evict_if_needed()

    def clear_history(self, chat_id: str) -> None:
        """Clear the conversation history for a single chat."""
        with self._lock:
            if chat_id in self._histories:
                self._histories[chat_id] = []
            self._session_ids.pop(chat_id, None)

    def get_session_id(self, chat_id: str) -> str | None:
        """Return the session ID associated with *chat_id*, if any."""
        with self._lock:
            return self._session_ids.get(chat_id)

    def set_session_id(self, chat_id: str, session_id: str) -> None:
        """Associate a session store ID with *chat_id*."""
        with self._lock:
            self._session_ids[chat_id] = session_id

    @property
    def active_conversations(self) -> int:
        """Number of conversations currently tracked."""
        with self._lock:
            return len(self._histories)

    def _evict_if_needed(self) -> None:
        """Evict the oldest conversation if over capacity.  Caller holds lock."""
        if self._max and len(self._histories) > self._max:
            evicted_id, _ = self._histories.popitem(last=False)
            self._session_ids.pop(evicted_id, None)
            logger.debug(f"Evicted conversation {evicted_id} (capacity={self._max})")
