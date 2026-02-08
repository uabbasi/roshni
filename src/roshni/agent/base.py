"""Base agent — abstract interface for tool-calling conversational agents.

Defines the contract that all agents must follow, without coupling to
any specific LLM framework (LangChain, LiteLLM, etc.).  Concrete
implementations provide the LLM invocation and tool execution logic.
"""

from __future__ import annotations

import queue
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatResult:
    """Result of an agent chat interaction."""

    text: str
    """The agent's text response."""

    duration: float = 0.0
    """Wall-clock seconds for the full interaction."""

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    """List of tool calls made: [{"name": ..., "args": ..., "result": ...}]."""

    model: str = ""
    """Model used for the interaction."""


class BaseAgent(ABC):
    """Abstract base for tool-calling conversational agents.

    Subclasses implement :meth:`chat` with their LLM framework of choice.
    The base class provides thread-safe steering, follow-up queues, and
    a busy flag so callers can coordinate with running agents.

    Usage::

        class MyAgent(BaseAgent):
            def chat(self, message, **kwargs):
                # Your LLM + tool loop here
                return ChatResult(text="Hello!")

        agent = MyAgent(name="assistant")
        result = agent.chat("Hi there")
    """

    def __init__(self, name: str = "agent"):
        self.name = name
        self._steering_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._followup_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._busy = threading.Event()

    @property
    def is_busy(self) -> bool:
        """True while :meth:`chat` is executing (thread-safe)."""
        return self._busy.is_set()

    def steer(self, message: str) -> None:
        """Inject a steering message into the running tool loop.

        If the agent is mid-tool-loop, the next iteration picks this up,
        skips remaining tools, and re-invokes the LLM with the new context.
        """
        self._steering_queue.put(message)

    def enqueue_followup(self, message: str) -> None:
        """Queue a follow-up message to extend the agent loop.

        After the current tool loop finishes, the agent processes up to
        *max_followups* queued messages before returning.
        """
        self._followup_queue.put(message)

    def drain_steering(self) -> str | None:
        """Drain the steering queue, returning only the latest message."""
        latest = None
        while True:
            try:
                latest = self._steering_queue.get_nowait()
            except queue.Empty:
                break
        return latest

    def drain_followups(self) -> list[str]:
        """Drain all queued follow-ups, returning them as a list."""
        items: list[str] = []
        while True:
            try:
                items.append(self._followup_queue.get_nowait())
            except queue.Empty:
                break
        return items

    @abstractmethod
    def chat(
        self,
        message: str,
        *,
        mode: str | None = None,
        call_type: str | None = None,
        channel: str | None = None,
        max_iterations: int = 5,
        on_tool_start: Callable[[str, int, dict | None], None] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Process a user message, potentially calling tools.

        Args:
            message: User query.
            mode: Optional mode hint (e.g. from a slash command).
            call_type: Optional call classification (e.g. ``"heartbeat"``,
                ``"scheduled"``).  Implementations may use this to adjust
                behavior such as skipping conversation history.
            channel: Channel identifier (telegram, cli, etc.).
            max_iterations: Max tool-call rounds.
            on_tool_start: Progress callback ``(tool_name, index, args)``.

        Returns:
            :class:`ChatResult` with the response text and metadata.
        """

    async def invoke(
        self,
        query: str,
        *,
        mode: str | None = None,
        call_type: str | None = None,
        channel: str | None = None,
        on_tool_start: Callable[[str, int, dict | None], None] | None = None,
        **kwargs: Any,
    ) -> str:
        """Async wrapper that runs :meth:`chat` in an executor.

        Returns only the text response for convenience.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.chat(
                query,
                mode=mode,
                call_type=call_type,
                channel=channel,
                on_tool_start=on_tool_start,
                **kwargs,
            ),
        )
        return result.text

    def clear_history(self) -> None:  # noqa: B027
        """Clear conversation history. Override in subclasses with state."""
