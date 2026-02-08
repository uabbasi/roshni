"""Tests for roshni.agent.base."""

import asyncio
from collections.abc import Callable
from typing import Any

from roshni.agent.base import BaseAgent, ChatResult


class StubAgent(BaseAgent):
    """Minimal concrete agent for testing the ABC."""

    def __init__(self, response: str = "ok", **kwargs):
        super().__init__(**kwargs)
        self._response = response
        self.last_message: str | None = None
        self.last_mode: str | None = None
        self.history: list[str] = []

    def chat(
        self,
        message: str,
        *,
        mode: str | None = None,
        channel: str | None = None,
        max_iterations: int = 5,
        on_tool_start: Callable[[str, int, dict | None], None] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.last_message = message
        self.last_mode = mode
        self.history.append(message)
        return ChatResult(text=self._response, duration=0.01)

    def clear_history(self) -> None:
        self.history.clear()


class TestBaseAgent:
    def test_chat_returns_result(self):
        agent = StubAgent(response="hello")
        result = agent.chat("hi")
        assert result.text == "hello"
        assert result.duration > 0

    def test_name(self):
        agent = StubAgent(name="test-agent")
        assert agent.name == "test-agent"

    def test_default_name(self):
        agent = StubAgent()
        assert agent.name == "agent"

    def test_clear_history(self):
        agent = StubAgent()
        agent.chat("one")
        agent.chat("two")
        assert len(agent.history) == 2
        agent.clear_history()
        assert len(agent.history) == 0


class TestSteering:
    def test_steering_queue(self):
        agent = StubAgent()
        agent.steer("stop that")
        agent.steer("do this instead")
        # Only latest should be returned
        result = agent.drain_steering()
        assert result == "do this instead"

    def test_empty_steering(self):
        agent = StubAgent()
        assert agent.drain_steering() is None

    def test_followup_queue(self):
        agent = StubAgent()
        agent.enqueue_followup("also check X")
        agent.enqueue_followup("and Y")
        items = agent.drain_followups()
        assert items == ["also check X", "and Y"]

    def test_empty_followups(self):
        agent = StubAgent()
        assert agent.drain_followups() == []


class TestBusyFlag:
    def test_not_busy_initially(self):
        agent = StubAgent()
        assert not agent.is_busy


class TestInvoke:
    def test_async_invoke(self):
        agent = StubAgent(response="async result")
        result = asyncio.run(agent.invoke("test query"))
        assert result == "async result"
        assert agent.last_message == "test query"

    def test_invoke_passes_mode(self):
        agent = StubAgent()
        asyncio.run(agent.invoke("q", mode="analyze"))
        assert agent.last_mode == "analyze"


class TestChatResult:
    def test_defaults(self):
        r = ChatResult(text="hi")
        assert r.duration == 0.0
        assert r.tool_calls == []
        assert r.model == ""

    def test_with_tool_calls(self):
        r = ChatResult(
            text="done",
            tool_calls=[{"name": "search", "args": {}, "result": "found"}],
        )
        assert len(r.tool_calls) == 1
