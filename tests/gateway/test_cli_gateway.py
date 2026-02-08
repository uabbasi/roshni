"""Tests for the CLI gateway."""

import pytest

from roshni.agent.base import BaseAgent, ChatResult
from roshni.gateway.cli_gateway import CliGateway


class MockAgent(BaseAgent):
    """Simple mock agent for testing."""

    def __init__(self):
        super().__init__(name="test")
        self.last_message = ""
        self.response = "Mock response"
        self._history_cleared = False

    def chat(self, message, *, mode=None, channel=None, max_iterations=5, on_tool_start=None, **kwargs):
        self.last_message = message
        return ChatResult(text=self.response)

    def clear_history(self):
        self._history_cleared = True


class TestCliGateway:
    def test_init(self):
        agent = MockAgent()
        gw = CliGateway(agent)
        assert gw.agent is agent
        assert gw.user_id == "cli_user"

    @pytest.mark.asyncio
    async def test_handle_message(self):
        agent = MockAgent()
        agent.response = "Hello from mock"
        gw = CliGateway(agent)
        result = await gw.handle_message("Hi", "user1")
        assert result == "Hello from mock"

    @pytest.mark.asyncio
    async def test_stop(self):
        agent = MockAgent()
        gw = CliGateway(agent)
        gw._running = True
        await gw.stop()
        assert not gw._running
