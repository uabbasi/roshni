"""Tests for roshni.gateway.base."""

import asyncio

from roshni.gateway.base import BotGateway


class StubGateway(BotGateway):
    """Minimal concrete gateway for testing."""

    def __init__(self):
        self.started = False
        self.stopped = False

    async def start(self):
        self.started = True

    async def handle_message(self, message: str, user_id: str) -> str:
        return f"reply to {user_id}: {message}"

    async def stop(self):
        self.stopped = True


class TestBotGateway:
    def test_start(self):
        gw = StubGateway()
        asyncio.run(gw.start())
        assert gw.started

    def test_handle_message(self):
        gw = StubGateway()
        result = asyncio.run(gw.handle_message("hello", "user123"))
        assert result == "reply to user123: hello"

    def test_stop(self):
        gw = StubGateway()
        asyncio.run(gw.stop())
        assert gw.stopped

    def test_is_subclass(self):
        assert issubclass(StubGateway, BotGateway)
