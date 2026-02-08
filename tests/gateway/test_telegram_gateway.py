"""Tests for the Telegram gateway (unit tests, no real Telegram connection)."""

import pytest

from roshni.agent.base import BaseAgent, ChatResult
from roshni.gateway.plugins.telegram.bot import TelegramGateway, _md_to_html, _split_message


class MockAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="test")
        self.response = "Mock response"

    def chat(self, message, *, mode=None, channel=None, max_iterations=5, on_tool_start=None, **kwargs):
        return ChatResult(text=self.response)

    def clear_history(self):
        pass


class TestMdToHtml:
    def test_bold(self):
        assert "<b>hello</b>" in _md_to_html("**hello**")

    def test_italic(self):
        assert "<i>hello</i>" in _md_to_html("*hello*")

    def test_inline_code(self):
        assert "<code>foo</code>" in _md_to_html("`foo`")

    def test_code_block(self):
        text = "```\nprint('hi')\n```"
        result = _md_to_html(text)
        assert "<pre>" in result
        assert "print" in result

    def test_header(self):
        assert "<b>Title</b>" in _md_to_html("## Title")

    def test_bullets(self):
        result = _md_to_html("- item one\n- item two")
        assert "item one" in result
        assert "item two" in result

    def test_html_escape(self):
        result = _md_to_html("x < y & z > w")
        assert "&lt;" in result
        assert "&amp;" in result
        assert "&gt;" in result

    def test_plain_text_passthrough(self):
        assert _md_to_html("Hello world") == "Hello world"


class TestSplitMessage:
    def test_short_message(self):
        assert _split_message("hello") == ["hello"]

    def test_long_message_splits_at_newline(self):
        text = "a" * 2000 + "\n" + "b" * 2000 + "\n" + "c" * 2000
        chunks = _split_message(text, max_length=4096)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) <= 4096

    def test_exact_limit(self):
        text = "a" * 4096
        assert _split_message(text) == [text]


class TestTelegramGatewayInit:
    def test_basic_init(self):
        agent = MockAgent()
        gw = TelegramGateway(agent=agent, bot_token="fake-token")
        assert gw.bot_token == "fake-token"
        assert gw.allowed_user_ids == set()

    def test_with_user_ids(self):
        agent = MockAgent()
        gw = TelegramGateway(agent=agent, bot_token="fake", allowed_user_ids=["123", 456])
        assert gw.allowed_user_ids == {123, 456}

    def test_authorization_empty_allowlist(self):
        agent = MockAgent()
        gw = TelegramGateway(agent=agent, bot_token="fake")
        assert not gw._is_authorized(999)

    def test_authorization_with_allowlist(self):
        agent = MockAgent()
        gw = TelegramGateway(agent=agent, bot_token="fake", allowed_user_ids=["123"])
        assert gw._is_authorized(123)
        assert not gw._is_authorized(456)

    @pytest.mark.asyncio
    async def test_handle_message(self):
        agent = MockAgent()
        agent.response = "Bot says hi"
        gw = TelegramGateway(agent=agent, bot_token="fake")
        result = await gw.handle_message("hello", "user1")
        assert result == "Bot says hi"
