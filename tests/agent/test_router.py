"""Tests for roshni.agent.router."""

import asyncio
import re
from collections.abc import Callable
from typing import Any

from roshni.agent.base import BaseAgent, ChatResult
from roshni.agent.router import Router


class EchoAgent(BaseAgent):
    """Test agent that echoes back the message."""

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
        return ChatResult(text=f"echo: {message}", model="test")


class TestCommandParsing:
    def test_plain_message(self):
        router = Router()
        result = router.parse_command("Hello there")
        assert result.agent_hint is None
        assert result.mode is None
        assert result.query == "Hello there"

    def test_slash_command(self):
        router = Router(command_modes={"/analyze": "analyze", "/coach": "coach"})
        result = router.parse_command("/analyze my sleep patterns")
        assert result.mode == "analyze"
        assert result.query == "my sleep patterns"

    def test_unknown_slash_passes_through(self):
        router = Router(command_modes={"/analyze": "analyze"})
        result = router.parse_command("/unknown do something")
        assert result.mode is None
        assert result.query == "/unknown do something"

    def test_prefix_route(self):
        router = Router(prefix_routes={"cfo:": "cfo", "ask:": None})
        result = router.parse_command("cfo: what is my net worth?")
        assert result.agent_hint == "cfo"
        assert result.query == "what is my net worth?"

    def test_keyword_pattern(self):
        pattern = re.compile(r"\bportfolio\b", re.IGNORECASE)
        router = Router(keyword_patterns=[(pattern, "cfo")])
        result = router.parse_command("Check my portfolio allocation")
        assert result.agent_hint == "cfo"

    def test_bare_command(self):
        router = Router(command_modes={"/analyze": "analyze"})
        result = router.parse_command("/analyze")
        assert result.mode == "analyze"
        assert result.query == ""


class TestRouting:
    def test_route_to_agent(self):
        router = Router(agent_factory=lambda: EchoAgent(name="echo"))
        result = asyncio.run(router.route("Hello"))
        assert result.text == "echo: Hello"
        assert result.agent_name == "echo"

    def test_bare_command_returns_label(self):
        router = Router(
            command_modes={"/analyze": "analyze"},
            mode_labels={"analyze": "Analysis mode"},
            agent_factory=lambda: EchoAgent(),
        )
        result = asyncio.run(router.route("/analyze"))
        assert "Analysis mode" in result.text
        assert result.agent_name == "router"

    def test_is_busy(self):
        router = Router(agent_factory=lambda: EchoAgent())
        assert not router.is_busy

    def test_steer_passthrough(self):
        agent = EchoAgent()
        router = Router(agent_factory=lambda: agent)
        # Force agent initialization
        asyncio.run(router.route("init"))
        router.steer("new direction")
        assert agent.drain_steering() == "new direction"

    def test_no_factory_returns_error(self):
        router = Router()
        result = asyncio.run(router.route("test"))
        assert "Error" in result.text
        assert "agent_factory" in result.text
