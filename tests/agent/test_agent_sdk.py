"""Tests for AgentSDKAgent — Claude Agent SDK integration."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from roshni.agent.base import ChatResult
from roshni.agent.tools import ToolDefinition
from roshni.core.config import Config
from roshni.core.secrets import SecretsManager

# ---------------------------------------------------------------------------
# Mock claude_agent_sdk module
# ---------------------------------------------------------------------------


def _build_mock_sdk():
    """Build a mock claude_agent_sdk module with the required types."""
    sdk = types.ModuleType("claude_agent_sdk")

    # ClaudeAgentOptions
    class MockClaudeAgentOptions:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    sdk.ClaudeAgentOptions = MockClaudeAgentOptions

    # Message types
    class MockTextBlock:
        def __init__(self, text=""):
            self.text = text

    class MockToolUseBlock:
        def __init__(self, name="", input=None):
            self.name = name
            self.input = input or {}

    class MockAssistantMessage:
        def __init__(self, content=None):
            self.content = content or []

    class MockResultMessage:
        def __init__(self, result=None):
            self.result = result
            self.subtype = "result"
            self.duration_ms = 100
            self.is_error = False
            self.num_turns = 1
            self.session_id = "test"
            self.total_cost_usd = 0.01
            self.usage = {}

    sdk.TextBlock = MockTextBlock
    sdk.ToolUseBlock = MockToolUseBlock
    sdk.AssistantMessage = MockAssistantMessage
    sdk.ResultMessage = MockResultMessage

    # query() — async generator
    async def mock_query(prompt="", options=None):
        msg = MockAssistantMessage(content=[MockTextBlock(text=f"Response to: {prompt}")])
        yield msg

    sdk.query = mock_query

    # ClaudeSDKClient — async context manager
    class MockClaudeSDKClient:
        def __init__(self, options=None):
            self.options = options
            self._messages = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def query(self, prompt):
            self._messages = [MockAssistantMessage(content=[MockTextBlock(text=f"Client response to: {prompt}")])]

        async def receive_response(self):
            for msg in self._messages:
                yield msg

    sdk.ClaudeSDKClient = MockClaudeSDKClient

    # @tool decorator — returns the function unchanged with metadata attached
    def mock_tool(name, description, schema):
        def decorator(fn):
            fn._tool_name = name
            fn._tool_description = description
            fn._tool_schema = schema
            return fn

        return decorator

    sdk.tool = mock_tool

    # create_sdk_mcp_server — returns a mock server object
    def mock_create_sdk_mcp_server(name="", version="", tools=None):
        return MagicMock(name=f"mcp_server:{name}", tools=tools or [])

    sdk.create_sdk_mcp_server = mock_create_sdk_mcp_server

    return sdk


_mock_sdk = _build_mock_sdk()


@pytest.fixture(autouse=True)
def _install_mock_sdk():
    """Install mock claude_agent_sdk for all tests in this module."""
    sys.modules["claude_agent_sdk"] = _mock_sdk
    yield
    sys.modules.pop("claude_agent_sdk", None)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config(tmp_dir):
    return Config(
        data_dir=tmp_dir,
        defaults={"llm": {"provider": "anthropic", "model": "anthropic/claude-sonnet-4-6"}},
    )


@pytest.fixture
def secrets():
    return SecretsManager(providers=[])


@pytest.fixture
def echo_tool():
    return ToolDefinition(
        name="echo",
        description="Echo the input back",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to echo"},
            },
            "required": ["text"],
        },
        function=lambda text: f"Echo: {text}",
    )


@pytest.fixture
def calc_tool():
    return ToolDefinition(
        name="add",
        description="Add two numbers",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
        function=lambda a, b: str(int(a) + int(b)),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentSDKAgentInit:
    def test_basic_init(self, config, secrets):
        from roshni.agent.agent_sdk import AgentSDKAgent

        agent = AgentSDKAgent(config=config, secrets=secrets)
        assert agent.name == "assistant"
        assert agent.model == "claude-agent-sdk"
        assert agent.provider == "anthropic"
        assert agent.tools == []

    def test_custom_name(self, config, secrets):
        from roshni.agent.agent_sdk import AgentSDKAgent

        agent = AgentSDKAgent(config=config, secrets=secrets, name="Roshni")
        assert agent.name == "Roshni"

    def test_custom_system_prompt(self, config, secrets):
        from roshni.agent.agent_sdk import AgentSDKAgent

        agent = AgentSDKAgent(config=config, secrets=secrets, system_prompt="You are a cat.")
        assert agent._system_prompt == "You are a cat."

    def test_default_system_prompt(self, config, secrets):
        from roshni.agent.agent_sdk import AgentSDKAgent

        agent = AgentSDKAgent(config=config, secrets=secrets)
        assert "helpful" in agent._system_prompt.lower()

    def test_with_tools(self, config, secrets, echo_tool):
        from roshni.agent.agent_sdk import AgentSDKAgent

        agent = AgentSDKAgent(config=config, secrets=secrets, tools=[echo_tool])
        assert len(agent.tools) == 1
        assert agent._mcp_server is not None
        assert "mcp__roshni-tools__echo" in agent._allowed_tools

    def test_with_multiple_tools(self, config, secrets, echo_tool, calc_tool):
        from roshni.agent.agent_sdk import AgentSDKAgent

        agent = AgentSDKAgent(config=config, secrets=secrets, tools=[echo_tool, calc_tool])
        assert len(agent.tools) == 2
        assert len(agent._allowed_tools) == 2

    def test_no_tools_no_mcp_server(self, config, secrets):
        from roshni.agent.agent_sdk import AgentSDKAgent

        agent = AgentSDKAgent(config=config, secrets=secrets)
        assert agent._mcp_server is None
        assert agent._allowed_tools == []

    def test_max_turns(self, config, secrets):
        from roshni.agent.agent_sdk import AgentSDKAgent

        agent = AgentSDKAgent(config=config, secrets=secrets, max_turns=10)
        assert agent._max_turns == 10

    def test_model_parameter(self, config, secrets):
        from roshni.agent.agent_sdk import AgentSDKAgent

        agent = AgentSDKAgent(config=config, secrets=secrets, model="sonnet")
        assert agent.model == "sonnet"
        assert agent._model == "sonnet"

    def test_model_default(self, config, secrets):
        from roshni.agent.agent_sdk import AgentSDKAgent

        agent = AgentSDKAgent(config=config, secrets=secrets)
        assert agent.model == "claude-agent-sdk"


class TestAgentSDKAgentChat:
    def test_simple_chat(self, config, secrets):
        from roshni.agent.agent_sdk import AgentSDKAgent

        agent = AgentSDKAgent(config=config, secrets=secrets)
        result = agent.chat("Hello")
        assert isinstance(result, ChatResult)
        assert "Response to: Hello" in result.text
        assert result.model == "claude-agent-sdk"
        assert result.duration >= 0
        assert result.tool_calls == []

    def test_chat_with_tools_uses_client(self, config, secrets, echo_tool):
        from roshni.agent.agent_sdk import AgentSDKAgent

        agent = AgentSDKAgent(config=config, secrets=secrets, tools=[echo_tool])
        result = agent.chat("Echo something")
        assert isinstance(result, ChatResult)
        assert "Client response to: Echo something" in result.text

    def test_chat_busy_flag(self, config, secrets):
        from roshni.agent.agent_sdk import AgentSDKAgent

        agent = AgentSDKAgent(config=config, secrets=secrets)
        assert not agent.is_busy
        result = agent.chat("test")
        assert not agent.is_busy
        assert isinstance(result, ChatResult)

    def test_chat_error_handling(self, config, secrets):
        from roshni.agent.agent_sdk import AgentSDKAgent

        agent = AgentSDKAgent(config=config, secrets=secrets)

        with patch.object(agent, "_achat", side_effect=RuntimeError("SDK failed")):
            result = agent.chat("test")
            assert "error" in result.text.lower()
            assert result.tool_calls == []

    def test_chat_stream_callback(self, config, secrets):
        from roshni.agent.agent_sdk import AgentSDKAgent

        agent = AgentSDKAgent(config=config, secrets=secrets)
        chunks = []
        result = agent.chat("test", on_stream=chunks.append)
        assert isinstance(result, ChatResult)
        assert len(chunks) > 0


class TestAgentSDKToolConversion:
    def test_build_mcp_server(self, echo_tool):
        from roshni.agent.agent_sdk import _build_mcp_server

        server = _build_mcp_server([echo_tool])
        assert server is not None

    def test_build_mcp_server_multiple_tools(self, echo_tool, calc_tool):
        from roshni.agent.agent_sdk import _build_mcp_server

        server = _build_mcp_server([echo_tool, calc_tool])
        assert server is not None

    def test_build_mcp_server_empty_params(self):
        from roshni.agent.agent_sdk import _build_mcp_server

        tool = ToolDefinition(
            name="ping",
            description="Ping",
            parameters={"type": "object", "properties": {}},
            function=lambda: "pong",
        )
        server = _build_mcp_server([tool])
        assert server is not None


class TestAgentSDKMessageExtraction:
    def test_extract_text_from_assistant_message(self):
        from roshni.agent.agent_sdk import _extract_text_from_sdk_message

        msg = _mock_sdk.AssistantMessage(content=[_mock_sdk.TextBlock(text="hello")])
        assert _extract_text_from_sdk_message(msg) == "hello"

    def test_extract_text_from_multiple_blocks(self):
        from roshni.agent.agent_sdk import _extract_text_from_sdk_message

        msg = _mock_sdk.AssistantMessage(
            content=[
                _mock_sdk.TextBlock(text="hello "),
                _mock_sdk.TextBlock(text="world"),
            ]
        )
        assert _extract_text_from_sdk_message(msg) == "hello world"

    def test_extract_text_skips_tool_use_blocks(self):
        from roshni.agent.agent_sdk import _extract_text_from_sdk_message

        msg = _mock_sdk.AssistantMessage(
            content=[
                _mock_sdk.TextBlock(text="thinking..."),
                _mock_sdk.ToolUseBlock(name="echo", input={"text": "hi"}),
            ]
        )
        assert _extract_text_from_sdk_message(msg) == "thinking..."

    def test_extract_text_from_empty_message(self):
        from roshni.agent.agent_sdk import _extract_text_from_sdk_message

        msg = _mock_sdk.AssistantMessage(content=[])
        assert _extract_text_from_sdk_message(msg) == ""

    def test_extract_text_from_string_content(self):
        from roshni.agent.agent_sdk import _extract_text_from_sdk_message

        msg = MagicMock()
        msg.content = "plain string"
        assert _extract_text_from_sdk_message(msg) == "plain string"

    def test_extract_text_from_none_content(self):
        from roshni.agent.agent_sdk import _extract_text_from_sdk_message

        msg = MagicMock(spec=[])
        assert _extract_text_from_sdk_message(msg) == ""

    def test_extract_text_from_result_message(self):
        from roshni.agent.agent_sdk import _extract_text_from_sdk_message

        msg = _mock_sdk.ResultMessage(result="Final answer here")
        assert _extract_text_from_sdk_message(msg) == "Final answer here"

    def test_extract_text_from_result_message_none(self):
        from roshni.agent.agent_sdk import _extract_text_from_sdk_message

        msg = _mock_sdk.ResultMessage(result=None)
        assert _extract_text_from_sdk_message(msg) == ""

    def test_extract_tool_calls(self):
        from roshni.agent.agent_sdk import _extract_tool_calls_from_sdk_message

        msg = _mock_sdk.AssistantMessage(
            content=[
                _mock_sdk.ToolUseBlock(name="echo", input={"text": "hi"}),
            ]
        )
        calls = _extract_tool_calls_from_sdk_message(msg)
        assert len(calls) == 1
        assert calls[0]["name"] == "echo"
        assert calls[0]["args"] == {"text": "hi"}

    def test_extract_tool_calls_empty(self):
        from roshni.agent.agent_sdk import _extract_tool_calls_from_sdk_message

        msg = _mock_sdk.AssistantMessage(content=[_mock_sdk.TextBlock(text="no tools")])
        calls = _extract_tool_calls_from_sdk_message(msg)
        assert calls == []


class TestRunAsync:
    def test_run_async_creates_loop(self):
        from roshni.agent.agent_sdk import _run_async

        async def coro():
            return 42

        assert _run_async(coro()) == 42


class TestImportError:
    def test_import_error_without_sdk(self, config, secrets):
        """AgentSDKAgent raises ImportError when claude_agent_sdk is missing."""
        saved = sys.modules.pop("claude_agent_sdk", None)
        # Insert a sentinel that makes `import claude_agent_sdk` raise
        sys.modules["claude_agent_sdk"] = None  # type: ignore[assignment]
        try:
            # Force re-import of agent_sdk so it picks up the missing module
            import importlib

            import roshni.agent.agent_sdk as mod

            importlib.reload(mod)

            with pytest.raises(ImportError, match="claude-agent-sdk"):
                mod.AgentSDKAgent(config=config, secrets=secrets)
        finally:
            sys.modules.pop("claude_agent_sdk", None)
            if saved is not None:
                sys.modules["claude_agent_sdk"] = saved
            else:
                # Restore real module if it was freshly importable
                try:
                    import claude_agent_sdk

                    sys.modules["claude_agent_sdk"] = claude_agent_sdk
                except ImportError:
                    pass
