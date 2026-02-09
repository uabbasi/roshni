"""Tests for DefaultAgent."""

import os
from unittest.mock import MagicMock, patch

import pytest

from roshni.agent.base import ChatResult
from roshni.agent.default import DefaultAgent
from roshni.agent.tools import ToolDefinition
from roshni.core.config import Config
from roshni.core.secrets import SecretsManager


@pytest.fixture
def config(tmp_dir):
    """Create a minimal config."""
    return Config(
        data_dir=tmp_dir,
        defaults={
            "llm": {"provider": "openai", "model": "gpt-5.2-chat-latest"},
        },
    )


@pytest.fixture
def secrets():
    return SecretsManager(providers=[])


@pytest.fixture
def echo_tool():
    """A simple tool that echoes its input."""
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
def write_tool():
    return ToolDefinition(
        name="write_thing",
        description="Write a value",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        function=lambda text: f"Wrote: {text}",
        permission="write",
    )


def _make_response(content, tool_calls=None):
    """Helper to build a mock litellm completion response."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock(choices=[choice])
    resp.usage = None
    return resp


class TestDefaultAgentInit:
    def test_basic_init(self, config, secrets):
        agent = DefaultAgent(config=config, secrets=secrets)
        assert agent.model == "gpt-5.2-chat-latest"
        assert agent.provider == "openai"
        assert agent.name == "assistant"

    def test_custom_name(self, config, secrets):
        agent = DefaultAgent(config=config, secrets=secrets, name="Roshni")
        assert agent.name == "Roshni"

    def test_custom_system_prompt(self, config, secrets):
        agent = DefaultAgent(config=config, secrets=secrets, system_prompt="You are a cat.")
        assert agent._llm.system_prompt == "You are a cat."

    def test_persona_dir(self, config, secrets, tmp_dir):
        persona_dir = os.path.join(tmp_dir, "persona")
        os.makedirs(persona_dir)
        with open(os.path.join(persona_dir, "IDENTITY.md"), "w") as f:
            f.write("# TestBot\nYou are TestBot.")

        agent = DefaultAgent(config=config, secrets=secrets, persona_dir=persona_dir)
        assert "TestBot" in agent._llm.system_prompt

    def test_with_tools(self, config, secrets, echo_tool):
        agent = DefaultAgent(config=config, secrets=secrets, tools=[echo_tool])
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "echo"


class TestDefaultAgentChat:
    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_simple_chat(self, mock_completion, config, secrets):
        mock_completion.return_value = _make_response("Hello! How can I help?")

        agent = DefaultAgent(config=config, secrets=secrets)
        result = agent.chat("Hi there")

        assert isinstance(result, ChatResult)
        assert result.text == "Hello! How can I help?"
        assert result.tool_calls == []
        assert result.model == "gpt-5.2-chat-latest"

    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_chat_with_tool_call(self, mock_completion, config, secrets, echo_tool):
        # First response: tool call
        tool_call = MagicMock()
        tool_call.id = "call_123"
        tool_call.function.name = "echo"
        tool_call.function.arguments = '{"text": "hello"}'

        first_resp = _make_response(None, tool_calls=[tool_call])
        second_resp = _make_response("The echo says: Echo: hello")

        mock_completion.side_effect = [first_resp, second_resp]

        agent = DefaultAgent(config=config, secrets=secrets, tools=[echo_tool])
        result = agent.chat("Echo hello")

        assert result.text == "The echo says: Echo: hello"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "echo"
        assert result.tool_calls[0]["result"] == "Echo: hello"

    def test_clear_history(self, config, secrets):
        agent = DefaultAgent(config=config, secrets=secrets)
        agent.message_history.append({"role": "user", "content": "test"})
        assert len(agent.message_history) == 1
        agent.clear_history()
        assert len(agent.message_history) == 0

    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_busy_flag(self, mock_completion, config, secrets):
        mock_completion.return_value = _make_response("Done")

        agent = DefaultAgent(config=config, secrets=secrets)
        assert not agent.is_busy
        agent.chat("test")
        assert not agent.is_busy

    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_error_handling(self, mock_completion, config, secrets):
        mock_completion.side_effect = RuntimeError("API down")

        agent = DefaultAgent(config=config, secrets=secrets)
        result = agent.chat("test")
        assert "went wrong" in result.text.lower() or "API down" in result.text

    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_completion_called_with_tools(self, mock_completion, config, secrets, echo_tool):
        """Verify that tool schemas are passed through to LLMClient.completion()."""
        mock_completion.return_value = _make_response("Done")

        agent = DefaultAgent(config=config, secrets=secrets, tools=[echo_tool])
        agent.chat("test")

        call_kwargs = mock_completion.call_args
        tools_arg = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools")
        assert tools_arg is not None
        assert len(tools_arg) == 1
        assert tools_arg[0]["function"]["name"] == "echo"

    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_write_tool_requires_approval(self, mock_completion, config, secrets, write_tool):
        tool_call = MagicMock()
        tool_call.id = "call_approval"
        tool_call.function.name = "write_thing"
        tool_call.function.arguments = '{"text": "hello"}'
        mock_completion.return_value = _make_response(None, tool_calls=[tool_call])

        agent = DefaultAgent(config=config, secrets=secrets, tools=[write_tool])
        result = agent.chat("Save hello")
        assert "Approval required" in result.text

        approved = agent.chat("approve")
        assert "Approved and executed" in approved.text
        assert approved.tool_calls[0]["name"] == "write_thing"
        assert "Wrote: hello" in approved.text

    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_write_tool_denied(self, mock_completion, config, secrets, write_tool):
        tool_call = MagicMock()
        tool_call.id = "call_approval"
        tool_call.function.name = "write_thing"
        tool_call.function.arguments = '{"text": "hello"}'
        mock_completion.return_value = _make_response(None, tool_calls=[tool_call])

        agent = DefaultAgent(config=config, secrets=secrets, tools=[write_tool])
        _ = agent.chat("Save hello")
        denied = agent.chat("deny")
        assert "Canceled" in denied.text


class TestDefaultAgentMultiProviderConfig:
    """Tests for the new multi-provider config format."""

    def test_new_config_format(self, tmp_dir, secrets):
        config = Config(
            data_dir=tmp_dir,
            defaults={
                "llm": {
                    "default": "anthropic",
                    "providers": {
                        "anthropic": {"model": "anthropic/claude-sonnet-4-20250514"},
                    },
                },
            },
        )
        agent = DefaultAgent(config=config, secrets=secrets)
        assert agent.provider == "anthropic"
        assert agent.model == "anthropic/claude-sonnet-4-20250514"

    def test_new_config_with_fallback(self, tmp_dir, secrets):
        config = Config(
            data_dir=tmp_dir,
            defaults={
                "llm": {
                    "default": "anthropic",
                    "fallback": "openai",
                    "providers": {
                        "anthropic": {"model": "anthropic/claude-sonnet-4-20250514"},
                        "openai": {"model": "gpt-5.2-chat-latest"},
                    },
                },
            },
        )
        agent = DefaultAgent(config=config, secrets=secrets)
        assert agent.provider == "anthropic"
        assert agent._llm.fallback_model == "gpt-5.2-chat-latest"
        assert agent._llm.fallback_provider == "openai"

    def test_fallback_uses_default_model_when_not_specified(self, tmp_dir, secrets):
        config = Config(
            data_dir=tmp_dir,
            defaults={
                "llm": {
                    "default": "anthropic",
                    "fallback": "deepseek",
                    "providers": {
                        "anthropic": {"model": "anthropic/claude-sonnet-4-20250514"},
                        "deepseek": {},  # no model specified
                    },
                },
            },
        )
        agent = DefaultAgent(config=config, secrets=secrets)
        assert agent._llm.fallback_model == "deepseek/deepseek-chat"
        assert agent._llm.fallback_provider == "deepseek"

    def test_legacy_config_still_works(self, tmp_dir, secrets):
        """Old llm.provider format should continue working."""
        config = Config(
            data_dir=tmp_dir,
            defaults={
                "llm": {"provider": "openai", "model": "gpt-5.2-chat-latest"},
            },
        )
        agent = DefaultAgent(config=config, secrets=secrets)
        assert agent.provider == "openai"
        assert agent.model == "gpt-5.2-chat-latest"
        assert agent._llm.fallback_model is None

    def test_no_fallback_when_not_configured(self, tmp_dir, secrets):
        config = Config(
            data_dir=tmp_dir,
            defaults={
                "llm": {
                    "default": "openai",
                    "providers": {
                        "openai": {"model": "gpt-5.2-chat-latest"},
                    },
                },
            },
        )
        agent = DefaultAgent(config=config, secrets=secrets)
        assert agent._llm.fallback_model is None


class TestDefaultAgentBuildMessages:
    def test_includes_system_prompt(self, config, secrets):
        agent = DefaultAgent(config=config, secrets=secrets, system_prompt="Be helpful.")
        agent.message_history.append({"role": "user", "content": "hi"})
        messages = agent._build_messages()
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be helpful."
        assert messages[1]["role"] == "user"

    def test_trims_long_history(self, config, secrets):
        agent = DefaultAgent(config=config, secrets=secrets, max_history_messages=4)
        for i in range(10):
            agent.message_history.append({"role": "user", "content": f"msg {i}"})
        messages = agent._build_messages()
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) == 4
        assert user_msgs[0]["content"] == "msg 6"
