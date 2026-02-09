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


class TestMessageSanitization:
    """Tests that messages sent to the LLM never contain null content or orphaned tool messages.

    These reproduce real failures: OpenAI rejects messages with null content
    and tool-role messages not preceded by an assistant message with tool_calls.
    """

    def _all_messages_have_string_content(self, messages: list[dict]) -> bool:
        """Check that every message has a string content field (not None/missing)."""
        for msg in messages:
            content = msg.get("content")
            if content is None:
                return False
        return True

    def _no_orphaned_tool_messages(self, messages: list[dict]) -> bool:
        """Check that every tool message is preceded by an assistant with tool_calls."""
        for i, msg in enumerate(messages):
            if msg.get("role") == "tool":
                # Walk backward to find the parent assistant message
                found_parent = False
                for j in range(i - 1, -1, -1):
                    if messages[j].get("role") == "assistant" and messages[j].get("tool_calls"):
                        found_parent = True
                        break
                    if messages[j].get("role") in ("user", "system"):
                        break
                if not found_parent:
                    return False
        return True

    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_tool_call_with_null_content_produces_empty_string(self, mock_completion, config, secrets, echo_tool):
        """When LLM returns tool calls with content=None, history must have content=''."""
        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "echo"
        tool_call.function.arguments = '{"text": "hi"}'

        mock_completion.side_effect = [
            _make_response(None, tool_calls=[tool_call]),
            _make_response("Done"),
        ]

        agent = DefaultAgent(config=config, secrets=secrets, tools=[echo_tool])
        agent.chat("test")

        # Every message in history must have string content
        for msg in agent.message_history:
            assert msg.get("content") is not None, f"Message has null content: {msg}"
            assert isinstance(msg["content"], str), f"Content is not a string: {msg}"

    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_multi_turn_tool_calls_no_null_content(self, mock_completion, config, secrets, echo_tool):
        """Multiple tool call rounds should never produce null content in messages."""
        tc1 = MagicMock()
        tc1.id = "call_1"
        tc1.function.name = "echo"
        tc1.function.arguments = '{"text": "first"}'

        tc2 = MagicMock()
        tc2.id = "call_2"
        tc2.function.name = "echo"
        tc2.function.arguments = '{"text": "second"}'

        mock_completion.side_effect = [
            _make_response(None, tool_calls=[tc1]),  # null content
            _make_response(None, tool_calls=[tc2]),  # null content again
            _make_response("All done"),
        ]

        agent = DefaultAgent(config=config, secrets=secrets, tools=[echo_tool])
        agent.chat("test")

        for msg in agent.message_history:
            assert msg.get("content") is not None, f"Null content in: {msg}"

    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_build_messages_no_null_content(self, mock_completion, config, secrets, echo_tool):
        """The messages list passed to the LLM must never have null content."""
        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "echo"
        tool_call.function.arguments = '{"text": "hi"}'

        mock_completion.side_effect = [
            _make_response(None, tool_calls=[tool_call]),
            _make_response("Done"),
        ]

        agent = DefaultAgent(config=config, secrets=secrets, tools=[echo_tool])
        agent.chat("test")

        messages = agent._build_messages()
        assert self._all_messages_have_string_content(messages)

    def test_build_messages_strips_orphaned_tool_messages(self, config, secrets):
        """If history trimming orphans tool messages, _build_messages must strip them."""
        agent = DefaultAgent(config=config, secrets=secrets, max_history_messages=4)

        # Simulate history that starts mid-tool-sequence (as if trimmed)
        agent.message_history = [
            {"role": "tool", "tool_call_id": "call_old", "content": "orphaned result"},
            {"role": "tool", "tool_call_id": "call_old2", "content": "another orphan"},
            {"role": "user", "content": "new message"},
            {"role": "assistant", "content": "response"},
        ]

        messages = agent._build_messages()
        non_system = [m for m in messages if m["role"] != "system"]

        # The orphaned tool messages should be stripped
        assert non_system[0]["role"] == "user"
        assert self._no_orphaned_tool_messages(messages)

    def test_build_messages_strips_orphaned_assistant_with_tool_calls(self, config, secrets):
        """An assistant message with tool_calls but no following tool results should be stripped."""
        agent = DefaultAgent(config=config, secrets=secrets, max_history_messages=4)

        agent.message_history = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "x", "arguments": "{}"},
                    }
                ],
            },
            {"role": "user", "content": "new message"},
            {"role": "assistant", "content": "response"},
        ]

        messages = agent._build_messages()
        non_system = [m for m in messages if m["role"] != "system"]

        # The orphaned assistant+tool_calls should be stripped
        assert non_system[0]["role"] == "user"

    def test_build_messages_preserves_valid_tool_sequence(self, config, secrets):
        """A complete tool sequence (assistant+tool_calls → tool results) should be kept."""
        agent = DefaultAgent(config=config, secrets=secrets, max_history_messages=20)

        agent.message_history = [
            {"role": "user", "content": "do something"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "x", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
            {"role": "assistant", "content": "done"},
        ]

        messages = agent._build_messages()
        non_system = [m for m in messages if m["role"] != "system"]

        assert len(non_system) == 4
        assert non_system[0]["role"] == "user"
        assert non_system[1]["role"] == "assistant"
        assert non_system[2]["role"] == "tool"
        assert non_system[3]["role"] == "assistant"

    def test_build_messages_sanitizes_null_content_anywhere(self, config, secrets):
        """Any message with content=None in history must be sanitized to a string."""
        agent = DefaultAgent(config=config, secrets=secrets, max_history_messages=20)

        # Simulate history with null content scattered throughout
        # (could happen from prior buggy code, deserialization, or provider quirks)
        agent.message_history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": None},  # null!
            {"role": "user", "content": "next"},
            {
                "role": "assistant",
                "content": "ok",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "x", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": None},  # null tool result!
            {"role": "assistant", "content": "done"},
        ]

        messages = agent._build_messages()
        assert self._all_messages_have_string_content(messages)

    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_prior_turn_tool_calls_dont_corrupt_next_turn(self, mock_completion, config, secrets, echo_tool):
        """Tool calls from a prior turn shouldn't cause errors on the next user message.

        Reproduces: boot actuation leaves tool messages in history, then user message fails.
        """
        tool_call = MagicMock()
        tool_call.id = "call_boot"
        tool_call.function.name = "echo"
        tool_call.function.arguments = '{"text": "setup"}'

        # Turn 1: tool call with null content
        mock_completion.side_effect = [
            _make_response(None, tool_calls=[tool_call]),
            _make_response("Boot done"),
        ]

        agent = DefaultAgent(config=config, secrets=secrets, tools=[echo_tool])
        agent.chat("boot prompt")

        # Turn 2: normal user message — should not fail
        mock_completion.side_effect = [_make_response("Hello!")]
        result = agent.chat("hi are you up?")

        # Must succeed and messages must be clean
        assert "went wrong" not in result.text.lower()
        messages = agent._build_messages()
        assert self._all_messages_have_string_content(messages)
        assert self._no_orphaned_tool_messages(messages)


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
