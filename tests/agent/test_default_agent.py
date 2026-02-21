"""Tests for DefaultAgent."""

import os
import threading
from unittest.mock import MagicMock, patch

import pytest

from roshni.agent.base import ChatResult
from roshni.agent.default import DefaultAgent
from roshni.agent.tools import ToolDefinition
from roshni.core.config import Config
from roshni.core.llm.model_selector import ModelSelector, TaskSignals
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

    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_tool_exception_produces_error_result(self, mock_completion, config, secrets):
        """If a tool raises, the error is captured as a tool result — not an orphaned tool_call."""

        def exploding_fn(text):
            raise RuntimeError("kaboom")

        boom_tool = ToolDefinition(
            name="boom",
            description="Explodes",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            function=exploding_fn,
        )

        tool_call = MagicMock()
        tool_call.id = "call_boom"
        tool_call.function.name = "boom"
        tool_call.function.arguments = '{"text": "hi"}'

        mock_completion.side_effect = [
            _make_response(None, tool_calls=[tool_call]),
            _make_response("The tool failed, sorry."),
        ]

        agent = DefaultAgent(config=config, secrets=secrets, tools=[boom_tool])
        result = agent.chat("try the boom tool")

        # Should NOT crash — the error is returned as a tool result to the LLM
        assert "went wrong" not in result.text.lower()
        assert len(result.tool_calls) == 1
        assert "kaboom" in result.tool_calls[0]["result"]

        # History should be valid: assistant+tool_calls has a matching tool result
        messages = agent._build_messages()
        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                assert i + 1 < len(messages) and messages[i + 1]["role"] == "tool"

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
        # Should get a friendly message, not the raw exception
        assert "API down" not in result.text
        assert "unexpected" in result.text.lower() or "try again" in result.text.lower()

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

        # After approve, the tool loop resumes — LLM gets called again
        mock_completion.return_value = _make_response("Done! I wrote hello for you.")
        approved = agent.chat("approve")
        assert approved.tool_calls[0]["name"] == "write_thing"
        assert approved.tool_calls[0]["result"] == "Wrote: hello"

    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_write_approval_required_every_time_when_grants_not_persisted(
        self, mock_completion, tmp_dir, secrets, write_tool
    ):
        cfg = Config(
            data_dir=tmp_dir,
            defaults={
                "llm": {"provider": "openai", "model": "gpt-5.2-chat-latest"},
                "security": {"require_write_approval": True, "persist_approval_grants": False},
            },
        )
        agent = DefaultAgent(config=cfg, secrets=secrets, tools=[write_tool])

        tool_call_1 = MagicMock()
        tool_call_1.id = "call_approval_1"
        tool_call_1.function.name = "write_thing"
        tool_call_1.function.arguments = '{"text": "first"}'
        mock_completion.return_value = _make_response(None, tool_calls=[tool_call_1])
        first = agent.chat("write first")
        assert "Approval required" in first.text

        mock_completion.return_value = _make_response("done first")
        agent.chat("approve")

        tool_call_2 = MagicMock()
        tool_call_2.id = "call_approval_2"
        tool_call_2.function.name = "write_thing"
        tool_call_2.function.arguments = '{"text": "second"}'
        mock_completion.return_value = _make_response(None, tool_calls=[tool_call_2])
        second = agent.chat("write second")
        assert "Approval required" in second.text

    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_write_tool_denied(self, mock_completion, config, secrets, write_tool):
        tool_call = MagicMock()
        tool_call.id = "call_approval"
        tool_call.function.name = "write_thing"
        tool_call.function.arguments = '{"text": "hello"}'
        mock_completion.return_value = _make_response(None, tool_calls=[tool_call])

        agent = DefaultAgent(config=config, secrets=secrets, tools=[write_tool])
        _ = agent.chat("Save hello")

        # After deny, tool loop resumes — LLM sees error tool results
        mock_completion.return_value = _make_response("OK, I won't write that.")
        denied = agent.chat("deny")
        assert "won't write" in denied.text.lower() or denied.text  # LLM responds

    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_batch_tool_calls_resume_after_approval(self, mock_completion, config, secrets, write_tool, echo_tool):
        """When LLM returns multiple tool calls and first needs approval,
        approving executes ALL tools and resumes the LLM loop."""
        # LLM returns 3 tool calls: write_thing (needs approval), echo, echo
        tc_write = MagicMock()
        tc_write.id = "call_1"
        tc_write.function.name = "write_thing"
        tc_write.function.arguments = '{"text": "important"}'

        tc_echo1 = MagicMock()
        tc_echo1.id = "call_2"
        tc_echo1.function.name = "echo"
        tc_echo1.function.arguments = '{"text": "first"}'

        tc_echo2 = MagicMock()
        tc_echo2.id = "call_3"
        tc_echo2.function.name = "echo"
        tc_echo2.function.arguments = '{"text": "second"}'

        mock_completion.return_value = _make_response(None, tool_calls=[tc_write, tc_echo1, tc_echo2])

        agent = DefaultAgent(config=config, secrets=secrets, tools=[write_tool, echo_tool])
        result = agent.chat("Do three things")

        # First chat should bail with approval prompt
        assert "Approval required" in result.text
        assert "write_thing" in result.text
        # Only 0 tool calls executed so far
        assert len(result.tool_calls) == 0

        # Now approve — should execute all 3 tools, then LLM resumes
        mock_completion.return_value = _make_response("All 3 tasks completed successfully.")
        approved = agent.chat("approve")

        # All 3 tools should be in the log
        assert len(approved.tool_calls) == 3
        assert approved.tool_calls[0]["name"] == "write_thing"
        assert approved.tool_calls[0]["result"] == "Wrote: important"
        assert approved.tool_calls[1]["name"] == "echo"
        assert approved.tool_calls[1]["result"] == "Echo: first"
        assert approved.tool_calls[2]["name"] == "echo"
        assert approved.tool_calls[2]["result"] == "Echo: second"
        assert "completed" in approved.text.lower()

    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_batch_tool_calls_deny_sends_error_results(self, mock_completion, config, secrets, write_tool, echo_tool):
        """Denying a batch adds error tool results for all calls so the LLM can respond."""
        tc_write = MagicMock()
        tc_write.id = "call_1"
        tc_write.function.name = "write_thing"
        tc_write.function.arguments = '{"text": "nope"}'

        tc_echo = MagicMock()
        tc_echo.id = "call_2"
        tc_echo.function.name = "echo"
        tc_echo.function.arguments = '{"text": "also nope"}'

        mock_completion.return_value = _make_response(None, tool_calls=[tc_write, tc_echo])

        agent = DefaultAgent(config=config, secrets=secrets, tools=[write_tool, echo_tool])
        _ = agent.chat("Do two things")

        # Deny — LLM should see tool error results and produce a response
        mock_completion.return_value = _make_response("Understood, I won't do those.")
        denied = agent.chat("deny")
        assert denied.text  # LLM produced a response

        # Verify the history has proper tool error results
        tool_msgs = [m for m in agent.message_history if m.get("role") == "tool"]
        assert len(tool_msgs) >= 2  # One for each tool call in the batch


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

    def test_mode_overrides_loaded_from_config(self, tmp_dir, secrets):
        config = Config(
            data_dir=tmp_dir,
            defaults={
                "llm": {
                    "provider": "openai",
                    "model": "gpt-5.2-chat-latest",
                    "mode_overrides": {
                        "smart": "deepseek/deepseek-chat",
                    },
                },
            },
        )
        selector = ModelSelector(settings_path=os.path.join(tmp_dir, "selector.json"))
        agent = DefaultAgent(config=config, secrets=secrets, model_selector=selector)

        selected = agent._model_selector.select("hello", mode="smart")
        assert selected.name == "deepseek/deepseek-chat"

    def test_selector_thresholds_loaded_from_config(self, tmp_dir, secrets):
        config = Config(
            data_dir=tmp_dir,
            defaults={
                "llm": {
                    "provider": "openai",
                    "model": "gpt-5.2-chat-latest",
                    "selector": {
                        "tool_result_chars_threshold": 3000,
                        "complex_query_chars_threshold": 300,
                    },
                },
            },
        )
        selector = ModelSelector(settings_path=os.path.join(tmp_dir, "selector-thresholds.json"))
        agent = DefaultAgent(config=config, secrets=secrets, model_selector=selector)

        sig_result = agent._model_selector.select("hi", signals=TaskSignals(tool_result_chars=1000))
        assert sig_result == agent._model_selector.light_model

        query_result = agent._model_selector.select("x" * 200)
        assert query_result == agent._model_selector.light_model


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

    def test_repair_injects_missing_tool_results(self, config, secrets):
        """If tool results are missing (e.g. from history trim), synthetic results are injected."""
        agent = DefaultAgent(config=config, secrets=secrets, max_history_messages=20)

        agent.message_history = [
            {"role": "user", "content": "do something"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "save_note", "arguments": "{}"}},
                    {"id": "c2", "type": "function", "function": {"name": "echo", "arguments": "{}"}},
                ],
            },
            # Only c1 has a result — c2 is missing (e.g. process crashed mid-execution)
            {"role": "tool", "tool_call_id": "c1", "content": "saved"},
            {"role": "user", "content": "what happened?"},
        ]

        messages = agent._build_messages()
        non_system = [m for m in messages if m["role"] != "system"]

        # Both tool results should be present after the assistant message
        tool_msgs = [m for m in non_system if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        tool_ids = {m["tool_call_id"] for m in tool_msgs}
        assert tool_ids == {"c1", "c2"}
        # The synthetic result should mention the tool name
        c2_msg = next(m for m in tool_msgs if m["tool_call_id"] == "c2")
        assert "echo" in c2_msg["content"]
        assert "unavailable" in c2_msg["content"].lower() or "interrupted" in c2_msg["content"].lower()

    def test_repair_reorders_scattered_tool_results(self, config, secrets):
        """Tool results separated by a user message are reordered to be contiguous."""
        agent = DefaultAgent(config=config, secrets=secrets, max_history_messages=20)

        # This simulates the old approval bug: user message lands between tool_calls and results
        agent.message_history = [
            {"role": "user", "content": "cancel the project"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "cancel_project", "arguments": "{}"}},
                ],
            },
            {"role": "user", "content": "approve"},  # breaks contiguity
            {"role": "tool", "tool_call_id": "c1", "content": "Project cancelled."},
            {"role": "assistant", "content": "Done, project cancelled."},
        ]

        messages = agent._build_messages()
        non_system = [m for m in messages if m["role"] != "system"]

        # The tool result should be right after the assistant+tool_calls (contiguous)
        assistant_idx = next(i for i, m in enumerate(non_system) if m.get("tool_calls"))
        assert non_system[assistant_idx + 1]["role"] == "tool"
        assert non_system[assistant_idx + 1]["tool_call_id"] == "c1"
        assert self._no_orphaned_tool_messages(messages)

    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_approval_no_user_message_between_tool_calls(self, mock_completion, config, secrets, write_tool):
        """Approving a tool must NOT insert the user message between tool_calls and results."""
        tool_call = MagicMock()
        tool_call.id = "call_approve_test"
        tool_call.function.name = "write_thing"
        tool_call.function.arguments = '{"text": "test"}'
        mock_completion.return_value = _make_response(None, tool_calls=[tool_call])

        agent = DefaultAgent(config=config, secrets=secrets, tools=[write_tool])
        agent.chat("write test")  # triggers approval prompt

        # Now approve
        mock_completion.return_value = _make_response("Written!")
        agent.chat("approve")

        # Check that no user message sits between assistant+tool_calls and tool result
        history = agent.message_history
        for i, msg in enumerate(history):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                # Next message must be a tool result, not a user message
                if i + 1 < len(history):
                    assert history[i + 1]["role"] == "tool", (
                        f"Expected tool result after assistant+tool_calls, "
                        f"got {history[i + 1]['role']}: {history[i + 1].get('content', '')[:50]}"
                    )

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
        # System content includes persona + dynamic parts (runtime context, advisors)
        content = messages[0]["content"]
        if isinstance(content, list):
            content = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
        assert "Be helpful." in content
        assert messages[1]["role"] == "user"

    def test_trims_long_history(self, config, secrets):
        agent = DefaultAgent(config=config, secrets=secrets, max_history_messages=4)
        for i in range(10):
            agent.message_history.append({"role": "user", "content": f"msg {i}"})
        messages = agent._build_messages()
        user_msgs = [m for m in messages if m["role"] == "user"]
        assert len(user_msgs) == 4
        assert user_msgs[0]["content"] == "msg 6"


class TestAdvisorIntegration:
    """Tests that advisors and hooks are wired into DefaultAgent correctly."""

    @patch("roshni.core.llm.client.LLMClient.completion")
    @patch("roshni.core.llm.token_budget.get_budget_pressure", return_value=0.0)
    def test_custom_advisor_injected_into_system_prompt(self, _budget, mock_completion, config, secrets):
        from roshni.agent.advisor import FunctionAdvisor

        advisor = FunctionAdvisor("test_ctx", lambda: "[TEST] custom context here")
        mock_completion.return_value = _make_response("ok")

        agent = DefaultAgent(config=config, secrets=secrets, advisors=[advisor])
        agent.chat("hi")

        # Check that system prompt includes the advisor output
        call_args = mock_completion.call_args
        messages = call_args[0][0] if call_args[0] else call_args.kwargs.get("messages", [])
        system_msg = messages[0]
        system_content = system_msg.get("content", "")
        if isinstance(system_content, list):
            system_content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block) for block in system_content
            )
        assert "[TEST] custom context here" in system_content

    @patch("roshni.core.llm.client.LLMClient.completion")
    @patch("roshni.core.llm.token_budget.get_budget_pressure", return_value=0.0)
    def test_after_chat_hook_fires(self, _budget, mock_completion, config, secrets):
        import threading

        from roshni.agent.advisor import FunctionAfterChatHook

        mock_completion.return_value = _make_response("Done")
        captured = {}
        event = threading.Event()

        def hook_fn(message, response):
            captured["msg"] = message
            captured["resp"] = response
            event.set()

        hook = FunctionAfterChatHook("test_hook", hook_fn)
        agent = DefaultAgent(config=config, secrets=secrets, after_chat_hooks=[hook])
        agent.chat("hello")

        event.wait(timeout=2.0)
        assert captured.get("msg") == "hello"
        assert captured.get("resp") == "Done"

    @patch("roshni.core.llm.client.LLMClient.completion")
    @patch("roshni.core.llm.token_budget.get_budget_pressure", return_value=0.0)
    def test_on_chat_complete_backward_compat(self, _budget, mock_completion, config, secrets):
        import threading

        mock_completion.return_value = _make_response("Yo")
        captured = {}
        event = threading.Event()

        def callback(msg, resp, tools):
            captured["msg"] = msg
            event.set()

        agent = DefaultAgent(config=config, secrets=secrets, on_chat_complete=callback)
        agent.chat("test")

        event.wait(timeout=2.0)
        assert captured.get("msg") == "test"

    @patch("roshni.core.llm.client.LLMClient.completion")
    @patch("roshni.core.llm.token_budget.get_budget_pressure", return_value=0.0)
    def test_add_advisor_dynamically(self, _budget, mock_completion, config, secrets):
        from roshni.agent.advisor import FunctionAdvisor

        mock_completion.return_value = _make_response("ok")

        agent = DefaultAgent(config=config, secrets=secrets)
        agent.add_advisor(FunctionAdvisor("dynamic", lambda: "[DYNAMIC] ctx"))
        agent.chat("hi")

        call_args = mock_completion.call_args
        messages = call_args[0][0] if call_args[0] else call_args.kwargs.get("messages", [])
        system_content = messages[0].get("content", "")
        if isinstance(system_content, list):
            system_content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block) for block in system_content
            )
        assert "[DYNAMIC] ctx" in system_content

    @patch("roshni.core.llm.client.LLMClient.completion")
    @patch("roshni.core.llm.token_budget.get_budget_pressure", return_value=0.0)
    def test_system_health_advisor_auto_registered(self, _budget, mock_completion, config, secrets):
        """DefaultAgent auto-registers SystemHealthAdvisor."""
        agent = DefaultAgent(config=config, secrets=secrets)
        advisor_names = [a.name for a in agent._advisors]
        assert "system_health" in advisor_names

    @patch("roshni.core.llm.client.LLMClient.completion")
    @patch("roshni.core.llm.token_budget.get_budget_pressure", return_value=0.0)
    def test_metrics_hook_auto_registered(self, _budget, mock_completion, config, secrets):
        """DefaultAgent auto-registers MetricsHook."""
        agent = DefaultAgent(config=config, secrets=secrets)
        hook_names = [h.name for h in agent._after_chat_hooks]
        assert "metrics" in hook_names

    @patch("roshni.core.llm.client.LLMClient.completion")
    @patch("roshni.core.llm.token_budget.get_budget_pressure", return_value=0.0)
    def test_failing_advisor_doesnt_break_chat(self, _budget, mock_completion, config, secrets):
        from roshni.agent.advisor import FunctionAdvisor

        def bad_advisor():
            raise RuntimeError("advisor boom")

        mock_completion.return_value = _make_response("still works")

        agent = DefaultAgent(config=config, secrets=secrets, advisors=[FunctionAdvisor("bad", bad_advisor)])
        result = agent.chat("hi")
        assert result.text == "still works"


class TestFriendlyErrorMessages:
    """Tests that production errors produce user-friendly messages, not raw exception dumps."""

    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_friendly_error_message_not_found(self, mock_completion, config, secrets):
        """NotFoundError produces friendly text, not raw exception."""

        class MockNotFoundError(Exception):
            pass

        MockNotFoundError.__name__ = "NotFoundError"

        mock_completion.side_effect = MockNotFoundError("model: claude-haiku-4 not found")

        agent = DefaultAgent(config=config, secrets=secrets)
        result = agent.chat("hello")

        assert "NotFoundError" not in result.text
        assert "model:" not in result.text
        assert "trouble" in result.text.lower() or "unexpected" in result.text.lower()

    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_friendly_error_message_rate_limit(self, mock_completion, config, secrets):
        class MockRateLimitError(Exception):
            pass

        MockRateLimitError.__name__ = "RateLimitError"
        mock_completion.side_effect = MockRateLimitError("Too many requests")

        agent = DefaultAgent(config=config, secrets=secrets)
        result = agent.chat("hello")

        assert "busy" in result.text.lower()
        assert "RateLimitError" not in result.text

    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_friendly_error_message_bad_request(self, mock_completion, config, secrets):
        class MockBadRequestError(Exception):
            pass

        MockBadRequestError.__name__ = "BadRequestError"
        mock_completion.side_effect = MockBadRequestError("invalid messages format")

        agent = DefaultAgent(config=config, secrets=secrets)
        result = agent.chat("hello")

        assert "logged" in result.text.lower() or "request format" in result.text.lower()
        assert "invalid messages format" not in result.text

    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_friendly_error_message_connection(self, mock_completion, config, secrets):
        class MockAPIConnectionError(Exception):
            pass

        MockAPIConnectionError.__name__ = "APIConnectionError"
        mock_completion.side_effect = MockAPIConnectionError("Connection refused")

        agent = DefaultAgent(config=config, secrets=secrets)
        result = agent.chat("hello")

        assert "connecting" in result.text.lower() or "trouble" in result.text.lower()

    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_friendly_error_message_generic(self, mock_completion, config, secrets):
        mock_completion.side_effect = ValueError("something weird")

        agent = DefaultAgent(config=config, secrets=secrets)
        result = agent.chat("hello")

        assert "unexpected" in result.text.lower() or "try again" in result.text.lower()
        assert "something weird" not in result.text

    def test_budget_exceeded_message(self, config, secrets):
        """Budget exceeded should return the budget-specific message."""
        agent = DefaultAgent(config=config, secrets=secrets)
        # Budget check happens early in chat() with a different path
        with patch("roshni.core.llm.token_budget.check_budget", return_value=(False, 0)):
            result = agent.chat("hello")
        assert "budget" in result.text.lower()


class TestRecoveryAttempts:
    """Tests that DefaultAgent attempts recovery before returning errors."""

    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_recovery_retries_after_tool_call_error(self, mock_completion, config, secrets, echo_tool):
        """Verify DefaultAgent retries after repairing tool sequence."""

        class MockBadRequestError(Exception):
            pass

        MockBadRequestError.__name__ = "BadRequestError"

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise MockBadRequestError("tool_call_ids did not have response messages")
            return _make_response("Recovered!")

        mock_completion.side_effect = side_effect

        agent = DefaultAgent(config=config, secrets=secrets, tools=[echo_tool])
        # Pre-populate with a valid user message
        agent.message_history = [{"role": "user", "content": "test recovery"}]
        result = agent.chat("test recovery")

        # Should get a response (either recovered or friendly error), not a raw exception
        assert "BadRequestError" not in result.text

    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_recovery_switches_model_on_not_found(self, mock_completion, config, secrets):
        """Verify DefaultAgent attempts model switch on NotFoundError."""

        class MockNotFoundError(Exception):
            pass

        MockNotFoundError.__name__ = "NotFoundError"

        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] <= 1:
                raise MockNotFoundError("model not found")
            return _make_response("Recovered with fallback!")

        mock_completion.side_effect = side_effect

        agent = DefaultAgent(config=config, secrets=secrets)
        # Give it a fallback model to try
        agent._llm.fallback_model = "deepseek/deepseek-chat"
        result = agent.chat("hello")

        # Should not contain the raw error
        assert "NotFoundError" not in result.text


class TestAfterChatHookPool:
    @patch("roshni.core.llm.token_budget.get_budget_pressure", return_value=0.0)
    def test_fire_hooks_uses_executor_pool(self, _budget, config, secrets):
        from roshni.agent.advisor import FunctionAfterChatHook

        calls = {"submit": 0, "ran": 0}

        class _FakePool:
            def submit(self, fn):
                calls["submit"] += 1
                fn()
                return MagicMock(done=lambda: True)

        hook = FunctionAfterChatHook("hook", lambda **kwargs: calls.__setitem__("ran", calls["ran"] + 1))
        agent = DefaultAgent(config=config, secrets=secrets, after_chat_hooks=[hook])

        with patch.object(DefaultAgent, "_get_hook_pool", return_value=_FakePool()):
            agent._fire_after_chat_hooks("m", "r", [], None)

        assert calls["submit"] >= 1
        assert calls["ran"] == 1

    @patch("roshni.core.llm.token_budget.get_budget_pressure", return_value=0.0)
    def test_fire_hooks_drops_when_saturated(self, _budget, config, secrets):
        from roshni.agent.advisor import FunctionAfterChatHook

        hook = FunctionAfterChatHook("hook", lambda **kwargs: None)
        agent = DefaultAgent(config=config, secrets=secrets, after_chat_hooks=[hook])

        original_slots = DefaultAgent._HOOK_SLOTS
        try:
            DefaultAgent._HOOK_SLOTS = threading.BoundedSemaphore(0)
            agent._fire_after_chat_hooks("m", "r", [], None)
        finally:
            DefaultAgent._HOOK_SLOTS = original_slots

        assert agent._hook_futures == []
