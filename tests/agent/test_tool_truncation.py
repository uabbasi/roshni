"""Tests for tool result truncation in DefaultAgent."""

from unittest.mock import MagicMock, patch

import pytest

from roshni.agent.default import DefaultAgent
from roshni.core.config import Config
from roshni.core.secrets import SecretsManager


@pytest.fixture
def config(tmp_dir):
    return Config(
        data_dir=tmp_dir,
        defaults={"llm": {"provider": "openai", "model": "gpt-5.2-chat-latest"}},
    )


@pytest.fixture
def secrets():
    return SecretsManager(providers=[])


class TestTruncateToolResult:
    def test_short_result_unchanged(self):
        result = DefaultAgent._truncate_tool_result("short", 4000)
        assert result == "short"

    def test_exact_threshold_unchanged(self):
        text = "a" * 4000
        result = DefaultAgent._truncate_tool_result(text, 4000)
        assert result == text

    def test_over_threshold_truncated(self):
        text = "a" * 5000
        result = DefaultAgent._truncate_tool_result(text, 4000)
        assert len(result) < 5000
        assert result.startswith("a" * 4000)
        assert "[TRUNCATED: 5000 chars, showing first 4000]" in result

    def test_truncation_marker_includes_original_size(self):
        text = "x" * 10000
        result = DefaultAgent._truncate_tool_result(text, 100)
        assert "10000 chars" in result
        assert "showing first 100" in result

    def test_empty_string(self):
        result = DefaultAgent._truncate_tool_result("", 4000)
        assert result == ""


class TestToolTruncationIntegration:
    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_large_tool_result_truncated_in_history(self, mock_completion, config, secrets):
        """Tool results exceeding max_tool_result_chars should be truncated in both log and history."""
        large_tool = MagicMock()
        large_tool.name = "big_tool"
        large_tool.description = "Returns lots of data"
        large_tool.parameters = {"type": "object", "properties": {}, "required": []}
        large_tool.permission = "read"
        large_tool.requires_approval = False
        large_tool.needs_approval.return_value = False
        large_tool.execute.return_value = "x" * 8000
        large_tool.to_litellm_schema.return_value = {
            "type": "function",
            "function": {
                "name": "big_tool",
                "description": "big",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }

        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "big_tool"
        tool_call.function.arguments = "{}"

        msg = MagicMock()
        msg.content = None
        msg.tool_calls = [tool_call]
        choice = MagicMock()
        choice.message = msg
        first_resp = MagicMock(choices=[choice])
        first_resp.usage = None

        msg2 = MagicMock()
        msg2.content = "Done"
        msg2.tool_calls = None
        choice2 = MagicMock()
        choice2.message = msg2
        second_resp = MagicMock(choices=[choice2])
        second_resp.usage = None

        mock_completion.side_effect = [first_resp, second_resp]

        agent = DefaultAgent(config=config, secrets=secrets, tools=[large_tool], max_tool_result_chars=100)
        result = agent.chat("run big_tool")

        # The tool call log should have truncated result
        assert len(result.tool_calls) == 1
        assert "[TRUNCATED:" in result.tool_calls[0]["result"]

        # History tool message should also be truncated
        tool_msgs = [m for m in agent.message_history if m.get("role") == "tool"]
        assert tool_msgs
        assert "[TRUNCATED:" in tool_msgs[0]["content"]
