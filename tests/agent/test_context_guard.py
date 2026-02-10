"""Tests for context window guard in DefaultAgent."""

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


class TestHasSufficientContext:
    @patch("roshni.core.llm.token_management.get_model_context_limit", return_value=128000)
    @patch("roshni.core.llm.token_management.estimate_token_count", return_value=1000)
    def test_plenty_of_space(self, mock_estimate, mock_limit, config, secrets):
        agent = DefaultAgent(config=config, secrets=secrets, min_context_tokens=4096)
        messages = [{"role": "user", "content": "hi"}]
        assert agent._has_sufficient_context(messages) is True

    @patch("roshni.core.llm.token_management.get_model_context_limit", return_value=10000)
    @patch("roshni.core.llm.token_management.estimate_token_count", return_value=9000)
    def test_context_nearly_full(self, mock_estimate, mock_limit, config, secrets):
        agent = DefaultAgent(config=config, secrets=secrets, min_context_tokens=4096)
        messages = [{"role": "user", "content": "hi"}]
        # 9000 > 10000 - 4096 = 5904
        assert agent._has_sufficient_context(messages) is False

    @patch("roshni.core.llm.token_management.get_model_context_limit", return_value=10000)
    @patch("roshni.core.llm.token_management.estimate_token_count", return_value=5000)
    def test_just_enough_space(self, mock_estimate, mock_limit, config, secrets):
        agent = DefaultAgent(config=config, secrets=secrets, min_context_tokens=4096)
        messages = [{"role": "user", "content": "hi"}]
        # 5000 <= 10000 - 4096 = 5904
        assert agent._has_sufficient_context(messages) is True


class TestContextGuardInToolLoop:
    @patch("roshni.core.llm.token_management.get_model_context_limit", return_value=10000)
    @patch("roshni.core.llm.token_management.estimate_token_count", return_value=9500)
    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_tool_loop_breaks_when_context_full(self, mock_completion, mock_estimate, mock_limit, config, secrets):
        """When context is nearly full, tool loop should break without calling LLM."""
        agent = DefaultAgent(config=config, secrets=secrets, min_context_tokens=4096)
        agent.message_history.append({"role": "user", "content": "test"})

        tool_call_log: list[dict] = []
        agent._run_tool_loop(
            tool_schemas=None,
            tool_call_log=tool_call_log,
            max_iterations=5,
            on_tool_start=None,
        )

        # LLM should NOT have been called
        mock_completion.assert_not_called()

    @patch("roshni.core.llm.token_management.get_model_context_limit", return_value=128000)
    @patch("roshni.core.llm.token_management.estimate_token_count", return_value=1000)
    @patch("roshni.core.llm.client.LLMClient.completion")
    def test_tool_loop_proceeds_when_space_available(self, mock_completion, mock_estimate, mock_limit, config, secrets):
        """When there's plenty of context space, tool loop should proceed normally."""
        msg = MagicMock()
        msg.content = "Hello"
        msg.tool_calls = None
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock(choices=[choice])
        resp.usage = None
        mock_completion.return_value = resp

        agent = DefaultAgent(config=config, secrets=secrets, min_context_tokens=4096)
        agent.message_history.append({"role": "user", "content": "test"})

        tool_call_log: list[dict] = []
        agent._run_tool_loop(
            tool_schemas=None,
            tool_call_log=tool_call_log,
            max_iterations=5,
            on_tool_start=None,
        )

        mock_completion.assert_called_once()
