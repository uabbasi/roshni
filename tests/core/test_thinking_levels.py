"""Tests for thinking level support (ThinkingLevel enum, ModelSelector, LLMClient)."""

from roshni.core.llm.config import THINKING_BUDGET_MAP, ModelConfig, ThinkingLevel
from roshni.core.llm.model_selector import ModelSelector


class TestThinkingLevel:
    def test_off_is_zero(self):
        assert ThinkingLevel.OFF == 0

    def test_ordering(self):
        assert ThinkingLevel.OFF < ThinkingLevel.LOW < ThinkingLevel.MEDIUM < ThinkingLevel.HIGH

    def test_budget_map_has_all_levels(self):
        for level in ThinkingLevel:
            assert level in THINKING_BUDGET_MAP

    def test_budget_values(self):
        assert THINKING_BUDGET_MAP[ThinkingLevel.OFF] == 0
        assert THINKING_BUDGET_MAP[ThinkingLevel.LOW] == 1024
        assert THINKING_BUDGET_MAP[ThinkingLevel.MEDIUM] == 4096
        assert THINKING_BUDGET_MAP[ThinkingLevel.HIGH] == 16384


class TestModelSelectorThinkingLevel:
    def test_thinking_level_off_returns_normal(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        result = ms.select("hi", thinking_level=ThinkingLevel.OFF)
        # Should return light model (default for short query)
        assert result == ms.light_model

    def test_thinking_level_low(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        result = ms.select("anything", thinking_level=ThinkingLevel.LOW)
        assert result.name == ms.thinking_model.name
        assert result.thinking_budget_tokens == 1024

    def test_thinking_level_medium(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        result = ms.select("anything", thinking_level=ThinkingLevel.MEDIUM)
        assert result.name == ms.thinking_model.name
        assert result.thinking_budget_tokens == 4096

    def test_thinking_level_high(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        result = ms.select("anything", thinking_level=ThinkingLevel.HIGH)
        assert result.name == ms.thinking_model.name
        assert result.thinking_budget_tokens == 16384

    def test_think_flag_with_no_level_uses_medium(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        result = ms.select("anything", think=True)
        assert result.name == ms.thinking_model.name
        assert result.thinking_budget_tokens == 4096  # MEDIUM default


class TestModelConfigThinkingBudget:
    def test_default_is_none(self):
        mc = ModelConfig(name="test", display_name="Test")
        assert mc.thinking_budget_tokens is None

    def test_can_set(self):
        mc = ModelConfig(name="test", display_name="Test", thinking_budget_tokens=8192)
        assert mc.thinking_budget_tokens == 8192


class TestLLMClientThinkingKwargs:
    def test_thinking_passed_to_completion_kwargs(self):
        from roshni.core.llm.client import LLMClient

        client = LLMClient(model="gpt-5.2-chat-latest", provider="openai")
        kwargs = client._build_completion_kwargs(
            [{"role": "user", "content": "hi"}],
            thinking={"type": "enabled", "budget_tokens": 4096},
        )
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 4096}

    def test_no_thinking_omits_key(self):
        from roshni.core.llm.client import LLMClient

        client = LLMClient(model="gpt-5.2-chat-latest", provider="openai")
        kwargs = client._build_completion_kwargs(
            [{"role": "user", "content": "hi"}],
        )
        assert "thinking" not in kwargs
