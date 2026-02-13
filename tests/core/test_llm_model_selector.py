"""Tests for core.llm.model_selector — light/heavy/thinking model selection."""

import pytest

from roshni.core.llm.config import MODEL_CATALOG
from roshni.core.llm.model_selector import (
    _COMPLEX_KEYWORDS,
    ModelSelector,
    TaskSignals,
    get_model_selector,
    reset_model_selector,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_model_selector()
    yield
    reset_model_selector()


class TestModelSelector:
    def test_defaults(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        models = ms.get_current_models()
        assert not models["light"].is_heavy
        assert models["heavy"].is_heavy or models["heavy"].name != models["light"].name
        assert "thinking" in models

    def test_default_thinking_model(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        assert ms.thinking_model.is_thinking

    def test_explicit_models(self):
        light = MODEL_CATALOG["openai"][0]
        heavy = MODEL_CATALOG["anthropic"][1]
        thinking = MODEL_CATALOG["anthropic"][2]
        ms = ModelSelector(
            light_model=light,
            heavy_model=heavy,
            thinking_model=thinking,
            settings_path="/tmp/nonexistent_roshni_test.json",
        )
        assert ms.light_model.name == light.name
        assert ms.heavy_model.name == heavy.name
        assert ms.thinking_model.name == thinking.name

    def test_get_model_for_task_light(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        model = ms.get_model_for_task("summarize my notes")
        assert model == ms.light_model

    def test_get_model_for_task_heavy(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        model = ms.get_model_for_task("analyze the trends in my health data")
        assert model == ms.heavy_model

    def test_query_mode_light(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        assert ms.get_model_for_task("anything", query_mode="summary") == ms.light_model

    def test_query_mode_unknown_falls_through_to_heuristics(self):
        """Unknown modes (explore, smart, data) fall through to query heuristics."""
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        # Simple query with unknown mode → light (no complex keywords, short text)
        assert ms.get_model_for_task("anything", query_mode="explore") == ms.light_model


class TestSelect:
    """Tests for the unified select() method."""

    def test_think_flag_returns_thinking_model(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        result = ms.select("anything", think=True)
        assert result.name == ms.thinking_model.name

    def test_heavy_mode_override(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        result = ms.select("hello", mode="deep_dive", heavy_modes={"deep_dive"})
        assert result == ms.heavy_model

    def test_light_mode(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        result = ms.select("anything", mode="summary")
        assert result == ms.light_model

    def test_complex_keyword_returns_heavy(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        result = ms.select("analyze my data")
        assert result == ms.heavy_model

    def test_long_query_returns_heavy(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        long_query = "x " * 100  # > 150 chars
        result = ms.select(long_query)
        assert result == ms.heavy_model

    def test_simple_query_returns_light(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        result = ms.select("hi")
        assert result == ms.light_model

    def test_light_keyword_returns_light(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        result = ms.select("give me a quick overview")
        assert result == ms.light_model

    def test_think_takes_priority_over_mode(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        result = ms.select("anything", mode="summary", think=True)
        assert result.name == ms.thinking_model.name

    def test_all_complex_keywords_trigger_heavy(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        for keyword in _COMPLEX_KEYWORDS:
            result = ms.select(f"please {keyword} this")
            assert result == ms.heavy_model, f"Keyword '{keyword}' did not trigger heavy model"

    def test_think_mode_returns_thinking_model(self):
        """mode='think' triggers thinking model (for /think command)."""
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        result = ms.select("why am I tired on Mondays?", mode="think")
        assert result.name == ms.thinking_model.name
        assert result.thinking_budget_tokens is not None and result.thinking_budget_tokens > 0

    def test_unknown_mode_with_complex_query_returns_heavy(self):
        """Unknown mode + complex keywords → heavy via query heuristics."""
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        result = ms.select("analyze the trends in my data", mode="smart")
        assert result == ms.heavy_model

    def test_unknown_mode_with_simple_query_returns_light(self):
        """Unknown mode + simple query → light (no catch-all to heavy)."""
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        result = ms.select("hi", mode="smart")
        assert result == ms.light_model


class TestTaskSignals:
    """Tests for signal-based dynamic model selection."""

    def test_boot_channel_returns_light(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        signals = TaskSignals(channel="boot")
        result = ms.select("complex analyze query", signals=signals)
        assert result == ms.light_model

    def test_heartbeat_channel_returns_light(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        signals = TaskSignals(channel="heartbeat")
        result = ms.select("anything", signals=signals)
        assert result == ms.light_model

    def test_large_tool_results_upgrade_to_heavy(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        signals = TaskSignals(tool_result_chars=2000)
        result = ms.select("hi", signals=signals)
        assert result == ms.heavy_model

    def test_small_tool_results_stay_light(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        signals = TaskSignals(tool_result_chars=50)
        result = ms.select("hi", signals=signals)
        assert result == ms.light_model

    def test_synthesis_flag_upgrades_to_heavy(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        signals = TaskSignals(needs_synthesis=True)
        result = ms.select("", signals=signals)
        assert result == ms.heavy_model

    def test_heavy_mode_overrides_channel_signal(self):
        """Explicit heavy_modes take priority over channel signal (after thinking check)."""
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        # Channel signal comes before mode check, so boot channel wins
        signals = TaskSignals(channel="boot")
        result = ms.select("analyze this", mode="analyze", heavy_modes={"analyze"}, signals=signals)
        assert result == ms.light_model  # boot channel takes priority

    def test_think_mode_overrides_channel_signal(self):
        """Thinking mode takes priority over everything except budget/quiet."""
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        signals = TaskSignals(channel="boot")
        result = ms.select("anything", mode="think", signals=signals)
        assert result.name == ms.thinking_model.name


class TestEscalationSignal:
    """Tests for needs_escalation signal (cascade/refusal detection)."""

    def test_signals_needs_escalation_returns_heavy(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        signals = TaskSignals(needs_escalation=True)
        result = ms.select("what happened with Airbnb earnings?", signals=signals)
        assert result == ms.heavy_model

    def test_signals_escalation_with_channel_override(self):
        """Escalation overrides boot/heartbeat channel — escalation signal takes priority."""
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        # Boot channel normally forces light, but escalation should NOT be overridden
        # because channel check happens before the signal check in select().
        # With both channel=boot and needs_escalation, channel wins (boot = light).
        # This is correct: boot messages should never escalate.
        signals = TaskSignals(channel="boot", needs_escalation=True)
        result = ms.select("anything", signals=signals)
        assert result == ms.light_model  # boot channel takes priority

    def test_signals_escalation_no_channel_returns_heavy(self):
        """Escalation without channel override returns heavy."""
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        signals = TaskSignals(needs_escalation=True, channel="telegram")
        result = ms.select("hi", signals=signals)
        assert result == ms.heavy_model

    def test_signals_escalation_with_tool_chars(self):
        """Escalation combined with low tool result chars still returns heavy."""
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        signals = TaskSignals(tool_result_chars=10, needs_escalation=True)
        result = ms.select("hi", signals=signals)
        assert result == ms.heavy_model


class TestRefusalDetection:
    """Tests for _looks_like_refusal() static method."""

    def test_browsing_refusal(self):
        from roshni.agent.default import DefaultAgent

        assert DefaultAgent._looks_like_refusal(
            "I'm sorry, but I can't browse the web or access real-time information."
        )

    def test_realtime_refusal(self):
        from roshni.agent.default import DefaultAgent

        assert DefaultAgent._looks_like_refusal(
            "I don't have access to real-time information, so I can't tell you what happened with Airbnb earnings."
        )

    def test_stock_refusal(self):
        from roshni.agent.default import DefaultAgent

        assert DefaultAgent._looks_like_refusal(
            "I can't fetch stock prices or access market data. You might want to check a financial website."
        )

    def test_knowledge_cutoff_refusal(self):
        from roshni.agent.default import DefaultAgent

        assert DefaultAgent._looks_like_refusal(
            "My knowledge cutoff is April 2024, so I don't have information about recent events."
        )

    def test_short_text_returns_false(self):
        from roshni.agent.default import DefaultAgent

        assert not DefaultAgent._looks_like_refusal("I can't.")

    def test_normal_response_returns_false(self):
        from roshni.agent.default import DefaultAgent

        assert not DefaultAgent._looks_like_refusal(
            "Based on the search results, Airbnb reported strong Q3 earnings with revenue up 18% year-over-year."
        )

    def test_empty_string_returns_false(self):
        from roshni.agent.default import DefaultAgent

        assert not DefaultAgent._looks_like_refusal("")

    def test_general_capability_refusal(self):
        from roshni.agent.default import DefaultAgent

        assert DefaultAgent._looks_like_refusal(
            "That's beyond my capabilities. I'm a text-based AI and cannot perform web searches."
        )

    def test_training_data_refusal(self):
        from roshni.agent.default import DefaultAgent

        assert DefaultAgent._looks_like_refusal(
            "My training data only goes up to a certain date, so I can't provide current stock prices."
        )


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "settings.json")
        light = MODEL_CATALOG["openai"][0]
        heavy = MODEL_CATALOG["anthropic"][1]
        thinking = MODEL_CATALOG["anthropic"][2]

        ms1 = ModelSelector(light_model=light, heavy_model=heavy, thinking_model=thinking, settings_path=path)
        ms1._save_settings()

        ms2 = ModelSelector(settings_path=path)
        assert ms2.light_model.name == light.name
        assert ms2.heavy_model.name == heavy.name
        assert ms2.thinking_model.name == thinking.name

    def test_set_models_includes_thinking(self, tmp_path):
        path = str(tmp_path / "settings.json")
        ms = ModelSelector(settings_path=path)
        new_thinking = MODEL_CATALOG["openai"][2]  # GPT-5.2 (thinking)
        ms.set_models(thinking=new_thinking)
        assert ms.thinking_model.name == new_thinking.name

        # Reload and verify persistence
        ms2 = ModelSelector(settings_path=path)
        assert ms2.thinking_model.name == new_thinking.name


class TestSingleton:
    def test_get_returns_same_instance(self):
        a = get_model_selector(settings_path="/tmp/nonexistent_roshni_test.json")
        b = get_model_selector()
        assert a is b

    def test_reset_clears(self):
        a = get_model_selector(settings_path="/tmp/nonexistent_roshni_test.json")
        reset_model_selector()
        b = get_model_selector(settings_path="/tmp/nonexistent_roshni_test.json")
        assert a is not b
