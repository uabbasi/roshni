"""Tests for core.llm.model_selector — light/heavy/thinking model selection."""

import pytest

from roshni.core.llm.config import MODEL_CATALOG
from roshni.core.llm.model_selector import (
    _COMPLEX_KEYWORDS,
    ModelSelector,
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

    def test_query_mode_heavy(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        assert ms.get_model_for_task("anything", query_mode="explore") == ms.heavy_model


class TestSelect:
    """Tests for the unified select() method."""

    def test_think_flag_returns_thinking_model(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        result = ms.select("anything", think=True)
        assert result == ms.thinking_model

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
        assert result == ms.thinking_model

    def test_all_complex_keywords_trigger_heavy(self):
        ms = ModelSelector(settings_path="/tmp/nonexistent_roshni_test.json")
        for keyword in _COMPLEX_KEYWORDS:
            result = ms.select(f"please {keyword} this")
            assert result == ms.heavy_model, f"Keyword '{keyword}' did not trigger heavy model"


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
