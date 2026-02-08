"""Tests for core.llm.model_selector — light/heavy model selection."""

import pytest

from roshni.core.llm.config import MODEL_CATALOG
from roshni.core.llm.model_selector import ModelSelector, get_model_selector, reset_model_selector


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

    def test_explicit_models(self):
        light = MODEL_CATALOG["openai"][0]
        heavy = MODEL_CATALOG["anthropic"][-1]
        ms = ModelSelector(light_model=light, heavy_model=heavy, settings_path="/tmp/nonexistent_roshni_test.json")
        assert ms.light_model.name == light.name
        assert ms.heavy_model.name == heavy.name

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


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "settings.json")
        light = MODEL_CATALOG["openai"][0]
        heavy = MODEL_CATALOG["anthropic"][-1]

        ms1 = ModelSelector(light_model=light, heavy_model=heavy, settings_path=path)
        ms1._save_settings()

        ms2 = ModelSelector(settings_path=path)
        assert ms2.light_model.name == light.name
        assert ms2.heavy_model.name == heavy.name


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
