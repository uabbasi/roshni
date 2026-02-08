"""Tests for core.llm.config — model constants and token limits."""

from roshni.core.llm.config import (
    MODEL_CATALOG,
    ModelConfig,
    get_default_model,
    get_model_max_tokens,
    infer_provider,
)


class TestGetDefaultModel:
    def test_known_providers(self):
        assert "gpt" in get_default_model("openai")
        assert "claude" in get_default_model("anthropic") or "anthropic" in get_default_model("anthropic")
        assert "gemini" in get_default_model("gemini")

    def test_unknown_falls_back_to_openai(self):
        assert get_default_model("unknown_provider") == get_default_model("openai")


class TestGetModelMaxTokens:
    def test_exact_match(self):
        assert get_model_max_tokens("o3") == 16_384

    def test_partial_match(self):
        # "gpt-4o" key should match "gpt-4o-mini"
        assert get_model_max_tokens("gpt-4o-mini") == 16_384

    def test_provider_fallback(self):
        assert get_model_max_tokens("some-unknown-model", provider="anthropic") == 4_096

    def test_conservative_fallback(self):
        assert get_model_max_tokens("totally-unknown") == 4_096


class TestInferProvider:
    def test_prefixed_models(self):
        assert infer_provider("anthropic/claude-sonnet-4") == "anthropic"
        assert infer_provider("gemini/gemini-2.5-flash") == "gemini"
        assert infer_provider("ollama/deepseek-r1") == "local"

    def test_openai_models(self):
        assert infer_provider("gpt-4o") == "openai"
        assert infer_provider("o3") == "openai"

    def test_unprefixed_claude(self):
        assert infer_provider("claude-sonnet-4-20250514") == "anthropic"

    def test_default_openai(self):
        assert infer_provider("some-random-model") == "openai"


class TestModelCatalog:
    def test_all_providers_present(self):
        assert set(MODEL_CATALOG.keys()) >= {"anthropic", "openai", "gemini", "local"}

    def test_entries_are_model_config(self):
        for models in MODEL_CATALOG.values():
            for m in models:
                assert isinstance(m, ModelConfig)
                assert m.name
                assert m.display_name
