"""Tests for core.llm.config — model constants and token limits."""

from roshni.core.llm.config import (
    ANTHROPIC_OPUS_MODEL,
    GOOGLE_FLASH_MODEL,
    GOOGLE_PRO_MODEL,
    MODEL_CATALOG,
    PROVIDER_ENV_MAP,
    ModelConfig,
    get_default_model,
    get_model_max_tokens,
    infer_provider,
    resolve_model_name,
)


class TestGetDefaultModel:
    def test_known_providers(self):
        assert "gpt-5" in get_default_model("openai")
        assert "claude" in get_default_model("anthropic") or "anthropic" in get_default_model("anthropic")
        assert "gemini" in get_default_model("gemini")

    def test_new_providers(self):
        assert "deepseek" in get_default_model("deepseek")
        assert "grok" in get_default_model("xai")
        assert "groq" in get_default_model("groq")
        assert "kimi-k2" in get_default_model("kimk2")

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

    def test_new_provider_limits(self):
        assert get_model_max_tokens("deepseek-chat") == 8_192
        assert get_model_max_tokens("grok-4-fast-non-reasoning") == 16_384
        assert get_model_max_tokens("llama-3.3-70b-versatile") == 8_192

    def test_gpt5_limits(self):
        assert get_model_max_tokens("gpt-5.2-chat-latest") == 16_384
        assert get_model_max_tokens("gpt-5.2-pro") == 16_384

    def test_llama4_limits(self):
        assert get_model_max_tokens("llama-4-maverick-17b-128e-instruct") == 8_192

    def test_new_provider_default_limits(self):
        assert get_model_max_tokens("some-unknown", provider="deepseek") == 8_192
        assert get_model_max_tokens("some-unknown", provider="xai") == 8_192
        assert get_model_max_tokens("some-unknown", provider="groq") == 8_192
        assert get_model_max_tokens("some-unknown", provider="kimk2") == 16_384


class TestInferProvider:
    def test_prefixed_models(self):
        assert infer_provider("anthropic/claude-sonnet-4") == "anthropic"
        assert infer_provider("gemini/gemini-2.5-flash") == "gemini"
        assert infer_provider("ollama/deepseek-r1") == "local"

    def test_new_prefixed_models(self):
        assert infer_provider("deepseek/deepseek-chat") == "deepseek"
        assert infer_provider("xai/grok-2") == "xai"
        assert infer_provider("groq/llama-3.3-70b-versatile") == "groq"
        assert infer_provider("openrouter/moonshotai/kimi-k2") == "kimk2"

    def test_openai_models(self):
        assert infer_provider("gpt-4o") == "openai"
        assert infer_provider("o3") == "openai"

    def test_gpt5_inferred_as_openai(self):
        assert infer_provider("gpt-5.2-chat-latest") == "openai"
        assert infer_provider("gpt-5.2-pro") == "openai"
        assert infer_provider("gpt-5.2") == "openai"

    def test_unprefixed_claude(self):
        assert infer_provider("claude-sonnet-4-20250514") == "anthropic"

    def test_unprefixed_grok(self):
        assert infer_provider("grok-2") == "xai"
        assert infer_provider("grok-4-fast-non-reasoning") == "xai"

    def test_default_openai(self):
        assert infer_provider("some-random-model") == "openai"

    def test_kimi_aliases(self):
        assert infer_provider("moonshotai/kimi-k2") == "kimk2"
        assert infer_provider("kimi-k2") == "kimk2"


class TestModelCatalog:
    def test_all_providers_present(self):
        expected = {"anthropic", "openai", "gemini", "deepseek", "xai", "groq", "kimk2", "local"}
        assert set(MODEL_CATALOG.keys()) >= expected

    def test_entries_are_model_config(self):
        for models in MODEL_CATALOG.values():
            for m in models:
                assert isinstance(m, ModelConfig)
                assert m.name
                assert m.display_name

    def test_is_thinking_field_exists(self):
        """All ModelConfig entries should have the is_thinking field."""
        for models in MODEL_CATALOG.values():
            for m in models:
                assert isinstance(m.is_thinking, bool)

    def test_thinking_models_present(self):
        """Each provider (except local) should have at least one thinking model."""
        for provider, models in MODEL_CATALOG.items():
            if provider == "local":
                continue
            thinking = [m for m in models if m.is_thinking]
            assert len(thinking) >= 1, f"Provider '{provider}' has no thinking model"

    def test_gpt5_in_catalog(self):
        openai_models = MODEL_CATALOG["openai"]
        names = [m.name for m in openai_models]
        assert "gpt-5.2-chat-latest" in names
        assert "gpt-5.2-pro" in names
        assert "gpt-5.2" in names

    def test_grok4_in_catalog(self):
        xai_models = MODEL_CATALOG["xai"]
        names = [m.name for m in xai_models]
        assert "xai/grok-4-fast-non-reasoning" in names
        assert "xai/grok-4-fast-reasoning" in names

    def test_groq_default_family_models_in_catalog(self):
        groq_models = MODEL_CATALOG["groq"]
        names = [m.name for m in groq_models]
        assert "groq/llama-3.3-70b-versatile" in names

    def test_kimi_k2_in_catalog(self):
        kimk2_models = MODEL_CATALOG["kimk2"]
        names = [m.name for m in kimk2_models]
        assert "openrouter/moonshotai/kimi-k2" in names

    def test_deepseek_reasoner_is_thinking(self):
        deepseek_models = MODEL_CATALOG["deepseek"]
        reasoner = next(m for m in deepseek_models if m.is_thinking)
        assert "reasoner" in reasoner.name

    def test_opus_is_thinking(self):
        anthropic_models = MODEL_CATALOG["anthropic"]
        opus = next(m for m in anthropic_models if "opus" in m.name)
        assert opus.is_thinking is True


class TestNewModelConstants:
    def test_google_pro_model(self):
        assert "gemini-3-pro" in GOOGLE_PRO_MODEL

    def test_google_flash_model(self):
        assert "gemini-3-flash" in GOOGLE_FLASH_MODEL

    def test_anthropic_opus_model(self):
        assert "claude-opus-4" in ANTHROPIC_OPUS_MODEL

    def test_gemini_3_in_catalog(self):
        gemini_models = MODEL_CATALOG["gemini"]
        names = [m.name for m in gemini_models]
        assert "gemini/gemini-3-pro-preview" in names
        assert "gemini/gemini-3-flash-preview" in names

    def test_gemini_3_pro_is_heavy(self):
        gemini_models = MODEL_CATALOG["gemini"]
        pro = next(m for m in gemini_models if "3-pro" in m.name)
        assert pro.is_heavy is True

    def test_gemini_3_flash_is_light(self):
        gemini_models = MODEL_CATALOG["gemini"]
        flash = next(m for m in gemini_models if "3-flash" in m.name)
        assert flash.is_heavy is False

    def test_gemini_3_output_limits(self):
        # gemini-3 key in MODEL_OUTPUT_TOKEN_LIMITS should match
        assert get_model_max_tokens("gemini-3-pro-preview") == 1048576
        assert get_model_max_tokens("gemini-3-flash-preview") == 1048576


class TestResolveModelName:
    def test_exact_match_returns_catalog_name(self):
        result = resolve_model_name("anthropic/claude-sonnet-4-6")
        assert result == "anthropic/claude-sonnet-4-6"

    def test_exact_match_haiku(self):
        result = resolve_model_name("anthropic/claude-haiku-4-5-20251001")
        assert result == "anthropic/claude-haiku-4-5-20251001"

    def test_fuzzy_match_claude_haiku_4(self):
        """The production error case: 'claude-haiku-4' should resolve to a valid model."""
        result = resolve_model_name("claude-haiku-4")
        assert result is not None
        assert "claude-haiku" in result
        assert "anthropic/" in result

    def test_fuzzy_match_with_provider_prefix(self):
        result = resolve_model_name("anthropic/claude-haiku-4")
        assert result is not None
        assert "claude-haiku" in result

    def test_fuzzy_match_partial_sonnet(self):
        result = resolve_model_name("claude-sonnet-4")
        assert result is not None
        assert "claude-sonnet" in result

    def test_no_match_returns_none(self):
        result = resolve_model_name("totally-fake-model-xyz-999")
        assert result is None

    def test_empty_string_returns_none(self):
        result = resolve_model_name("")
        assert result is None

    def test_whitespace_only_returns_none(self):
        result = resolve_model_name("   ")
        assert result is None

    def test_gemini_partial_match(self):
        result = resolve_model_name("gemini-3-flash")
        assert result is not None
        assert "gemini" in result

    def test_gpt_partial_match(self):
        result = resolve_model_name("gpt-5.2")
        assert result is not None


class TestProviderEnvMap:
    def test_all_cloud_providers_mapped(self):
        for provider in ("anthropic", "openai", "gemini", "deepseek", "xai", "groq", "kimk2"):
            assert provider in PROVIDER_ENV_MAP
            assert PROVIDER_ENV_MAP[provider].endswith("_API_KEY")

    def test_local_not_mapped(self):
        assert "local" not in PROVIDER_ENV_MAP
