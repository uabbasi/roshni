"""Tests for core.llm.token_management — token estimation, context limits, truncation."""

from roshni.core.llm.token_management import (
    MODEL_CONTEXT_LIMITS,
    RESPONSE_TOKEN_RESERVE,
    estimate_token_count,
    format_truncation_warning,
    get_model_context_limit,
    get_model_token_limit,
    get_response_token_reserve,
    truncate_context,
)


class TestEstimateTokenCount:
    def test_empty_string(self):
        assert estimate_token_count("") == 0

    def test_short_text(self):
        count = estimate_token_count("hello world")
        assert count > 0

    def test_returns_conservative_estimate(self):
        # "hello world" → 2 words, 11 chars
        # word estimate: int(2 / 0.75) = 2
        # char estimate: int(11 / 4) = 2
        count = estimate_token_count("hello world")
        assert count >= 2

    def test_longer_text(self):
        text = "The quick brown fox jumps over the lazy dog. " * 10
        count = estimate_token_count(text)
        # Should be roughly proportional to length
        assert count > 50

    def test_monotonically_increasing(self):
        short = estimate_token_count("hello")
        medium = estimate_token_count("hello world, this is a test")
        long = estimate_token_count("hello world, this is a test " * 20)
        assert short <= medium <= long


class TestGetModelContextLimit:
    def test_exact_match(self):
        assert get_model_context_limit("gpt-4o") == 128000

    def test_partial_match(self):
        # "gpt-4o" key should match "gpt-4o-mini" via partial matching
        assert get_model_context_limit("gpt-4o-mini") == 128000

    def test_gemini_models(self):
        assert get_model_context_limit("gemini-2.5-pro") == 1048576
        assert get_model_context_limit("gemini-3-pro") == 1048576

    def test_claude_models(self):
        assert get_model_context_limit("claude-4-opus") == 200000
        assert get_model_context_limit("claude-3-opus") == 200000

    def test_provider_fallback(self):
        assert get_model_context_limit("some-unknown-gemini", provider="gemini") == 1048576
        assert get_model_context_limit("some-unknown-openai", provider="openai") == 128000
        assert get_model_context_limit("some-unknown-anthropic", provider="anthropic") == 200000

    def test_conservative_default(self):
        assert get_model_context_limit("totally-unknown-model") == 8192

    def test_backward_compat_alias(self):
        # get_model_token_limit should be the same function
        assert get_model_token_limit("gpt-4o") == get_model_context_limit("gpt-4o")


class TestGetResponseTokenReserve:
    def test_small_model_capped(self):
        # Models with output limit <= RESPONSE_TOKEN_RESERVE get their full output limit
        reserve = get_response_token_reserve("gpt-4")
        assert reserve <= RESPONSE_TOKEN_RESERVE

    def test_large_model_capped_at_reserve(self):
        # Even large output models are capped at RESPONSE_TOKEN_RESERVE
        reserve = get_response_token_reserve("gemini-2.5-pro")
        assert reserve == RESPONSE_TOKEN_RESERVE

    def test_provider_fallback(self):
        reserve = get_response_token_reserve("unknown-model", provider="anthropic")
        assert reserve <= RESPONSE_TOKEN_RESERVE

    def test_no_info_fallback(self):
        reserve = get_response_token_reserve("totally-unknown")
        assert reserve <= RESPONSE_TOKEN_RESERVE


class TestTruncateContext:
    def test_no_truncation_needed(self):
        context = "Short context."
        query = "Short query?"
        result, was_truncated, info = truncate_context(context, query, "gemini-2.5-pro")
        assert not was_truncated
        assert result == context
        assert info["was_truncated"] is False

    def test_truncation_on_small_model(self):
        # gpt-4 has 8192 context limit — create context that exceeds it
        context = "word " * 10000  # ~10000 words ≈ 13333 tokens
        query = "What is this about?"
        result, was_truncated, info = truncate_context(context, query, "gpt-4")
        assert was_truncated
        assert info["was_truncated"] is True
        assert len(result) < len(context)

    def test_document_separator_preservation(self):
        # Create context with document separators
        docs = [f"Document {i} content. " * 50 for i in range(20)]
        context = "\n\n---\n\n".join(docs)
        query = "Summarize?"
        result, was_truncated, info = truncate_context(context, query, "gpt-4")
        if was_truncated:
            assert info["kept_doc_count"] < info["original_doc_count"]
            assert "[NOTE:" in result or "[TRUNCATED" in result

    def test_info_dict_keys(self):
        context = "Some context."
        query = "Query?"
        _, _, info = truncate_context(context, query, "gpt-4o")
        assert "model_token_limit" in info
        assert "available_tokens" in info
        assert "query_tokens" in info
        assert "context_tokens" in info
        assert "was_truncated" in info


class TestFormatTruncationWarning:
    def test_no_truncation(self):
        info = {"was_truncated": False}
        assert format_truncation_warning(info) == ""

    def test_with_truncation(self):
        info = {
            "was_truncated": True,
            "kept_doc_count": 3,
            "original_doc_count": 10,
            "context_tokens_available": 5000,
        }
        warning = format_truncation_warning(info)
        assert "3/10" in warning
        assert "5,000" in warning

    def test_empty_info(self):
        assert format_truncation_warning({}) == ""


class TestModelContextLimitsDict:
    def test_gemini_3_entries_present(self):
        assert "gemini-3-pro" in MODEL_CONTEXT_LIMITS
        assert "gemini-3-flash" in MODEL_CONTEXT_LIMITS
        assert MODEL_CONTEXT_LIMITS["gemini-3-pro"] == 1048576
        assert MODEL_CONTEXT_LIMITS["gemini-3-flash"] == 1048576
