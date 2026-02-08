"""Tests for core.llm.caching — provider-aware prompt caching helpers."""

from roshni.core.llm.caching import (
    build_cached_system_message,
    build_system_content_blocks,
    is_cache_eligible,
)


class TestIsCacheEligible:
    def test_anthropic_eligible(self):
        assert is_cache_eligible("anthropic") is True

    def test_gemini_eligible(self):
        assert is_cache_eligible("gemini") is True

    def test_openai_not_eligible(self):
        assert is_cache_eligible("openai") is False

    def test_deepseek_not_eligible(self):
        assert is_cache_eligible("deepseek") is False

    def test_case_insensitive(self):
        assert is_cache_eligible("Anthropic") is True
        assert is_cache_eligible("GEMINI") is True


class TestBuildSystemContentBlocks:
    def test_cache_disabled_returns_string(self):
        result = build_system_content_blocks("You are a bot.", enable_cache=False)
        assert isinstance(result, str)
        assert result == "You are a bot."

    def test_cache_disabled_with_dynamic(self):
        result = build_system_content_blocks("Persona", "Memory ctx", enable_cache=False)
        assert isinstance(result, str)
        assert "Persona" in result
        assert "Memory ctx" in result
        assert result == "Persona\n\nMemory ctx"

    def test_cache_enabled_returns_blocks(self):
        result = build_system_content_blocks("Stable persona", enable_cache=True)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Stable persona"
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_cache_enabled_with_dynamic(self):
        result = build_system_content_blocks("Stable", "Dynamic", enable_cache=True)
        assert isinstance(result, list)
        assert len(result) == 2
        # Stable block has cache_control
        assert "cache_control" in result[0]
        # Dynamic block does NOT have cache_control
        assert "cache_control" not in result[1]
        assert result[1]["text"] == "Dynamic"

    def test_cache_with_ttl(self):
        result = build_system_content_blocks("Stable", enable_cache=True, ttl="3600s")
        assert isinstance(result, list)
        assert result[0]["cache_control"]["ttl"] == "3600s"

    def test_cache_without_ttl(self):
        result = build_system_content_blocks("Stable", enable_cache=True)
        assert "ttl" not in result[0]["cache_control"]


class TestBuildCachedSystemMessage:
    def test_eligible_provider_gets_blocks(self):
        msg = build_cached_system_message("Persona", provider="anthropic")
        assert msg["role"] == "system"
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_non_eligible_provider_gets_string(self):
        msg = build_cached_system_message("Persona", provider="openai")
        assert msg["role"] == "system"
        assert isinstance(msg["content"], str)
        assert msg["content"] == "Persona"

    def test_eligible_with_dynamic(self):
        msg = build_cached_system_message("Persona", "Memory", provider="gemini")
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 2

    def test_non_eligible_with_dynamic(self):
        msg = build_cached_system_message("Persona", "Memory", provider="openai")
        assert isinstance(msg["content"], str)
        assert "Persona" in msg["content"]
        assert "Memory" in msg["content"]

    def test_gemini_with_ttl(self):
        msg = build_cached_system_message("Persona", provider="gemini", ttl="3600s")
        assert msg["content"][0]["cache_control"]["ttl"] == "3600s"

    def test_ttl_ignored_for_non_eligible(self):
        msg = build_cached_system_message("Persona", provider="openai", ttl="3600s")
        # Non-eligible returns string, no cache_control at all
        assert isinstance(msg["content"], str)
