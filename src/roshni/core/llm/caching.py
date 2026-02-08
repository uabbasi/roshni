"""
Provider-aware prompt caching helpers.

LiteLLM >=1.60 provides a unified ``cache_control`` interface that translates
to the correct format per provider:
  - **Anthropic**: native ``cache_control`` blocks (``ephemeral``)
  - **Gemini 2.5**: ``/cachedContents`` API (90% discount with explicit markers)
  - **OpenAI/DeepSeek**: automatic prefix caching (no markers needed)

These helpers build system messages with ``cache_control`` annotations on
the *stable* prefix (persona, identity) while leaving dynamic parts
(memory, mode hints) uncached.  For non-eligible providers, plain strings
are returned — no wasted bytes, no log noise.
"""

from __future__ import annotations

from typing import Any

# Providers where explicit cache_control markers yield a discount.
# OpenAI/DeepSeek do automatic prefix caching — no markers needed.
_CACHE_ELIGIBLE_PROVIDERS = frozenset({"anthropic", "gemini"})


def is_cache_eligible(provider: str) -> bool:
    """Whether the provider supports explicit ``cache_control`` markers."""
    return provider.lower() in _CACHE_ELIGIBLE_PROVIDERS


def build_system_content_blocks(
    stable_text: str,
    dynamic_text: str | None = None,
    *,
    enable_cache: bool = False,
    ttl: str | None = None,
) -> str | list[dict[str, Any]]:
    """Build system message content with optional cache_control on the stable prefix.

    Returns a plain string when caching is disabled (backward compatible),
    or a content-block array when enabled.

    Args:
        stable_text: Persona / identity text that rarely changes.
        dynamic_text: Memory context, mode hints — changes per call.
        enable_cache: Whether to add cache_control annotations.
        ttl: Cache TTL string (e.g. ``"3600s"`` for Gemini).
            Defaults to ``None`` (provider default: 5 min Anthropic, 1 hr Gemini).
    """
    if not enable_cache:
        parts = [stable_text]
        if dynamic_text:
            parts.append(dynamic_text)
        return "\n\n".join(parts)

    # Content-block format with cache_control on the stable portion
    cache_control: dict[str, Any] = {"type": "ephemeral"}
    if ttl is not None:
        cache_control["ttl"] = ttl

    blocks: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": stable_text,
            "cache_control": cache_control,
        },
    ]

    if dynamic_text:
        blocks.append({"type": "text", "text": dynamic_text})

    return blocks


def build_cached_system_message(
    stable_text: str,
    dynamic_text: str | None = None,
    *,
    provider: str,
    ttl: str | None = None,
) -> dict[str, Any]:
    """Build a system message dict, auto-detecting cache eligibility.

    For cache-eligible providers, returns content as a block array with
    ``cache_control``.  For others, returns a plain string.

    Args:
        stable_text: Persona / identity text that rarely changes.
        dynamic_text: Memory context, mode hints — changes per call.
        provider: LLM provider name (e.g. ``"anthropic"``, ``"gemini"``).
        ttl: Optional cache TTL for Gemini (e.g. ``"3600s"``).
    """
    eligible = is_cache_eligible(provider)
    content = build_system_content_blocks(
        stable_text,
        dynamic_text,
        enable_cache=eligible,
        ttl=ttl if eligible else None,
    )
    return {"role": "system", "content": content}
