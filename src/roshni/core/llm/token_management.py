"""Token counting and context truncation utilities for managing LLM context windows.

This module provides tools for:
- Estimating token counts (heuristic-based, no tokenizer dependency)
- Getting model context/input limits
- Truncating context to fit within model limits
- Calculating appropriate response reserves

Ported from weeklies' core/llm/token_management.py — roshni is now the
canonical source for these utilities.
"""

import logging
from typing import Any

from .config import MODEL_OUTPUT_TOKEN_LIMITS, PROVIDER_DEFAULT_LIMITS

logger = logging.getLogger(__name__)

# Input/Context token limits for different models
# Note: These are INPUT limits, different from OUTPUT limits in config.py
MODEL_CONTEXT_LIMITS = {
    # Google Gemini
    "gemini-3-pro": 1048576,
    "gemini-3-flash": 1048576,
    "gemini-2.5-pro": 1048576,
    "gemini-2.5-flash": 1048576,
    # OpenAI
    "o1": 200000,
    "o3-max": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4o": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-3.5-turbo": 16384,
    # Anthropic
    "claude-4-opus": 200000,
    "claude-4-sonnet": 200000,
    "claude-3-haiku": 200000,
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-2.1": 200000,
    "claude-2": 100000,
}

# Reserve tokens for response (conservative default)
RESPONSE_TOKEN_RESERVE = 4000


def estimate_token_count(text: str) -> int:
    """
    Estimate token count using a simple heuristic.

    Uses the formula: roughly 1 token = 4 characters or 0.75 words.
    Returns the more conservative (higher) estimate.

    This is intentionally tokenizer-free to avoid heavy dependencies.
    For precise counts, use the actual tokenizer for your model.

    Args:
        text: The text to estimate tokens for

    Returns:
        Estimated token count
    """
    if not text:
        return 0

    # Count words and characters
    words = len(text.split())
    chars = len(text)

    # Use the more conservative estimate
    word_estimate = int(words / 0.75)
    char_estimate = int(chars / 4)

    return max(word_estimate, char_estimate)


def get_model_context_limit(model_name: str, provider: str | None = None) -> int:
    """
    Get the context/input token limit for a given model.

    Args:
        model_name: Name of the model (e.g., "gpt-4o", "claude-3-sonnet")
        provider: Optional provider hint (gemini, openai, anthropic)

    Returns:
        Maximum input token limit for the model
    """
    # Try exact match first
    if model_name in MODEL_CONTEXT_LIMITS:
        return MODEL_CONTEXT_LIMITS[model_name]

    # Try partial match
    model_lower = model_name.lower()
    for key, limit in MODEL_CONTEXT_LIMITS.items():
        if key.lower() in model_lower or model_lower in key.lower():
            return limit

    # Default limits by provider
    provider_defaults = {
        "gemini": 1048576,  # Most Gemini models support 1M
        "openai": 128000,  # Most modern OpenAI models support 128k
        "anthropic": 200000,  # Most Claude models support 200k
    }

    if provider and provider.lower() in provider_defaults:
        return provider_defaults[provider.lower()]

    # Conservative default
    return 8192


# Backward compatibility alias
get_model_token_limit = get_model_context_limit


def get_response_token_reserve(model_name: str, provider: str | None = None) -> int:
    """
    Get appropriate response token reserve based on model.

    The reserve is capped at RESPONSE_TOKEN_RESERVE (4000) for smaller models,
    but uses the full output limit for models that support more.

    Args:
        model_name: Name of the model
        provider: Optional provider hint

    Returns:
        Number of tokens to reserve for the response
    """
    output_limit = None

    # Check exact match first
    if model_name in MODEL_OUTPUT_TOKEN_LIMITS:
        output_limit = MODEL_OUTPUT_TOKEN_LIMITS[model_name]
    else:
        # Try partial matching
        for model_key, limit in MODEL_OUTPUT_TOKEN_LIMITS.items():
            if model_key in model_name:
                output_limit = limit
                break

    # Fall back to provider default if needed
    if output_limit is None and provider:
        output_limit = PROVIDER_DEFAULT_LIMITS.get(provider, 4096)

    if output_limit is None:
        output_limit = 4096

    # Use output limit as response reserve, capped at 4000 for smaller models
    return min(output_limit, RESPONSE_TOKEN_RESERVE)


def truncate_context(
    context: str,
    query: str,
    model_name: str,
    provider: str | None = None,
    keep_ratio: float = 0.8,
) -> tuple[str, bool, dict[str, Any]]:
    """
    Truncate context to fit within model token limits.

    Attempts to keep whole documents intact by splitting on document
    separators ("\\n\\n---\\n\\n") and removing documents from the end
    until the context fits.

    Args:
        context: The context string to potentially truncate
        query: The query string (counts against token limit)
        model_name: Name of the model being used
        provider: Provider name (gemini, openai, anthropic)
        keep_ratio: Ratio of token limit to actually use (safety margin)

    Returns:
        Tuple of (truncated_context, was_truncated, info_dict)
    """
    # Get token limit
    token_limit = get_model_context_limit(model_name, provider)

    # Get appropriate response reserve for this model
    response_reserve = get_response_token_reserve(model_name, provider)

    # Calculate available tokens (with safety margin and response reserve)
    available_tokens = int(token_limit * keep_ratio) - response_reserve

    # Estimate query tokens
    query_tokens = estimate_token_count(query)

    # Calculate tokens available for context
    context_tokens_available = available_tokens - query_tokens

    # Estimate context tokens
    context_tokens = estimate_token_count(context)

    info: dict[str, Any] = {
        "model_token_limit": token_limit,
        "available_tokens": available_tokens,
        "query_tokens": query_tokens,
        "context_tokens": context_tokens,
        "context_tokens_available": context_tokens_available,
        "was_truncated": False,
        "truncation_ratio": 1.0,
    }

    # Check if truncation is needed
    if context_tokens <= context_tokens_available:
        return context, False, info

    # Need to truncate
    logger.warning(
        f"Context exceeds token limit: {context_tokens} > {context_tokens_available}. "
        f"Truncating to fit {model_name} limit of {token_limit} tokens."
    )

    # Calculate how much to keep
    keep_pct = context_tokens_available / context_tokens

    # Split context by document separator
    doc_separator = "\n\n---\n\n"
    documents = context.split(doc_separator)

    # Try to keep whole documents
    kept_docs = []
    current_tokens = 0

    for doc in documents:
        doc_tokens = estimate_token_count(doc)
        if current_tokens + doc_tokens <= context_tokens_available:
            kept_docs.append(doc)
            current_tokens += doc_tokens
        else:
            # Stop adding documents
            break

    # If we kept at least some documents, use them
    if kept_docs:
        truncated_context = doc_separator.join(kept_docs)

        # Add truncation notice
        removed_count = len(documents) - len(kept_docs)
        if removed_count > 0:
            truncated_context += f"\n\n[NOTE: {removed_count} documents omitted due to token limits]"
    else:
        # If even one document is too large, truncate the first document
        char_limit = int(len(documents[0]) * keep_pct)
        truncated_context = documents[0][:char_limit] + "\n\n[TRUNCATED due to token limits]"

    info["was_truncated"] = True
    info["truncation_ratio"] = keep_pct
    info["original_doc_count"] = len(documents)
    info["kept_doc_count"] = len(kept_docs) if kept_docs else 0

    return truncated_context, True, info


def format_truncation_warning(info: dict[str, Any]) -> str:
    """
    Format a user-friendly truncation warning.

    Args:
        info: The info dict returned by truncate_context

    Returns:
        Warning message string, or empty string if no truncation occurred
    """
    if not info.get("was_truncated", False):
        return ""

    return (
        f"Context was truncated to fit model limits: "
        f"{info.get('kept_doc_count', 0)}/{info.get('original_doc_count', 0)} documents kept, "
        f"using {info.get('context_tokens_available', 0):,} tokens"
    )
