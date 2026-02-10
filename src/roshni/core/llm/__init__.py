"""
LLM client and utilities — powered by LiteLLM.

Requires ``roshni[llm]`` (i.e. ``litellm``).
"""

from .caching import build_cached_system_message, build_system_content_blocks, is_cache_eligible
from .config import (
    MODEL_CATALOG,
    MODEL_OUTPUT_TOKEN_LIMITS,
    THINKING_BUDGET_MAP,
    ModelConfig,
    ThinkingLevel,
    get_default_model,
    get_model_max_tokens,
    infer_provider,
)
from .model_selector import ModelSelector, get_model_selector, reset_model_selector
from .response_continuation import (
    ContinuationConfig,
    ContinuationResult,
    ResponseContinuationMixin,
    build_continuation_prompt,
    is_response_truncated,
    merge_responses,
)
from .token_budget import check_budget, get_usage_summary, record_usage
from .token_management import (
    estimate_token_count,
    format_truncation_warning,
    get_model_context_limit,
    truncate_context,
)
from .utils import extract_text_from_response

__all__ = [
    "MODEL_CATALOG",
    "MODEL_OUTPUT_TOKEN_LIMITS",
    "THINKING_BUDGET_MAP",
    "ContinuationConfig",
    "ContinuationResult",
    "ModelConfig",
    "ModelSelector",
    "ResponseContinuationMixin",
    "ThinkingLevel",
    "build_cached_system_message",
    "build_continuation_prompt",
    "build_system_content_blocks",
    "check_budget",
    "estimate_token_count",
    "extract_text_from_response",
    "format_truncation_warning",
    "get_default_model",
    "get_model_context_limit",
    "get_model_max_tokens",
    "get_model_selector",
    "get_usage_summary",
    "infer_provider",
    "is_cache_eligible",
    "is_response_truncated",
    "merge_responses",
    "record_usage",
    "reset_model_selector",
    "truncate_context",
]
