"""
LLM client and utilities — powered by LiteLLM.

Requires ``roshni[llm]`` (i.e. ``litellm``).
"""

from .config import (
    MODEL_CATALOG,
    MODEL_OUTPUT_TOKEN_LIMITS,
    ModelConfig,
    get_default_model,
    get_model_max_tokens,
    infer_provider,
)
from .model_selector import ModelSelector, get_model_selector, reset_model_selector
from .token_budget import check_budget, get_usage_summary, record_usage
from .utils import extract_text_from_response

__all__ = [
    "MODEL_CATALOG",
    "MODEL_OUTPUT_TOKEN_LIMITS",
    "ModelConfig",
    "ModelSelector",
    "check_budget",
    "extract_text_from_response",
    "get_default_model",
    "get_model_max_tokens",
    "get_model_selector",
    "get_usage_summary",
    "infer_provider",
    "record_usage",
    "reset_model_selector",
]
