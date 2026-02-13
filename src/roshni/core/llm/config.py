"""
LLM Configuration — model constants and token limits.

Central configuration for all LLM interactions.  Change defaults here
to affect every module that uses roshni's LLM client.
"""

from dataclasses import dataclass
from enum import IntEnum

# --- Default model names per provider ---

LOCAL_MODEL = "ollama/deepseek-r1"
GOOGLE_MODEL = "gemini/gemini-2.5-flash"
GOOGLE_PRO_MODEL = "gemini/gemini-3-pro-preview"
GOOGLE_FLASH_MODEL = "gemini/gemini-3-flash-preview"
OPENAI_MODEL = "gpt-5.2-chat-latest"
ANTHROPIC_MODEL = "anthropic/claude-sonnet-4-20250514"
ANTHROPIC_OPUS_MODEL = "anthropic/claude-opus-4-20250514"
DEEPSEEK_MODEL = "deepseek/deepseek-chat"
XAI_MODEL = "xai/grok-4-fast-non-reasoning"
GROQ_MODEL = "groq/llama-3.3-70b-versatile"

# --- Output token limits ---
# These are OUTPUT token limits (how many tokens the model can generate).
# Different from context/input limits.
# Keys are matched with partial string matching, so "gpt-4o" matches "gpt-4o-mini".

MODEL_OUTPUT_TOKEN_LIMITS: dict[str, int] = {
    # OpenAI
    "o4-mini": 16_384,
    "o3": 16_384,
    "o1": 16_384,
    "gpt-4o": 16_384,
    "gpt-4": 4_096,
    "gpt-5": 16_384,
    # Anthropic
    "claude-opus-4": 8_192,
    "claude-sonnet-4": 8_192,
    "claude-haiku-4": 8_192,
    "claude-sonnet": 4_096,
    "claude-haiku": 4_096,
    # Google Gemini
    "gemini-3": 1_048_576,
    "gemini-2.5": 1_048_576,
    "gemini-2.0": 8_192,
    "gemini-1.5": 8_192,
    "gemini-pro": 1_048_576,
    "gemini-flash": 1_048_576,
    # DeepSeek
    "deepseek-chat": 8_192,
    "deepseek-reasoner": 8_192,
    # xAI (Grok)
    "grok-4": 16_384,
    "grok-3": 16_384,
    "grok-2": 8_192,
    # Groq
    "llama-4-maverick": 8_192,
    "llama-3.3-70b": 8_192,
    "llama-3.1-8b": 8_192,
    "deepseek-r1-distill": 16_384,
    # Local
    "deepseek": 8_192,
}

PROVIDER_ENV_MAP: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
    "groq": "GROQ_API_KEY",
}

PROVIDER_DEFAULT_LIMITS: dict[str, int] = {
    "openai": 4_096,
    "anthropic": 4_096,
    "gemini": 1_048_576,
    "deepseek": 8_192,
    "xai": 8_192,
    "groq": 8_192,
    "local": 8_192,
}


class ThinkingLevel(IntEnum):
    """Gradation for extended thinking / reasoning budgets."""

    OFF = 0
    LOW = 1  # 1024 tokens
    MEDIUM = 2  # 4096 tokens
    HIGH = 3  # 16384 tokens


THINKING_BUDGET_MAP: dict[ThinkingLevel, int] = {
    ThinkingLevel.OFF: 0,
    ThinkingLevel.LOW: 1024,
    ThinkingLevel.MEDIUM: 4096,
    ThinkingLevel.HIGH: 16384,
}


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    name: str
    display_name: str
    provider: str = ""
    is_heavy: bool = False
    is_thinking: bool = False
    max_tokens: int | None = None
    cost_tier: str = "medium"  # low, medium, high, free
    thinking_budget_tokens: int | None = None


# Model catalog with light, heavy, and thinking options per provider.
MODEL_CATALOG: dict[str, list[ModelConfig]] = {
    "anthropic": [
        ModelConfig("anthropic/claude-haiku-4", "Claude Haiku 4", "anthropic", False, False, 8192, "low"),
        ModelConfig("anthropic/claude-sonnet-4-20250514", "Claude Sonnet 4", "anthropic", True, False, 8192, "medium"),
        ModelConfig("anthropic/claude-opus-4-20250514", "Claude Opus 4", "anthropic", True, True, 8192, "high"),
    ],
    "openai": [
        ModelConfig("gpt-5.2-chat-latest", "GPT-5.2 Chat", "openai", False, False, 16384, "medium"),
        ModelConfig("gpt-5.2-pro", "GPT-5.2 Pro", "openai", True, False, 16384, "high"),
        ModelConfig("gpt-5.2", "GPT-5.2", "openai", True, True, 16384, "high"),
    ],
    "gemini": [
        ModelConfig("gemini/gemini-3-flash-preview", "Gemini 3 Flash", "gemini", False, False, 1048576, "low"),
        ModelConfig("gemini/gemini-3-pro-preview", "Gemini 3 Pro", "gemini", True, False, 1048576, "medium"),
        ModelConfig("gemini/gemini-2.5-pro", "Gemini 2.5 Pro (Thinking)", "gemini", True, True, 64000, "medium"),
    ],
    "deepseek": [
        ModelConfig("deepseek/deepseek-chat", "DeepSeek Chat", "deepseek", False, False, 8192, "low"),
        ModelConfig("deepseek/deepseek-chat", "DeepSeek Chat (Heavy)", "deepseek", True, False, 8192, "low"),
        ModelConfig("deepseek/deepseek-reasoner", "DeepSeek Reasoner", "deepseek", True, True, 8192, "low"),
    ],
    "xai": [
        ModelConfig("xai/grok-4-fast-non-reasoning", "Grok 4 Fast", "xai", False, False, 16384, "medium"),
        ModelConfig("xai/grok-4-fast-reasoning", "Grok 4 Fast Reasoning", "xai", True, False, 16384, "high"),
        ModelConfig("xai/grok-4-fast-reasoning", "Grok 4 Fast Reasoning (Thinking)", "xai", True, True, 16384, "high"),
    ],
    "groq": [
        ModelConfig(
            "groq/llama-4-maverick-17b-128e-instruct",
            "Llama 4 Maverick (Groq)",
            "groq",
            False,
            False,
            8192,
            "low",
        ),
        ModelConfig("groq/llama-3.3-70b-versatile", "Llama 3.3 70B (Groq)", "groq", True, False, 8192, "low"),
        ModelConfig(
            "groq/deepseek-r1-distill-llama-70b", "DeepSeek R1 Distill (Groq)", "groq", True, True, 16384, "low"
        ),
    ],
    "local": [
        ModelConfig("ollama/deepseek-r1", "DeepSeek R1 (Local)", "local", False, False, 8192, "free"),
    ],
}


FAMILY_ALIASES: dict[str, str] = {
    "claude": "anthropic",
    "gpt": "openai",
    "llama": "groq",
    "grok": "xai",
    "anthropic": "anthropic",
    "openai": "openai",
    "gemini": "gemini",
    "deepseek": "deepseek",
    "xai": "xai",
    "groq": "groq",
    "local": "local",
}


def resolve_family(name: str) -> str | None:
    """Resolve a user-provided name to a MODEL_CATALOG key."""
    return FAMILY_ALIASES.get(name.strip().lower())


def get_family_models(provider_key: str) -> tuple[ModelConfig, ModelConfig, ModelConfig] | None:
    """Get (light, heavy, thinking) for a provider. None if < 3 models."""
    models = MODEL_CATALOG.get(provider_key)
    if not models or len(models) < 3:
        return None
    return (models[0], models[1], models[2])


def get_available_families() -> list[str]:
    """Provider keys with full light/heavy/thinking families."""
    return [k for k, v in MODEL_CATALOG.items() if len(v) >= 3]


def get_default_model(provider: str) -> str:
    """Get the default litellm model string for a provider."""
    model_map = {
        "anthropic": ANTHROPIC_MODEL,
        "openai": OPENAI_MODEL,
        "gemini": GOOGLE_MODEL,
        "deepseek": DEEPSEEK_MODEL,
        "xai": XAI_MODEL,
        "groq": GROQ_MODEL,
        "local": LOCAL_MODEL,
    }
    return model_map.get(provider, OPENAI_MODEL)


def get_model_max_tokens(model_name: str, provider: str | None = None) -> int:
    """Get the max output tokens for a model using partial string matching."""
    if model_name in MODEL_OUTPUT_TOKEN_LIMITS:
        return MODEL_OUTPUT_TOKEN_LIMITS[model_name]

    for model_key, limit in MODEL_OUTPUT_TOKEN_LIMITS.items():
        if model_key in model_name:
            return limit

    if provider and provider in PROVIDER_DEFAULT_LIMITS:
        return PROVIDER_DEFAULT_LIMITS[provider]

    return 4096


def infer_provider(model_name: str) -> str:
    """Infer provider from a litellm model string."""
    # Prefix-based (most reliable)
    if model_name.startswith("anthropic/"):
        return "anthropic"
    if model_name.startswith("gemini/"):
        return "gemini"
    if model_name.startswith("deepseek/"):
        return "deepseek"
    if model_name.startswith("xai/"):
        return "xai"
    if model_name.startswith("groq/"):
        return "groq"
    if model_name.startswith("ollama/"):
        return "local"
    # Substring-based fallbacks
    if any(k in model_name for k in ("gpt-", "gpt-5", "o1", "o3", "o4")):
        return "openai"
    if "claude" in model_name:
        return "anthropic"
    if "gemini" in model_name:
        return "gemini"
    if "grok" in model_name:
        return "xai"
    return "openai"
