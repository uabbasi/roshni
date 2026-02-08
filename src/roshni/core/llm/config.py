"""
LLM Configuration — model constants and token limits.

Central configuration for all LLM interactions.  Change defaults here
to affect every module that uses roshni's LLM client.
"""

from dataclasses import dataclass

# --- Default model names per provider ---

LOCAL_MODEL = "deepseek-r1"
GOOGLE_MODEL = "gemini/gemini-2.5-flash"
OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "anthropic/claude-sonnet-4-20250514"
DEEPSEEK_MODEL = "deepseek/deepseek-chat"
XAI_MODEL = "xai/grok-2"
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
    "grok-3": 16_384,
    "grok-2": 8_192,
    # Groq
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


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    name: str
    display_name: str
    provider: str = ""
    is_heavy: bool = False
    max_tokens: int | None = None
    cost_tier: str = "medium"  # low, medium, high, free


# Model catalog with light and heavy options per provider.
MODEL_CATALOG: dict[str, list[ModelConfig]] = {
    "anthropic": [
        ModelConfig("anthropic/claude-haiku-4", "Claude Haiku 4", "anthropic", False, 8192, "low"),
        ModelConfig("anthropic/claude-sonnet-4-20250514", "Claude Sonnet 4", "anthropic", False, 8192, "medium"),
        ModelConfig("anthropic/claude-opus-4-20250514", "Claude Opus 4", "anthropic", True, 8192, "high"),
    ],
    "openai": [
        ModelConfig("gpt-4o", "GPT-4o", "openai", False, 16384, "medium"),
        ModelConfig("o3", "O3", "openai", True, 16384, "high"),
    ],
    "gemini": [
        ModelConfig("gemini/gemini-2.0-flash", "Gemini 2.0 Flash", "gemini", False, 8192, "low"),
        ModelConfig("gemini/gemini-2.5-flash", "Gemini 2.5 Flash", "gemini", False, 64000, "low"),
        ModelConfig("gemini/gemini-2.5-pro", "Gemini 2.5 Pro", "gemini", True, 64000, "medium"),
    ],
    "deepseek": [
        ModelConfig("deepseek/deepseek-chat", "DeepSeek Chat", "deepseek", False, 8192, "low"),
        ModelConfig("deepseek/deepseek-reasoner", "DeepSeek Reasoner", "deepseek", True, 8192, "low"),
    ],
    "xai": [
        ModelConfig("xai/grok-2", "Grok 2", "xai", False, 8192, "medium"),
        ModelConfig("xai/grok-3", "Grok 3", "xai", True, 16384, "high"),
    ],
    "groq": [
        ModelConfig("groq/llama-3.3-70b-versatile", "Llama 3.3 70B (Groq)", "groq", False, 8192, "low"),
        ModelConfig("groq/llama-3.1-8b-instant", "Llama 3.1 8B (Groq)", "groq", False, 8192, "free"),
        ModelConfig("groq/deepseek-r1-distill-llama-70b", "DeepSeek R1 Distill (Groq)", "groq", True, 16384, "low"),
    ],
    "local": [
        ModelConfig("ollama/deepseek-r1", "DeepSeek R1 (Local)", "local", False, 8192, "free"),
    ],
}


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
    if any(k in model_name for k in ("gpt-", "o1", "o3", "o4")):
        return "openai"
    if "claude" in model_name:
        return "anthropic"
    if "gemini" in model_name:
        return "gemini"
    if "grok" in model_name:
        return "xai"
    return "openai"
