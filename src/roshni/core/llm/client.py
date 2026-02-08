"""
LLM Client — unified interface for LLM interactions via LiteLLM.

Provides a simple chat interface across multiple providers (OpenAI,
Anthropic, Gemini, Ollama) using litellm as the single routing layer.
"""

from time import time
from typing import Any

from loguru import logger

from .config import get_default_model, get_model_max_tokens, infer_provider
from .token_budget import check_budget, record_usage
from .utils import extract_text_from_response


class LLMClient:
    """
    Multi-provider LLM client backed by LiteLLM.

    Model names follow litellm conventions:
      - OpenAI:    ``"gpt-4o"``, ``"o3"``
      - Anthropic: ``"anthropic/claude-sonnet-4-20250514"``
      - Gemini:    ``"gemini/gemini-2.5-flash"``
      - Local:     ``"ollama/deepseek-r1"``
    """

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        system_prompt: str | None = None,
        system_prompt_path: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        max_history_messages: int | None = 10,
        timeout: int = 180,
        num_retries: int = 2,
    ):
        # Resolve model ─ accept either (model=) or (provider=) style
        if model:
            self.model = model
            self.provider = provider or infer_provider(model)
        elif provider:
            self.provider = provider
            self.model = get_default_model(provider)
        else:
            self.provider = "openai"
            self.model = get_default_model("openai")

        self.temperature = temperature
        self.timeout = timeout
        self.num_retries = num_retries
        self.max_history_messages = max_history_messages
        self.message_history: list[dict[str, str]] = []

        # Resolve max_tokens
        model_limit = get_model_max_tokens(self.model, self.provider)
        if max_tokens is not None and max_tokens > model_limit:
            logger.warning(f"max_tokens ({max_tokens}) exceeds model limit ({model_limit}). Capping.")
            self.max_tokens = model_limit
        else:
            self.max_tokens = max_tokens or model_limit

        # System prompt
        if system_prompt is not None:
            self.system_prompt = system_prompt
        elif system_prompt_path is not None:
            self.system_prompt = self._load_prompt_file(system_prompt_path)
        else:
            self.system_prompt = "You are a helpful AI assistant."

        logger.debug(f"LLMClient: model={self.model}  max_tokens={self.max_tokens}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, message: str, stop: list[str] | None = None, **kwargs: Any) -> tuple[str, float]:
        """Send a message and return (response_text, elapsed_seconds).

        Respects token budget, maintains conversation history, and
        records token usage.
        """
        try:
            import litellm
        except ImportError:
            raise ImportError("Install LLM support with: pip install roshni[llm]")

        start = time()

        if not message:
            return "Input message is empty.", 0.0

        within_budget, remaining = check_budget()
        if not within_budget:
            logger.warning(f"Daily token budget exceeded ({remaining} remaining)")
            return "Daily token budget exceeded. Try again tomorrow.", 0.0

        # Build messages list
        messages = self._build_messages(message, **kwargs)

        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop,
                timeout=self.timeout,
                num_retries=self.num_retries,
            )

            # Extract text
            content = response.choices[0].message.content or ""
            text = extract_text_from_response(content)

            # Record token usage
            usage = getattr(response, "usage", None)
            if usage:
                record_usage(
                    input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    provider=self.provider,
                    model=self.model,
                )

            # Update history
            self.message_history.append({"role": "user", "content": message})
            if text.strip():
                self.message_history.append({"role": "assistant", "content": text})

            return text, time() - start

        except Exception as e:
            logger.error(f"LLM error ({type(e).__name__}): {e}")
            return f"An error occurred while processing your request: {e!s}", time() - start

    async def achat(self, message: str, stop: list[str] | None = None, **kwargs: Any) -> tuple[str, float]:
        """Async version of chat()."""
        try:
            import litellm
        except ImportError:
            raise ImportError("Install LLM support with: pip install roshni[llm]")

        start = time()

        if not message:
            return "Input message is empty.", 0.0

        within_budget, _remaining = check_budget()
        if not within_budget:
            return "Daily token budget exceeded. Try again tomorrow.", 0.0

        messages = self._build_messages(message, **kwargs)

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop,
                timeout=self.timeout,
                num_retries=self.num_retries,
            )

            content = response.choices[0].message.content or ""
            text = extract_text_from_response(content)

            usage = getattr(response, "usage", None)
            if usage:
                record_usage(
                    input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    provider=self.provider,
                    model=self.model,
                )

            self.message_history.append({"role": "user", "content": message})
            if text.strip():
                self.message_history.append({"role": "assistant", "content": text})

            return text, time() - start

        except Exception as e:
            logger.error(f"LLM error ({type(e).__name__}): {e}")
            return f"An error occurred while processing your request: {e!s}", time() - start

    def format_system_prompt(self, **kwargs: Any) -> str:
        """Format the system prompt template with context variables."""
        return self.system_prompt.format(**kwargs)

    def clear_history(self) -> None:
        self.message_history = []

    def get_config_info(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "max_history_messages": self.max_history_messages,
            "history_length": len(self.message_history),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_messages(self, message: str, **kwargs: Any) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []

        # System prompt
        if kwargs:
            prompt = self.format_system_prompt(**kwargs)
        else:
            prompt = self.system_prompt
        if prompt and prompt.strip():
            messages.append({"role": "system", "content": prompt})

        # History
        history = list(self.message_history)
        if self.max_history_messages is not None:
            history = history[-self.max_history_messages :]
        messages.extend(history)

        messages.append({"role": "user", "content": message})
        return messages

    @staticmethod
    def _load_prompt_file(path: str) -> str:
        try:
            with open(path) as f:
                text = f.read().strip()
            return text or "You are a helpful AI assistant."
        except FileNotFoundError:
            logger.error(f"System prompt file not found: {path}")
            return "You are a helpful AI assistant."
