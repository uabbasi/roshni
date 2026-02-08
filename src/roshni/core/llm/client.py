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
        fallback_model: str | None = None,
        fallback_provider: str | None = None,
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

        # Fallback configuration
        self.fallback_model = fallback_model
        self.fallback_provider = fallback_provider or (infer_provider(fallback_model) if fallback_model else None)

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
        if self.fallback_model:
            logger.debug(f"LLMClient: fallback={self.fallback_model}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        stop: list[str] | None = None,
    ) -> Any:
        """Low-level completion call with budget checking, usage recording, and fallback.

        Checks token budget, calls litellm.completion(), and records
        token usage.  On retryable errors (rate limit, API error, connection),
        falls back to the fallback model if configured.

        Does NOT manage conversation history — the caller is responsible for that.

        Args:
            messages: Full messages list (system + history + user).
            tools: Optional tool schemas in OpenAI function-calling format.
            stop: Optional stop sequences.

        Returns:
            The raw litellm response object.

        Raises:
            ImportError: If litellm is not installed.
            RuntimeError: If the daily token budget is exceeded.
        """
        try:
            import litellm
        except ImportError:
            raise ImportError("Install LLM support with: pip install roshni[llm]")

        self._check_budget()
        kwargs = self._build_completion_kwargs(messages, tools=tools, stop=stop)

        try:
            response = litellm.completion(**kwargs)
            self._record_response_usage(response)
            return response
        except (litellm.RateLimitError, litellm.APIError, litellm.APIConnectionError) as e:
            if self.fallback_model:
                return self._fallback_completion(kwargs, e, is_async=False)
            raise

    async def acompletion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        stop: list[str] | None = None,
    ) -> Any:
        """Async version of completion() with fallback support."""
        try:
            import litellm
        except ImportError:
            raise ImportError("Install LLM support with: pip install roshni[llm]")

        self._check_budget()
        kwargs = self._build_completion_kwargs(messages, tools=tools, stop=stop)

        try:
            response = await litellm.acompletion(**kwargs)
            self._record_response_usage(response)
            return response
        except (litellm.RateLimitError, litellm.APIError, litellm.APIConnectionError) as e:
            if self.fallback_model:
                return await self._fallback_completion(kwargs, e, is_async=True)
            raise

    def chat(self, message: str, stop: list[str] | None = None, **kwargs: Any) -> tuple[str, float]:
        """Send a message and return (response_text, elapsed_seconds).

        Respects token budget, maintains conversation history, and
        records token usage.
        """
        start = time()

        if not message:
            return "Input message is empty.", 0.0

        messages = self._build_messages(message, **kwargs)

        try:
            response = self.completion(messages, stop=stop)

            content = response.choices[0].message.content or ""
            text = extract_text_from_response(content)

            self.message_history.append({"role": "user", "content": message})
            if text.strip():
                self.message_history.append({"role": "assistant", "content": text})

            return text, time() - start

        except RuntimeError as e:
            # Budget exceeded
            return str(e), 0.0
        except Exception as e:
            logger.error(f"LLM error ({type(e).__name__}): {e}")
            return f"An error occurred while processing your request: {e!s}", time() - start

    async def achat(self, message: str, stop: list[str] | None = None, **kwargs: Any) -> tuple[str, float]:
        """Async version of chat()."""
        start = time()

        if not message:
            return "Input message is empty.", 0.0

        messages = self._build_messages(message, **kwargs)

        try:
            response = await self.acompletion(messages, stop=stop)

            content = response.choices[0].message.content or ""
            text = extract_text_from_response(content)

            self.message_history.append({"role": "user", "content": message})
            if text.strip():
                self.message_history.append({"role": "assistant", "content": text})

            return text, time() - start

        except RuntimeError as e:
            return str(e), 0.0
        except Exception as e:
            logger.error(f"LLM error ({type(e).__name__}): {e}")
            return f"An error occurred while processing your request: {e!s}", time() - start

    def format_system_prompt(self, **kwargs: Any) -> str:
        """Format the system prompt template with context variables."""
        return self.system_prompt.format(**kwargs)

    def clear_history(self) -> None:
        self.message_history = []

    def get_config_info(self) -> dict:
        info = {
            "provider": self.provider,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "max_history_messages": self.max_history_messages,
            "history_length": len(self.message_history),
        }
        if self.fallback_model:
            info["fallback_model"] = self.fallback_model
            info["fallback_provider"] = self.fallback_provider
        return info

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_budget(self) -> None:
        """Raise RuntimeError if daily token budget is exceeded."""
        within_budget, remaining = check_budget()
        if not within_budget:
            logger.warning(f"Daily token budget exceeded ({remaining} remaining)")
            raise RuntimeError("Daily token budget exceeded. Try again tomorrow.")

    def _build_completion_kwargs(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build kwargs dict for litellm.completion / acompletion."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "num_retries": self.num_retries,
        }
        if tools:
            kwargs["tools"] = tools
        if stop:
            kwargs["stop"] = stop
        return kwargs

    def _fallback_completion(self, kwargs: dict[str, Any], original_error: Exception, *, is_async: bool) -> Any:
        """Retry with the fallback model. Returns response or coroutine."""
        import litellm

        assert self.fallback_model  # caller checks before calling

        logger.warning(
            f"Primary model {self.model} failed ({type(original_error).__name__}), "
            f"falling back to {self.fallback_model}"
        )

        # Cap max_tokens to fallback model's limit
        fallback_limit = get_model_max_tokens(self.fallback_model, self.fallback_provider)
        kwargs["model"] = self.fallback_model
        kwargs["max_tokens"] = min(kwargs.get("max_tokens", fallback_limit), fallback_limit)

        if is_async:
            return self._afallback_completion(kwargs, litellm)

        response = litellm.completion(**kwargs)
        self._record_response_usage(response, provider=self.fallback_provider, model=self.fallback_model)
        return response

    async def _afallback_completion(self, kwargs: dict[str, Any], litellm: Any) -> Any:
        """Async fallback completion."""
        response = await litellm.acompletion(**kwargs)
        self._record_response_usage(response, provider=self.fallback_provider, model=self.fallback_model)
        return response

    def _record_response_usage(
        self, response: Any, *, provider: str | None = None, model: str | None = None
    ) -> None:
        """Record token usage from a litellm response."""
        usage = getattr(response, "usage", None)
        if usage:
            record_usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                provider=provider or self.provider,
                model=model or self.model,
            )

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
