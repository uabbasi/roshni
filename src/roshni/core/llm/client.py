"""
LLM Client — unified interface for LLM interactions via LiteLLM.

Provides a simple chat interface across multiple providers (OpenAI,
Anthropic, Gemini, Ollama) using litellm as the single routing layer.
"""

from collections.abc import Callable
from time import time
from types import SimpleNamespace
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
        auth_profiles: list[Any] | None = None,
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

        # Auth profile rotation
        self._auth_profile_manager = None
        if auth_profiles:
            from .auth_profiles import AuthProfileManager

            self._auth_profile_manager = AuthProfileManager(auth_profiles)

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
        model: str | None = None,
        thinking: dict[str, Any] | None = None,
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
            model: Optional per-request model override. When provided,
                uses this model instead of ``self.model`` for this single call.
            thinking: Optional thinking config, e.g.
                ``{"type": "enabled", "budget_tokens": 4096}``.

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
        kwargs = self._build_completion_kwargs(messages, tools=tools, stop=stop, model=model, thinking=thinking)

        _BadRequestError = getattr(litellm, "BadRequestError", None)

        try:
            response = litellm.completion(**kwargs)
            self._record_response_usage(response)
            return response
        except Exception as e:
            # Thinking not supported by this model — retry without it
            if _BadRequestError and isinstance(e, _BadRequestError):
                if thinking and "thinking" in str(e).lower():
                    logger.warning(f"Thinking not supported by {kwargs.get('model')}, retrying without")
                    kwargs.pop("thinking", None)
                    response = litellm.completion(**kwargs)
                    self._record_response_usage(response)
                    return response
                raise

            if not isinstance(e, (litellm.RateLimitError, litellm.APIError, litellm.APIConnectionError)):
                raise

            # Try auth profile rotation before falling back to a different model
            if self._auth_profile_manager:
                active = self._auth_profile_manager.get_active()
                if active:
                    self._auth_profile_manager.mark_failed(active.name)
                next_profile = self._auth_profile_manager.rotate()
                if next_profile:
                    kwargs["api_key"] = next_profile.api_key
                    if next_profile.model:
                        kwargs["model"] = next_profile.model
                    try:
                        response = litellm.completion(**kwargs)
                        self._auth_profile_manager.mark_success(next_profile.name)
                        self._record_response_usage(response)
                        return response
                    except (litellm.RateLimitError, litellm.APIError, litellm.APIConnectionError):
                        self._auth_profile_manager.mark_failed(next_profile.name)
            if self.fallback_model:
                return self._fallback_completion(kwargs, e, is_async=False)
            raise

    def stream_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        stop: list[str] | None = None,
        model: str | None = None,
        thinking: dict[str, Any] | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> Any:
        """Streaming completion that calls on_chunk for content deltas.

        Iterates chunks from litellm.completion(stream=True), accumulating
        content and tool_calls. Content deltas are forwarded to *on_chunk*
        only when NO tool_calls are detected (i.e. final response only).

        Returns a duck-typed response object matching the non-streaming shape
        so the caller (tool loop) can treat it identically.
        """
        try:
            import litellm
        except ImportError:
            raise ImportError("Install LLM support with: pip install roshni[llm]")

        self._check_budget()
        kwargs = self._build_completion_kwargs(messages, tools=tools, stop=stop, model=model, thinking=thinking)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}

        try:
            stream = litellm.completion(**kwargs)
        except Exception:
            # Fall back to non-streaming on any stream setup error
            logger.debug("Streaming setup failed, falling back to non-streaming")
            kwargs.pop("stream", None)
            kwargs.pop("stream_options", None)
            return self.completion(messages, tools=tools, stop=stop, model=model, thinking=thinking)

        return self._consume_stream(stream, on_chunk=on_chunk, model=model)

    def _consume_stream(
        self,
        stream: Any,
        *,
        on_chunk: Callable[[str], None] | None = None,
        model: str | None = None,
    ) -> Any:
        """Iterate a litellm stream, accumulate content/tool_calls, return assembled response."""
        content_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        usage = None
        has_tool_calls = False

        for chunk in stream:
            # Extract usage from final chunk
            chunk_usage = getattr(chunk, "usage", None)
            if chunk_usage and getattr(chunk_usage, "prompt_tokens", None):
                usage = chunk_usage

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta

            # Accumulate content
            if hasattr(delta, "content") and delta.content:
                content_parts.append(delta.content)
                # Only stream to callback if no tool calls detected yet
                if on_chunk and not has_tool_calls:
                    on_chunk(delta.content)

            # Accumulate tool calls
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                has_tool_calls = True
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_by_index:
                        tool_calls_by_index[idx] = {
                            "id": tc_delta.id or "",
                            "name": "",
                            "arguments": "",
                        }
                    entry = tool_calls_by_index[idx]
                    if tc_delta.id:
                        entry["id"] = tc_delta.id
                    if hasattr(tc_delta, "function") and tc_delta.function:
                        if tc_delta.function.name:
                            entry["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            entry["arguments"] += tc_delta.function.arguments

        response = self._assemble_stream_response(
            content="".join(content_parts),
            tool_calls_by_index=tool_calls_by_index,
            usage=usage,
        )
        self._record_response_usage(response, model=model)
        return response

    @staticmethod
    def _assemble_stream_response(
        *,
        content: str,
        tool_calls_by_index: dict[int, dict[str, Any]],
        usage: Any,
    ) -> Any:
        """Build a duck-typed response matching litellm's non-streaming shape."""
        # Build tool_calls list
        assembled_tool_calls = None
        if tool_calls_by_index:
            assembled_tool_calls = []
            for idx in sorted(tool_calls_by_index):
                tc = tool_calls_by_index[idx]
                assembled_tool_calls.append(
                    SimpleNamespace(
                        id=tc["id"],
                        type="function",
                        function=SimpleNamespace(
                            name=tc["name"],
                            arguments=tc["arguments"],
                        ),
                    )
                )

        message = SimpleNamespace(
            content=content or None,
            tool_calls=assembled_tool_calls,
        )
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(
            choices=[choice],
            usage=usage,
        )

    async def acompletion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        stop: list[str] | None = None,
        model: str | None = None,
        thinking: dict[str, Any] | None = None,
    ) -> Any:
        """Async version of completion() with fallback support."""
        try:
            import litellm
        except ImportError:
            raise ImportError("Install LLM support with: pip install roshni[llm]")

        self._check_budget()
        kwargs = self._build_completion_kwargs(messages, tools=tools, stop=stop, model=model, thinking=thinking)

        _BadRequestError = getattr(litellm, "BadRequestError", None)

        try:
            response = await litellm.acompletion(**kwargs)
            self._record_response_usage(response)
            return response
        except Exception as e:
            if _BadRequestError and isinstance(e, _BadRequestError):
                if thinking and "thinking" in str(e).lower():
                    logger.warning(f"Thinking not supported by {kwargs.get('model')}, retrying without")
                    kwargs.pop("thinking", None)
                    response = await litellm.acompletion(**kwargs)
                    self._record_response_usage(response)
                    return response
                raise

            if not isinstance(e, (litellm.RateLimitError, litellm.APIError, litellm.APIConnectionError)):
                raise

            if self._auth_profile_manager:
                active = self._auth_profile_manager.get_active()
                if active:
                    self._auth_profile_manager.mark_failed(active.name)
                next_profile = self._auth_profile_manager.rotate()
                if next_profile:
                    kwargs["api_key"] = next_profile.api_key
                    if next_profile.model:
                        kwargs["model"] = next_profile.model
                    try:
                        response = await litellm.acompletion(**kwargs)
                        self._auth_profile_manager.mark_success(next_profile.name)
                        self._record_response_usage(response)
                        return response
                    except (litellm.RateLimitError, litellm.APIError, litellm.APIConnectionError):
                        self._auth_profile_manager.mark_failed(next_profile.name)
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
        model: str | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build kwargs dict for litellm.completion / acompletion.

        Args:
            thinking: Optional thinking config, e.g.
                ``{"type": "enabled", "budget_tokens": 4096}``.
                Passed through to litellm for models that support extended thinking.
        """
        effective_model = model or self.model
        max_tokens = self.max_tokens
        if model and model != self.model:
            override_limit = get_model_max_tokens(model, infer_provider(model))
            max_tokens = min(self.max_tokens, override_limit)
        kwargs: dict[str, Any] = {
            "model": effective_model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "timeout": self.timeout,
            "num_retries": self.num_retries,
        }
        if tools:
            kwargs["tools"] = tools
        if stop:
            kwargs["stop"] = stop
        if extra_headers:
            kwargs["extra_headers"] = extra_headers
        if extra_body:
            kwargs["extra_body"] = extra_body
        if thinking:
            kwargs["thinking"] = thinking
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

        # Gemini 3 models require temperature >= 1.0 to avoid degraded performance
        if "gemini-3" in self.fallback_model and kwargs.get("temperature", 1.0) < 1.0:
            kwargs["temperature"] = 1.0

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

    def _record_response_usage(self, response: Any, *, provider: str | None = None, model: str | None = None) -> None:
        """Record token usage from a litellm response, including cache metrics and cost."""
        usage = getattr(response, "usage", None)
        if usage:
            # Anthropic: cache_creation_input_tokens, prompt_tokens_details.cached_tokens
            cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            cache_read = getattr(prompt_details, "cached_tokens", 0) or 0 if prompt_details else 0

            # Gemini: cached_content_token_count (different field name)
            if not cache_read:
                cache_read = getattr(usage, "cached_content_token_count", 0) or 0

            # Compute actual dollar cost via litellm
            cost = 0.0
            try:
                import litellm

                cost = litellm.completion_cost(completion_response=response) or 0.0
            except Exception:
                pass  # cost tracking never blocks

            record_usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                provider=provider or self.provider,
                model=model or self.model,
                cache_creation_tokens=cache_creation,
                cache_read_tokens=cache_read,
                cost_usd=cost,
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
