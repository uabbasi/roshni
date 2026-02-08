"""DefaultAgent — concrete agent with LiteLLM tool calling.

A ready-to-use agent that combines persona loading, tool execution,
and multi-turn conversation via LLMClient's provider-agnostic API.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from time import time
from typing import Any

from loguru import logger

from roshni.agent.base import BaseAgent, ChatResult
from roshni.agent.tools import ToolDefinition
from roshni.core.config import Config
from roshni.core.llm.client import LLMClient
from roshni.core.secrets import SecretsManager


class DefaultAgent(BaseAgent):
    """Tool-calling agent backed by LLMClient.

    Uses LLMClient.completion() for tool support, getting budget
    enforcement, usage tracking, and retry logic. Loads persona
    from config_dir, resolves model/provider from Config, and
    executes tools in a loop.
    """

    def __init__(
        self,
        config: Config,
        secrets: SecretsManager,
        tools: list[ToolDefinition] | None = None,
        *,
        system_prompt: str | None = None,
        persona_dir: str | None = None,
        name: str = "assistant",
        temperature: float = 0.7,
        max_history_messages: int = 20,
    ):
        super().__init__(name=name)

        self.config = config
        self.secrets = secrets
        self.tools = tools or []
        self.max_history_messages = max_history_messages

        # Build system prompt
        if system_prompt:
            resolved_prompt = system_prompt
        elif persona_dir:
            from roshni.agent.persona import get_system_prompt

            resolved_prompt = get_system_prompt(persona_dir)
        else:
            resolved_prompt = "You are a helpful personal AI assistant."

        # Create LLMClient — single place for model resolution, budget, usage
        llm_kwargs = self._resolve_llm_config(config)
        self._llm = LLMClient(
            **llm_kwargs,
            system_prompt=resolved_prompt,
            temperature=temperature,
            max_history_messages=None,  # We manage history ourselves
        )

        self.message_history: list[dict[str, Any]] = []

    @property
    def model(self) -> str:
        return self._llm.model

    @property
    def provider(self) -> str:
        return self._llm.provider

    def chat(
        self,
        message: str,
        *,
        mode: str | None = None,
        channel: str | None = None,
        max_iterations: int = 5,
        on_tool_start: Callable[[str, int, dict | None], None] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Process a message with tool-calling loop."""
        self._busy.set()
        start = time()
        tool_call_log: list[dict[str, Any]] = []

        try:
            # Add user message to history
            self.message_history.append({"role": "user", "content": message})

            # Build tool schemas
            tool_schemas = [t.to_litellm_schema() for t in self.tools] if self.tools else None

            for _iteration in range(max_iterations):
                # Check for steering messages
                steering = self.drain_steering()
                if steering:
                    self.message_history.append({"role": "user", "content": f"[STEERING] {steering}"})

                messages = self._build_messages()

                # Call LLM via LLMClient (budget + usage + retries)
                response = self._llm.completion(messages, tools=tool_schemas)
                choice = response.choices[0]
                assistant_message = choice.message

                # Add assistant response to history
                msg_dict: dict[str, Any] = {"role": "assistant"}
                if assistant_message.content:
                    msg_dict["content"] = assistant_message.content
                if hasattr(assistant_message, "tool_calls") and assistant_message.tool_calls:
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in assistant_message.tool_calls
                    ]
                self.message_history.append(msg_dict)

                # If no tool calls, we're done
                if not hasattr(assistant_message, "tool_calls") or not assistant_message.tool_calls:
                    break

                # Execute tool calls
                tool_map = {t.name: t for t in self.tools}
                for i, tool_call in enumerate(assistant_message.tool_calls):
                    fn_name = tool_call.function.name
                    fn_args = tool_call.function.arguments

                    if on_tool_start:
                        try:
                            args_dict = json.loads(fn_args) if isinstance(fn_args, str) else fn_args
                        except json.JSONDecodeError:
                            args_dict = {}
                        on_tool_start(fn_name, i, args_dict)

                    logger.debug(f"Tool call: {fn_name}({fn_args})")

                    tool = tool_map.get(fn_name)
                    if tool:
                        result = tool.execute(fn_args)
                    else:
                        result = f"Unknown tool: {fn_name}"

                    tool_call_log.append(
                        {
                            "name": fn_name,
                            "args": fn_args,
                            "result": result,
                        }
                    )

                    # Add tool result to history
                    self.message_history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result,
                        }
                    )

            # Extract final text
            final_text = ""
            for msg in reversed(self.message_history):
                if msg.get("role") == "assistant" and msg.get("content"):
                    final_text = msg["content"]
                    break

            # Trim history
            self._trim_history()

            return ChatResult(
                text=final_text,
                duration=time() - start,
                tool_calls=tool_call_log,
                model=self.model,
            )

        except Exception as e:
            logger.error(f"DefaultAgent error: {e}")
            return ChatResult(
                text=f"Sorry, something went wrong: {e}",
                duration=time() - start,
                tool_calls=tool_call_log,
                model=self.model,
            )
        finally:
            self._busy.clear()

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.message_history = []

    def _build_messages(self) -> list[dict[str, Any]]:
        """Build the messages list with system prompt and trimmed history."""
        messages: list[dict[str, Any]] = []

        if self._llm.system_prompt:
            messages.append({"role": "system", "content": self._llm.system_prompt})

        # Use recent history
        history = list(self.message_history)
        if self.max_history_messages and len(history) > self.max_history_messages:
            history = history[-self.max_history_messages :]
        messages.extend(history)

        return messages

    def _trim_history(self) -> None:
        """Keep history within bounds."""
        if self.max_history_messages and len(self.message_history) > self.max_history_messages * 2:
            self.message_history = self.message_history[-self.max_history_messages :]

    @staticmethod
    def _resolve_llm_config(config: Config) -> dict[str, Any]:
        """Read multi-provider or legacy LLM config into LLMClient kwargs.

        New format (llm.default / llm.providers):
            llm:
              default: anthropic
              fallback: openai
              providers:
                anthropic:
                  model: anthropic/claude-sonnet-4-20250514

        Legacy format (llm.provider / llm.model):
            llm:
              provider: openai
              model: gpt-4o-mini
        """
        from roshni.core.llm.config import get_default_model

        # Try new multi-provider format first
        default_provider = config.get("llm.default", "")
        if default_provider:
            providers_cfg: dict = config.get("llm.providers", {}) or {}
            provider_cfg = providers_cfg.get(default_provider, {}) or {}
            model = provider_cfg.get("model") or None

            fallback_name = config.get("llm.fallback", "")
            fallback_model = None
            fallback_provider = None
            if fallback_name:
                fallback_cfg = providers_cfg.get(fallback_name, {}) or {}
                fallback_model = fallback_cfg.get("model") or get_default_model(fallback_name)
                fallback_provider = fallback_name

            return {
                "model": model,
                "provider": default_provider,
                "fallback_model": fallback_model,
                "fallback_provider": fallback_provider,
            }

        # Legacy single-provider format
        provider = config.get("llm.provider", "openai")
        model = config.get("llm.model", "") or None
        return {
            "model": model,
            "provider": provider,
            "fallback_model": None,
            "fallback_provider": None,
        }
