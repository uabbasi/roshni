"""DefaultAgent — concrete agent with LiteLLM tool calling.

A ready-to-use agent that combines persona loading, tool execution,
and multi-turn conversation via LLMClient's provider-agnostic API.

Enhanced with: context compression, follow-up queue processing,
mode hints, model auto-selection, empty-response synthesis,
conversation logging, and persistent memory.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from time import time
from typing import Any

from loguru import logger

from roshni.agent.base import BaseAgent, ChatResult
from roshni.agent.tools import ToolDefinition
from roshni.core.config import Config
from roshni.core.llm.client import LLMClient
from roshni.core.llm.model_selector import ModelSelector
from roshni.core.secrets import SecretsManager

# Keywords that suggest a query needs a heavier model.
# TODO: Tune based on downstream usage patterns.
_COMPLEX_KEYWORDS: set[str] = {
    "analyze",
    "compare",
    "explain",
    "plan",
    "design",
    "refactor",
    "summarize",
    "review",
    "debug",
    "evaluate",
    "research",
    "strategy",
    "architect",
    "optimize",
    "trade-off",
    "tradeoff",
    "pros and cons",
}


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
        # -- New (all keyword-only, all have defaults → backward compatible) --
        mode_hints: dict[str, str] | None = None,
        model_selector: ModelSelector | None = None,
        heavy_modes: set[str] | None = None,
        enable_compression: bool = False,
        memory_path: str | None = None,
        on_chat_complete: Callable[[str, str, list[dict]], None] | None = None,
    ):
        super().__init__(name=name)

        self.config = config
        self.secrets = secrets
        self.tools = list(tools) if tools else []
        self.max_history_messages = max_history_messages

        # New feature state
        self._mode_hints = mode_hints or {}
        self._model_selector = model_selector
        self._heavy_modes = heavy_modes or set()
        self._enable_compression = enable_compression
        self._on_chat_complete = on_chat_complete

        # Build system prompt
        if system_prompt:
            resolved_prompt = system_prompt
        elif persona_dir:
            from roshni.agent.persona import get_system_prompt

            resolved_prompt = get_system_prompt(persona_dir)
        else:
            resolved_prompt = "You are a helpful personal AI assistant."

        # Store persona separately — stable prefix for caching
        self._persona_prompt = resolved_prompt

        # Create LLMClient — single place for model resolution, budget, usage
        llm_kwargs = self._resolve_llm_config(config)
        self._llm = LLMClient(
            **llm_kwargs,
            system_prompt=resolved_prompt,
            temperature=temperature,
            max_history_messages=None,  # We manage history ourselves
        )

        self.message_history: list[dict[str, Any]] = []
        self._pending_approval: dict[str, Any] | None = None
        self._require_write_approval = bool(config.get("security.require_write_approval", True))

        # Memory system
        self._memory_manager = None
        if memory_path:
            from roshni.agent.memory import MemoryManager, create_save_memory_tool

            self._memory_manager = MemoryManager(memory_path)
            self.tools.append(create_save_memory_tool(self._memory_manager))

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
        call_type: str | None = None,
        channel: str | None = None,
        max_iterations: int = 5,
        on_tool_start: Callable[[str, int, dict | None], None] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Process a message with tool-calling loop.

        Args:
            call_type: Optional call classification (e.g. ``"heartbeat"``,
                ``"scheduled"``).  Heartbeat/scheduled calls skip conversation
                history to reduce token usage.
        """
        self._busy.set()
        start = time()
        tool_call_log: list[dict[str, Any]] = []
        max_followups = int(kwargs.get("max_followups", 3))
        clear_history = call_type in ("heartbeat", "scheduled")

        try:
            pending = self._handle_pending_approval(message, tool_call_log)
            if pending is not None:
                return pending

            # Track where this turn begins (for clear_history slicing)
            self._turn_start = len(self.message_history)

            # Add user message to history
            self.message_history.append({"role": "user", "content": message})

            # Compression check
            if self._enable_compression:
                self._maybe_compress_history()

            # Model selection
            selected_model = self._select_model(message, mode)

            # Build tool schemas
            tool_schemas = [t.to_litellm_schema() for t in self.tools] if self.tools else None

            # --- Main tool loop ---
            self._run_tool_loop(
                tool_schemas=tool_schemas,
                tool_call_log=tool_call_log,
                max_iterations=max_iterations,
                on_tool_start=on_tool_start,
                mode=mode,
                selected_model=selected_model,
                clear_history=clear_history,
            )
            if self._pending_approval:
                # Approval bail — _run_tool_loop set _pending_approval
                prompt = self.message_history[-1].get("content", "")
                self._trim_history()
                return ChatResult(
                    text=prompt,
                    duration=time() - start,
                    tool_calls=tool_call_log,
                    model=self.model,
                )

            # --- Follow-up queue processing ---
            followups = self.drain_followups()[:max_followups]
            for followup_msg in followups:
                if not self._within_token_budget(0.85):
                    logger.info("Skipping remaining followups — approaching context limit")
                    break
                self.message_history.append({"role": "user", "content": followup_msg})
                self._run_tool_loop(
                    tool_schemas=tool_schemas,
                    tool_call_log=tool_call_log,
                    max_iterations=max_iterations,
                    on_tool_start=on_tool_start,
                    mode=mode,
                    selected_model=selected_model,
                    clear_history=clear_history,
                )
                if self._pending_approval:
                    prompt = self.message_history[-1].get("content", "")
                    self._trim_history()
                    return ChatResult(
                        text=prompt,
                        duration=time() - start,
                        tool_calls=tool_call_log,
                        model=self.model,
                    )

            # --- Empty response synthesis ---
            final_text = self._extract_final_text()
            if not final_text and tool_call_log:
                final_text = self._synthesize_response(selected_model)

            # Trim history
            self._trim_history()

            result = ChatResult(
                text=final_text,
                duration=time() - start,
                tool_calls=tool_call_log,
                model=self.model,
            )

            # Fire logging hook (daemon thread — won't block response)
            if self._on_chat_complete:
                self._fire_chat_complete(message, final_text, tool_call_log)

            # Fire memory auto-extraction if needed
            if self._memory_manager and self._memory_manager.detect_trigger(message):
                save_memory_called = any(tc.get("name") == "save_memory" for tc in tool_call_log)
                if not save_memory_called:
                    self._fire_memory_extraction(message, selected_model)

            return result

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
        self._pending_approval = None

    # ------------------------------------------------------------------
    # Tool loop
    # ------------------------------------------------------------------

    def _run_tool_loop(
        self,
        *,
        tool_schemas: list[dict[str, Any]] | None,
        tool_call_log: list[dict[str, Any]],
        max_iterations: int,
        on_tool_start: Callable[[str, int, dict | None], None] | None,
        mode: str | None = None,
        selected_model: str | None = None,
        clear_history: bool = False,
    ) -> None:
        """Execute LLM → tool calls → record results loop.

        Mutates ``message_history`` and *tool_call_log* in place.
        If an approval-requiring tool is encountered, sets
        ``self._pending_approval`` and returns early.
        """
        for _iteration in range(max_iterations):
            # Check for steering messages
            steering = self.drain_steering()
            if steering:
                self.message_history.append({"role": "user", "content": f"[STEERING] {steering}"})

            messages = self._build_messages(mode=mode, clear_history=clear_history)

            # Call LLM via LLMClient (budget + usage + retries)
            response = self._llm.completion(messages, tools=tool_schemas, model=selected_model)
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
                    if self._should_require_approval(tool):
                        try:
                            parsed_args = json.loads(fn_args) if isinstance(fn_args, str) else fn_args
                        except json.JSONDecodeError:
                            parsed_args = {}

                        self._pending_approval = {
                            "tool": tool,
                            "tool_name": fn_name,
                            "raw_args": fn_args,
                            "parsed_args": parsed_args if isinstance(parsed_args, dict) else {},
                        }
                        prompt = self._build_approval_prompt(fn_name, self._pending_approval["parsed_args"])
                        self.message_history.append({"role": "assistant", "content": prompt})
                        return  # Caller checks _pending_approval
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

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def _build_messages(self, *, mode: str | None = None, clear_history: bool = False) -> list[dict[str, Any]]:
        """Build the messages list with system prompt, memory, and trimmed history.

        Args:
            mode: Optional mode hint key.
            clear_history: When True (heartbeat/scheduled calls), only include
                messages from the current turn to reduce token usage.
        """
        from roshni.core.llm.caching import build_cached_system_message

        messages: list[dict[str, Any]] = []

        if self._persona_prompt:
            # Stable portion: persona (changes rarely, cacheable)
            # Dynamic portion: memory + mode hints (changes per call)
            dynamic_parts: list[str] = []
            if self._memory_manager:
                mem_ctx = self._memory_manager.get_context()
                if mem_ctx:
                    dynamic_parts.append(mem_ctx)
            if mode and mode in self._mode_hints:
                dynamic_parts.append(f"MODE HINT: {self._mode_hints[mode]}")

            system_msg = build_cached_system_message(
                stable_text=self._persona_prompt,
                dynamic_text="\n\n".join(dynamic_parts) if dynamic_parts else None,
                provider=self._llm.provider,
            )
            messages.append(system_msg)

        # History: skip old history for heartbeat/scheduled calls
        if clear_history:
            turn_start = getattr(self, "_turn_start", 0)
            history = list(self.message_history[turn_start:])
        else:
            history = list(self.message_history)
            if self.max_history_messages and len(history) > self.max_history_messages:
                history = history[-self.max_history_messages :]
        messages.extend(history)

        return messages

    # ------------------------------------------------------------------
    # Empty response synthesis
    # ------------------------------------------------------------------

    def _extract_final_text(self) -> str:
        """Walk history backward to find the last assistant text."""
        for msg in reversed(self.message_history):
            if msg.get("role") == "assistant" and msg.get("content"):
                return str(msg["content"])
        return ""

    def _synthesize_response(self, selected_model: str | None = None) -> str:
        """Force a text response when the LLM only made tool calls."""
        self.message_history.append(
            {"role": "user", "content": "Based on all the tool results above, provide a complete written response."}
        )
        messages = self._build_messages()
        try:
            response = self._llm.completion(messages, model=selected_model)
            text = response.choices[0].message.content or ""
            if text:
                self.message_history.append({"role": "assistant", "content": text})
            return text
        except Exception as e:
            logger.warning(f"Synthesis call failed: {e}")
            return ""

    # ------------------------------------------------------------------
    # Model auto-selection
    # ------------------------------------------------------------------

    def _select_model(self, query: str, mode: str | None) -> str | None:
        """Choose light or heavy model based on query complexity and mode."""
        if not self._model_selector:
            return None

        # Mode-based override
        if mode and mode in self._heavy_modes:
            return self._model_selector.heavy_model.name

        # Keyword / length heuristic
        query_lower = query.lower()
        if len(query) > 150 or any(kw in query_lower for kw in _COMPLEX_KEYWORDS):
            return self._model_selector.heavy_model.name

        return self._model_selector.light_model.name

    # ------------------------------------------------------------------
    # Context compression
    # ------------------------------------------------------------------

    _STANDING_INSTRUCTION_RE = re.compile(
        r"(?:always|never|remember|don'?t forget|from now on|going forward)\s+.+",
        re.IGNORECASE,
    )

    def _maybe_compress_history(self) -> None:
        """Compress old history when approaching the context limit."""
        from roshni.core.llm.token_management import estimate_token_count, get_model_context_limit

        history = self.message_history
        if len(history) <= 4:
            return

        # Estimate total tokens in history
        total_text = " ".join(m.get("content", "") for m in history if m.get("content"))
        total_tokens = estimate_token_count(total_text)
        context_limit = get_model_context_limit(self._llm.model, self._llm.provider)

        if total_tokens < context_limit * 0.70:
            return

        logger.info(f"Compressing history: {total_tokens} tokens / {context_limit} limit")

        # Extract standing instructions from old messages
        old_messages = history[:-4]
        standing: list[str] = []
        for msg in old_messages:
            content = msg.get("content", "")
            for match in self._STANDING_INSTRUCTION_RE.finditer(content):
                standing.append(match.group(0).strip())

        # Try LLM summarisation
        summary = self._summarize_old_messages(old_messages)

        # Build compressed history
        compressed: list[dict[str, Any]] = []
        if standing:
            compressed.append(
                {"role": "user", "content": "[STANDING INSTRUCTIONS]\n" + "\n".join(f"- {s}" for s in standing)}
            )
        if summary:
            compressed.append({"role": "user", "content": f"[CONVERSATION SUMMARY]\n{summary}"})

        # Keep last 4 messages
        compressed.extend(history[-4:])
        self.message_history = compressed

    def _summarize_old_messages(self, messages: list[dict[str, Any]]) -> str:
        """Use the LLM to summarise old messages. Returns empty string on failure."""
        text_parts = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if content:
                text_parts.append(f"{role}: {content[:200]}")
        if not text_parts:
            return ""

        prompt_messages = [
            {"role": "system", "content": "Summarize this conversation in 2-3 sentences. Be concise."},
            {"role": "user", "content": "\n".join(text_parts)},
        ]
        try:
            light_model = self._model_selector.light_model.name if self._model_selector else None
            response = self._llm.completion(prompt_messages, model=light_model)
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            logger.warning(f"Summarization failed, falling back to truncation: {e}")
            return ""

    # ------------------------------------------------------------------
    # Token budget helper
    # ------------------------------------------------------------------

    def _within_token_budget(self, threshold: float = 0.85) -> bool:
        """Return True if current history is within *threshold* of context limit."""
        from roshni.core.llm.token_management import estimate_token_count, get_model_context_limit

        total_text = " ".join(m.get("content", "") for m in self.message_history if m.get("content"))
        total_tokens = estimate_token_count(total_text)
        context_limit = get_model_context_limit(self._llm.model, self._llm.provider)
        return total_tokens < context_limit * threshold

    # ------------------------------------------------------------------
    # Logging hook
    # ------------------------------------------------------------------

    def _fire_chat_complete(self, user_msg: str, response: str, tool_calls: list[dict]) -> None:
        """Fire on_chat_complete in a daemon thread (fire-and-forget)."""

        def _run() -> None:
            try:
                self._on_chat_complete(user_msg, response, tool_calls)  # type: ignore[misc]
            except Exception as e:
                logger.warning(f"on_chat_complete hook failed: {e}")

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # Memory auto-extraction
    # ------------------------------------------------------------------

    def _fire_memory_extraction(self, user_message: str, selected_model: str | None) -> None:
        """Background daemon thread: extract and save a standing instruction."""

        def _run() -> None:
            try:
                prompt_messages = [
                    {
                        "role": "system",
                        "content": (
                            "Extract the standing instruction or preference from the following message. "
                            "Reply with ONLY the extracted instruction, nothing else. "
                            "If there is no clear instruction, reply with exactly: NONE"
                        ),
                    },
                    {"role": "user", "content": user_message},
                ]
                light_model = self._model_selector.light_model.name if self._model_selector else None
                model = light_model or selected_model
                response = self._llm.completion(prompt_messages, model=model)
                extracted = (response.choices[0].message.content or "").strip()
                if not extracted or extracted.upper() == "NONE":
                    return

                # Determine section
                msg_lower = user_message.lower()
                if any(kw in msg_lower for kw in ("always", "never", "prefer", "like", "hate")):
                    section = "preferences"
                else:
                    section = "decisions"

                self._memory_manager.save(section, extracted)  # type: ignore[union-attr]
            except Exception as e:
                logger.debug(f"Memory auto-extraction failed (non-fatal): {e}")

        t = threading.Thread(target=_run, daemon=True)
        t.start()

    # ------------------------------------------------------------------
    # History management
    # ------------------------------------------------------------------

    def _trim_history(self) -> None:
        """Keep history within bounds."""
        if self.max_history_messages and len(self.message_history) > self.max_history_messages * 2:
            self.message_history = self.message_history[-self.max_history_messages :]

    # ------------------------------------------------------------------
    # Approval workflow
    # ------------------------------------------------------------------

    def _should_require_approval(self, tool: ToolDefinition) -> bool:
        return self._require_write_approval and tool.needs_approval()

    @staticmethod
    def _approval_decision(message: str) -> str:
        normalized = (message or "").strip().lower()
        if normalized in {"approve", "yes", "y", "confirm", "ok", "proceed"}:
            return "approve"
        if normalized in {"deny", "no", "n", "cancel", "stop"}:
            return "deny"
        return "unknown"

    @staticmethod
    def _format_args_for_prompt(args: dict[str, Any]) -> str:
        if not args:
            return "(no arguments)"
        shown = []
        for key, value in args.items():
            text = str(value)
            if len(text) > 120:
                text = text[:117] + "..."
            shown.append(f"- {key}: {text}")
        return "\n".join(shown)

    def _build_approval_prompt(self, tool_name: str, args: dict[str, Any]) -> str:
        return (
            "Approval required for write action:\n"
            f"- Tool: {tool_name}\n"
            f"{self._format_args_for_prompt(args)}\n\n"
            "Reply `approve` to continue or `deny` to cancel."
        )

    def _handle_pending_approval(self, message: str, tool_call_log: list[dict[str, Any]]) -> ChatResult | None:
        if not self._pending_approval:
            return None

        decision = self._approval_decision(message)
        tool = self._pending_approval["tool"]
        tool_name = self._pending_approval["tool_name"]
        raw_args = self._pending_approval["raw_args"]
        parsed_args = self._pending_approval["parsed_args"]

        self.message_history.append({"role": "user", "content": message})

        if decision == "approve":
            result = tool.execute(raw_args)
            tool_call_log.append({"name": tool_name, "args": raw_args, "result": result})
            self._pending_approval = None
            text = f"Approved and executed `{tool_name}`.\n\n{result}"
            self.message_history.append({"role": "assistant", "content": text})
            self._trim_history()
            return ChatResult(text=text, tool_calls=tool_call_log, model=self.model)

        if decision == "deny":
            self._pending_approval = None
            text = f"Canceled `{tool_name}`."
            self.message_history.append({"role": "assistant", "content": text})
            self._trim_history()
            return ChatResult(text=text, tool_calls=tool_call_log, model=self.model)

        text = self._build_approval_prompt(tool_name, parsed_args)
        self.message_history.append({"role": "assistant", "content": text})
        self._trim_history()
        return ChatResult(text=text, tool_calls=tool_call_log, model=self.model)

    # ------------------------------------------------------------------
    # Config resolution
    # ------------------------------------------------------------------

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
