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
from pathlib import Path
from time import time
from typing import Any

from loguru import logger

from roshni.agent.advisor import Advisor, AfterChatHook
from roshni.agent.approval import ApprovalStore
from roshni.agent.base import BaseAgent, ChatResult
from roshni.agent.tools import ToolDefinition
from roshni.core.config import Config
from roshni.core.events import (
    AGENT_CHAT_COMPLETE,
    AGENT_CHAT_START,
    AGENT_TOOL_CALLED,
    AGENT_TOOL_RESULT,
    Event,
    EventBus,
)
from roshni.core.llm.client import LLMClient
from roshni.core.llm.model_selector import ModelSelector
from roshni.core.secrets import SecretsManager

from .session import Session, SessionStore, Turn


def _build_runtime_context(
    *,
    model: str,
    provider: str,
    agent_name: str,
) -> str:
    """Build a short runtime metadata block for system prompt injection."""
    import platform
    import sys

    parts = [
        f"agent={agent_name}",
        f"model={model}",
        f"provider={provider}",
        f"python={sys.version_info.major}.{sys.version_info.minor}",
        f"os={platform.system()} ({platform.machine()})",
    ]
    return f"[RUNTIME] {' '.join(parts)}"


class DefaultAgent(BaseAgent):
    """Tool-calling agent backed by LLMClient.

    Uses LLMClient.completion() for tool support, getting budget
    enforcement, usage tracking, and retry logic. Loads persona
    from config_dir, resolves model/provider from Config, and
    executes tools in a loop.
    """

    @staticmethod
    def _extract_text(content: Any) -> str:
        """Extract plain text from message content (handles str and cache block lists)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    parts.append(block.get("text", ""))
            return " ".join(parts)
        return str(content) if content else ""

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
        event_bus: EventBus | None = None,
        session_store: SessionStore | None = None,
        max_tool_result_chars: int = 4000,
        archive_dir: str | None = None,
        min_context_tokens: int = 4096,
        tool_policy: Any | None = None,
        advisors: list[Advisor] | None = None,
        after_chat_hooks: list[AfterChatHook] | None = None,
        circuit_breaker: Any | None = None,
        persona_factory: Callable[[str | None], str] | None = None,
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
        self._event_bus = event_bus
        self._session_store = session_store
        self._active_session_id: str | None = None
        self._max_tool_result_chars = max_tool_result_chars
        self._archive_dir = archive_dir
        self._min_context_tokens = min_context_tokens
        self._tool_policy = tool_policy
        self._persona_factory = persona_factory

        # Advisor and hook registries
        self._advisors: list[Advisor] = list(advisors) if advisors else []
        self._after_chat_hooks: list[AfterChatHook] = list(after_chat_hooks) if after_chat_hooks else []

        # Build system prompt
        if system_prompt:
            resolved_prompt = system_prompt
            self._persona_prompt_compact = system_prompt
            self._persona_prompt_minimal = system_prompt
        elif persona_dir:
            from roshni.agent.persona import PromptMode, get_system_prompt

            resolved_prompt = get_system_prompt(persona_dir)
            self._persona_prompt_compact = get_system_prompt(persona_dir, mode=PromptMode.COMPACT)
            self._persona_prompt_minimal = get_system_prompt(persona_dir, mode=PromptMode.MINIMAL)
        else:
            resolved_prompt = "You are a helpful personal AI assistant."
            self._persona_prompt_compact = resolved_prompt
            self._persona_prompt_minimal = resolved_prompt

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

        # Approval grant store (persistent memory of user approvals)
        default_data_dir = config.get("paths.data_dir", "~/.roshni")
        grants_path = config.get(
            "security.approval_grants_path",
            str(Path(default_data_dir).expanduser() / "memory" / "approval-grants.json"),
        )
        self._approval_store = ApprovalStore(grants_path)

        # Channels that skip approval entirely (no user present)
        self._auto_approve_channels: set[str] = set(config.get("security.auto_approve_channels", []) or [])

        # Memory system
        self._memory_manager = None
        if memory_path:
            from roshni.agent.memory import MemoryManager, create_save_memory_tool

            self._memory_manager = MemoryManager(memory_path)
            self.tools.append(create_save_memory_tool(self._memory_manager))

            # Auto-register memory advisor + extraction hook
            from roshni.agent.advisors import MemoryAdvisor
            from roshni.agent.hooks import MemoryExtractionHook

            self._advisors.insert(0, MemoryAdvisor(self._memory_manager))
            self._after_chat_hooks.append(MemoryExtractionHook(self._memory_manager, self._llm, self._model_selector))

        # System health: wire CircuitBreaker into advisor + hook feedback loop
        from roshni.agent.advisors import SystemHealthAdvisor
        from roshni.agent.circuit_breaker import CircuitBreaker
        from roshni.agent.hooks import MetricsHook

        self._circuit_breaker = circuit_breaker if isinstance(circuit_breaker, CircuitBreaker) else CircuitBreaker()
        self._advisors.append(SystemHealthAdvisor(circuit_breaker=self._circuit_breaker))
        self._after_chat_hooks.append(MetricsHook(self._circuit_breaker))

        # Backward compat: wrap on_chat_complete as a hook
        if on_chat_complete:
            from roshni.agent.hooks import LoggingHook

            self._after_chat_hooks.append(LoggingHook(on_chat_complete))

        # Runtime context (injected into dynamic portion of system message)
        self._runtime_context = _build_runtime_context(
            model=self._llm.model,
            provider=self._llm.provider,
            agent_name=name,
        )

    @property
    def model(self) -> str:
        return self._llm.model

    @property
    def provider(self) -> str:
        return self._llm.provider

    @property
    def session_id(self) -> str | None:
        """Current active session ID, if any."""
        return self._active_session_id

    def resume_session(self, session_id: str) -> None:
        """Reload history from a stored session."""
        if not self._session_store:
            raise RuntimeError("No session_store configured")
        session = self._session_store.load_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id!r} not found")
        self._active_session_id = session_id
        self.message_history = [{"role": t.role, "content": t.content} for t in session.turns]

    def chat(
        self,
        message: str | list,
        *,
        mode: str | None = None,
        call_type: str | None = None,
        channel: str | None = None,
        max_iterations: int = 5,
        on_tool_start: Callable[[str, int, dict | None], None] | None = None,
        on_stream: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Process a message with tool-calling loop.

        Args:
            message: User message as a string, or a list of content blocks
                for multimodal input (e.g. text + images in OpenAI vision format).
            call_type: Optional call classification (e.g. ``"heartbeat"``,
                ``"scheduled"``).  Heartbeat/scheduled calls skip conversation
                history to reduce token usage.
        """
        self._busy.set()
        start = time()

        # Budget gate — fail fast with a clean message before adding to history
        from roshni.core.llm.token_budget import check_budget

        within_budget, _remaining = check_budget()
        if not within_budget:
            logger.warning("Daily budget exceeded — rejecting chat")
            self._busy.clear()
            return ChatResult(
                text="Daily token budget exceeded. Try again tomorrow.",
                duration=0.0,
                tool_calls=[],
                model=self.model,
            )

        # Extract text representation for logging/token counting/advisors
        message_text = self._extract_text(message) if isinstance(message, list) else message

        tool_call_log: list[dict[str, Any]] = []
        max_followups = int(kwargs.get("max_followups", 3))
        clear_history = call_type in ("heartbeat", "scheduled")

        # Session: lazily create on first chat()
        if self._session_store and not self._active_session_id:
            session = Session(agent_name=self.name, channel=channel or "")
            self._session_store.create_session(session)
            self._active_session_id = session.id

        # Emit AGENT_CHAT_START
        if self._event_bus:
            self._event_bus.emit_sync(
                Event(name=AGENT_CHAT_START, payload={"message": message_text, "channel": channel}, source=self.name)
            )

        try:
            # Apply tool policy filtering
            if self._tool_policy and self.tools:
                effective_tools = self._tool_policy.filter_tools(self.tools, channel=channel, agent_name=self.name)
            else:
                effective_tools = self.tools

            if self._pending_approval:
                result = self._handle_pending_approval(message_text, tool_call_log)
                if result is not None:
                    return result
                # Approved/denied — tools already executed, resume LLM loop
                selected_model = self._select_model(message_text, mode)
                tool_schemas = [t.to_litellm_schema() for t in effective_tools] if effective_tools else None
            else:
                # Track where this turn begins (for clear_history slicing)
                self._turn_start = len(self.message_history)

                # Add user message to history
                self.message_history.append({"role": "user", "content": message})

                # Compression check
                if self._enable_compression:
                    self._maybe_compress_history()

                # Model selection
                selected_model = self._select_model(message_text, mode)

                # Build tool schemas
                tool_schemas = [t.to_litellm_schema() for t in effective_tools] if effective_tools else None

            # Track selected model for accurate ChatResult reporting
            actual_model = selected_model or self.model

            # --- Main tool loop ---
            self._run_tool_loop(
                tool_schemas=tool_schemas,
                tool_call_log=tool_call_log,
                max_iterations=max_iterations,
                on_tool_start=on_tool_start,
                on_stream=on_stream,
                mode=mode,
                selected_model=selected_model,
                clear_history=clear_history,
                channel=channel,
                message=message_text,
            )
            if self._pending_approval:
                # Approval bail — _run_tool_loop set _pending_approval
                prompt = self._build_approval_prompt(
                    self._pending_approval["tool_name"],
                    self._pending_approval["parsed_args"],
                )
                self._trim_history()
                return ChatResult(
                    text=prompt,
                    duration=time() - start,
                    tool_calls=tool_call_log,
                    model=actual_model,
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
                    on_stream=None,
                    mode=mode,
                    selected_model=selected_model,
                    clear_history=clear_history,
                    channel=channel,
                    prompt_mode="compact",
                    message=followup_msg,
                )
                if self._pending_approval:
                    prompt = self._build_approval_prompt(
                        self._pending_approval["tool_name"],
                        self._pending_approval["parsed_args"],
                    )
                    self._trim_history()
                    return ChatResult(
                        text=prompt,
                        duration=time() - start,
                        tool_calls=tool_call_log,
                        model=actual_model,
                    )

            # --- Empty response synthesis ---
            final_text = self._extract_final_text()
            if tool_call_log and (not final_text or self._looks_like_thinking(final_text)):
                if final_text:
                    logger.info("Response looks like thinking/planning — re-synthesizing")
                final_text = self._synthesize_response(selected_model)

            # --- Proactive memory offer ---
            if final_text:
                from roshni.agent.memory import detect_memorable_event, detect_memory_trigger

                has_save_tool = any(t.name == "save_memory" for t in self.tools)
                used_save = any(tc.get("name") == "save_memory" for tc in tool_call_log)
                if has_save_tool and not used_save and not detect_memory_trigger(message_text):
                    if detect_memorable_event(message_text):
                        final_text += "\n\n*Should I save this to memory? It seems significant.*"

            # Trim history
            self._trim_history()

            result = ChatResult(
                text=final_text,
                duration=time() - start,
                tool_calls=tool_call_log,
                model=actual_model,
            )

            # Fire after-chat hooks (each in a daemon thread)
            self._fire_after_chat_hooks(message_text, final_text, tool_call_log, channel)

            # Emit AGENT_CHAT_COMPLETE
            if self._event_bus:
                self._event_bus.emit_sync(
                    Event(
                        name=AGENT_CHAT_COMPLETE,
                        payload={"message": message_text, "response": final_text, "tool_calls": tool_call_log},
                        source=self.name,
                    )
                )

            # Persist session turns
            if self._session_store and self._active_session_id:
                self._session_store.save_turn(
                    self._active_session_id,
                    Turn(role="user", content=message_text, metadata={"channel": channel or ""}),
                )
                self._session_store.save_turn(
                    self._active_session_id,
                    Turn(
                        role="assistant",
                        content=final_text,
                        metadata={
                            "model": actual_model,
                            "duration": result.duration,
                            "tools_called": [tc.get("name") for tc in tool_call_log],
                        },
                    ),
                )

            return result

        except Exception as e:
            logger.error(f"DefaultAgent error: {e}")
            return ChatResult(
                text=f"Sorry, something went wrong: {e}",
                duration=time() - start,
                tool_calls=tool_call_log,
                model=locals().get("actual_model", self.model),
            )
        finally:
            self._busy.clear()

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.message_history = []
        self._pending_approval = None

    # ------------------------------------------------------------------
    # Tool result truncation
    # ------------------------------------------------------------------

    @staticmethod
    def _truncate_tool_result(result: str, max_chars: int) -> str:
        """Truncate a tool result string if it exceeds *max_chars*."""
        if len(result) <= max_chars:
            return result
        return result[:max_chars] + f"\n\n[TRUNCATED: {len(result)} chars, showing first {max_chars}]"

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
        on_stream: Callable[[str], None] | None = None,
        mode: str | None = None,
        selected_model: str | None = None,
        clear_history: bool = False,
        channel: str | None = None,
        prompt_mode: str = "full",
        message: str = "",
    ) -> None:
        """Execute LLM → tool calls → record results loop.

        Mutates ``message_history`` and *tool_call_log* in place.
        If an approval-requiring tool is encountered, sets
        ``self._pending_approval`` and returns early.

        When *on_stream* is provided, the final LLM iteration (the one
        that produces content without tool_calls) is streamed, forwarding
        text deltas to the callback.
        """
        for _iteration in range(max_iterations):
            # Check for steering messages
            steering = self.drain_steering()
            if steering:
                self.message_history.append({"role": "user", "content": f"[STEERING] {steering}"})

            messages = self._build_messages(
                mode=mode, clear_history=clear_history, prompt_mode=prompt_mode, message=message, channel=channel
            )

            # Context window guard — stop loop if context is nearly full
            if not self._has_sufficient_context(messages):
                logger.warning("Context window nearly full, stopping tool loop")
                break

            # Build thinking config if the selected model has a thinking budget
            thinking_kwargs: dict[str, Any] | None = None
            model_cfg = getattr(self, "_last_selected_model_config", None)
            if model_cfg and getattr(model_cfg, "thinking_budget_tokens", None):
                thinking_kwargs = {"type": "enabled", "budget_tokens": model_cfg.thinking_budget_tokens}

            # Call LLM — use streaming when on_stream is provided
            if on_stream:
                response = self._llm.stream_completion(
                    messages, tools=tool_schemas, model=selected_model, thinking=thinking_kwargs, on_chunk=on_stream
                )
            else:
                response = self._llm.completion(
                    messages, tools=tool_schemas, model=selected_model, thinking=thinking_kwargs
                )
            choice = response.choices[0]
            assistant_message = choice.message

            # Add assistant response to history
            # Always include content as a string — OpenAI rejects null content
            msg_dict: dict[str, Any] = {
                "role": "assistant",
                "content": self._normalize_content(assistant_message.content) if assistant_message.content else "",
            }
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

                # Emit AGENT_TOOL_CALLED
                if self._event_bus:
                    self._event_bus.emit_sync(
                        Event(name=AGENT_TOOL_CALLED, payload={"tool": fn_name, "args": fn_args}, source=self.name)
                    )

                tool = tool_map.get(fn_name)
                if tool:
                    if self._should_require_approval(tool, channel=channel):
                        try:
                            parsed_args = json.loads(fn_args) if isinstance(fn_args, str) else fn_args
                        except json.JSONDecodeError:
                            parsed_args = {}

                        self._pending_approval = {
                            "tool": tool,
                            "tool_name": fn_name,
                            "raw_args": fn_args,
                            "parsed_args": parsed_args if isinstance(parsed_args, dict) else {},
                            "tool_call_id": tool_call.id,
                            "remaining_tool_calls": [
                                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                                for tc in assistant_message.tool_calls[i + 1 :]
                            ],
                        }
                        return  # Caller checks _pending_approval
                    result = tool.execute(fn_args)
                else:
                    result = f"Unknown tool: {fn_name}"

                # Emit AGENT_TOOL_RESULT
                if self._event_bus:
                    self._event_bus.emit_sync(
                        Event(
                            name=AGENT_TOOL_RESULT,
                            payload={"tool": fn_name, "result": result},
                            source=self.name,
                        )
                    )

                # Truncate oversized tool results
                result = self._truncate_tool_result(str(result), self._max_tool_result_chars)

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
                        "content": self._normalize_content(result),
                    }
                )

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        *,
        mode: str | None = None,
        clear_history: bool = False,
        prompt_mode: str = "full",
        message: str = "",
        channel: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build the messages list with system prompt, advisors, and trimmed history.

        Args:
            mode: Optional mode hint key.
            clear_history: When True (heartbeat/scheduled calls), only include
                messages from the current turn to reduce token usage.
            prompt_mode: One of ``"full"``, ``"compact"``, ``"minimal"``.
                Controls which persona prompt variant to use.
            message: Current user message (passed to advisors).
            channel: Current channel (passed to advisors).
        """
        from roshni.core.llm.caching import build_cached_system_message

        # Minimal mode: bare persona + history, skip advisors entirely.
        # Used for synthesis calls where advisor context wastes tokens.
        if prompt_mode == "minimal":
            messages: list[dict[str, Any]] = []
            if self._persona_prompt_minimal:
                messages.append({"role": "system", "content": self._persona_prompt_minimal})
            if clear_history:
                turn_start = getattr(self, "_turn_start", 0)
                history = list(self.message_history[turn_start:])
            else:
                history = list(self.message_history)
            messages.extend(history)
            for msg in messages:
                if msg.get("content") is None:
                    msg["content"] = ""
            return messages

        # Refresh persona via factory if provided (channel-aware timestamps, etc.)
        if self._persona_factory:
            self._persona_prompt = self._persona_factory(channel)

        messages = []

        # Select persona prompt variant
        if prompt_mode == "compact":
            persona = self._persona_prompt_compact
        else:
            persona = self._persona_prompt

        if persona:
            # Stable portion: persona (changes rarely, cacheable)
            # Dynamic portion: advisors + runtime + mode hints (changes per call)
            dynamic_parts: list[str] = []
            for advisor in self._advisors:
                try:
                    ctx = advisor.advise(message=message, channel=channel)
                    if ctx:
                        dynamic_parts.append(ctx)
                except Exception as e:
                    logger.warning(f"Advisor '{advisor.name}' failed: {e}")
            if self._runtime_context:
                dynamic_parts.append(self._runtime_context)
            if mode and mode in self._mode_hints:
                dynamic_parts.append(f"MODE HINT: {self._mode_hints[mode]}")

            system_msg = build_cached_system_message(
                stable_text=persona,
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

        # Sanitize: ensure history starts at a valid boundary.
        # After trimming, we may land mid-tool-sequence — e.g. orphaned
        # tool-result messages or an assistant+tool_calls without its results.
        # Strip until we reach a user message or a plain assistant message.
        while history:
            role = history[0].get("role")
            if role == "user":
                break
            if role == "assistant" and not history[0].get("tool_calls"):
                break
            history.pop(0)

        messages.extend(history)

        # Final pass: ensure every message has string content.
        # Some providers return null content on tool-call-only responses,
        # and OpenAI rejects any message with content=null.
        for msg in messages:
            if msg.get("content") is None:
                msg["content"] = ""

        return messages

    # ------------------------------------------------------------------
    # Content normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_content(content: Any) -> str:
        """Ensure message content is a string for API compatibility.

        LiteLLM wraps multiple providers; Gemini in particular can return
        content as a dict (``{"text": "..."}``), which violates the OpenAI
        message schema expected downstream.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Array of content blocks — extract text parts
            parts = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text", str(block)))
                else:
                    parts.append(str(block))
            return "\n".join(parts) if parts else ""
        if isinstance(content, dict):
            return content.get("text", json.dumps(content))
        return str(content)

    # ------------------------------------------------------------------
    # Empty response synthesis
    # ------------------------------------------------------------------

    def _extract_final_text(self) -> str:
        """Walk current-turn history backward to find the last assistant text.

        Only considers messages from the current turn (after _turn_start) to
        avoid picking up stale responses from previous queries.

        Skips assistant messages that carry tool_calls — those contain
        internal tool-calling syntax (e.g. Gemini's ``call:default_api:...``),
        not user-facing prose.
        """
        turn_start = getattr(self, "_turn_start", 0)
        turn_messages = self.message_history[turn_start:]
        for msg in reversed(turn_messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                if msg.get("tool_calls"):
                    continue
                return str(msg["content"])
        return ""

    @staticmethod
    def _looks_like_thinking(text: str) -> bool:
        """Detect if response text is model thinking/planning rather than a real answer.

        Some models (e.g. Gemini 3 Pro) can leak chain-of-thought reasoning as
        visible text.  Common signals: meta-commentary about strategy, explicit
        planning steps, and self-referential "I will" statements without actual
        content delivery.
        """
        if len(text) < 50:
            return False
        lower = text[:500].lower()
        signals = 0
        for pattern in (
            "my strategy is",
            "my plan is",
            "i will structure",
            "i have sufficient information",
            "i don't need more searches",
            "let me plan",
            "let me structure",
            "the user is asking",
            "the user wants",
            "the user is feeling",
            "quotes to use:",
            "plan:",
        ):
            if pattern in lower:
                signals += 1
        return signals >= 2

    def _synthesize_response(self, selected_model: str | None = None) -> str:
        """Force a text response when the LLM only made tool calls."""
        self.message_history.append(
            {"role": "user", "content": "Based on all the tool results above, provide a complete written response."}
        )
        messages = self._build_messages(prompt_mode="minimal")
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
        """Choose light, heavy, or thinking model via ModelSelector."""
        if not self._model_selector:
            return None
        config = self._model_selector.select(
            query,
            mode=mode,
            heavy_modes=self._heavy_modes,
        )
        self._last_selected_model_config = config
        return config.name

    # ------------------------------------------------------------------
    # Context window guard
    # ------------------------------------------------------------------

    def _has_sufficient_context(self, messages: list[dict[str, Any]]) -> bool:
        """Return True if there is enough context headroom to call the LLM."""
        from roshni.core.llm.token_management import estimate_token_count, get_model_context_limit

        total_text = " ".join(self._extract_text(m.get("content", "")) for m in messages if m.get("content"))
        total_tokens = estimate_token_count(total_text)
        context_limit = get_model_context_limit(self._llm.model, self._llm.provider)
        return total_tokens <= context_limit - self._min_context_tokens

    # ------------------------------------------------------------------
    # Pre-compaction archival
    # ------------------------------------------------------------------

    def _archive_conversation(self, messages: list[dict[str, Any]]) -> None:
        """Write *messages* to a markdown file in the archive directory (fire-and-forget)."""
        if not self._archive_dir:
            return
        try:
            from datetime import datetime

            archive_path = Path(self._archive_dir)
            archive_path.mkdir(parents=True, exist_ok=True)
            session_id = self._active_session_id or "nosession"
            filename = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{session_id[:8]}.md"
            lines: list[str] = []
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                lines.append(f"## {role}\n{content}\n")
            (archive_path / filename).write_text("\n".join(lines), encoding="utf-8")
            logger.debug(f"Archived {len(messages)} messages to {filename}")
        except Exception as e:
            logger.warning(f"Failed to archive conversation: {e}")

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
        total_text = " ".join(self._extract_text(m.get("content", "")) for m in history if m.get("content"))
        total_tokens = estimate_token_count(total_text)
        context_limit = get_model_context_limit(self._llm.model, self._llm.provider)

        if total_tokens < context_limit * 0.70:
            return

        logger.info(f"Compressing history: {total_tokens} tokens / {context_limit} limit")

        # Archive old messages before compacting
        old_messages = history[:-4]
        self._archive_conversation(old_messages)

        # Extract standing instructions from old messages
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

        system_content = self._persona_prompt_minimal or "You are a helpful assistant."
        summarize_instruction = "Summarize this conversation in 2-3 sentences. Be concise."
        prompt_messages = [
            {"role": "system", "content": f"{system_content}\n\n{summarize_instruction}"},
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

        total_text = " ".join(
            self._extract_text(m.get("content", "")) for m in self.message_history if m.get("content")
        )
        total_tokens = estimate_token_count(total_text)
        context_limit = get_model_context_limit(self._llm.model, self._llm.provider)
        return total_tokens < context_limit * threshold

    # ------------------------------------------------------------------
    # After-chat hooks
    # ------------------------------------------------------------------

    def _fire_after_chat_hooks(
        self,
        message: str,
        response: str,
        tool_calls: list[dict[str, Any]],
        channel: str | None = None,
    ) -> None:
        """Run each AfterChatHook in a daemon thread (fire-and-forget)."""
        for hook in self._after_chat_hooks:

            def _run(h: AfterChatHook = hook) -> None:
                try:
                    h.run(message=message, response=response, tool_calls=tool_calls, channel=channel)
                except Exception as e:
                    logger.warning(f"AfterChatHook '{h.name}' failed: {e}")

            threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------------
    # Advisor / hook registration
    # ------------------------------------------------------------------

    def add_advisor(self, advisor: Advisor) -> None:
        """Register an advisor for pre-chat context injection."""
        self._advisors.append(advisor)

    def add_after_chat_hook(self, hook: AfterChatHook) -> None:
        """Register a hook for post-chat side-effects."""
        self._after_chat_hooks.append(hook)

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

    def _should_require_approval(self, tool: ToolDefinition, *, channel: str | None = None) -> bool:
        if channel and channel in self._auto_approve_channels:
            return False
        if not (self._require_write_approval and tool.needs_approval()):
            return False
        return not self._approval_store.is_approved(tool.name)

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
        """Handle a pending approval prompt.

        Returns ``None`` when the approval is resolved (approve/deny) —
        signalling ``chat()`` to resume the tool loop.  Returns a
        ``ChatResult`` only when re-prompting (unrecognised input).
        """
        if not self._pending_approval:
            return None

        decision = self._approval_decision(message)
        tool = self._pending_approval["tool"]
        tool_name = self._pending_approval["tool_name"]
        raw_args = self._pending_approval["raw_args"]
        parsed_args = self._pending_approval["parsed_args"]
        tool_call_id = self._pending_approval["tool_call_id"]
        remaining = self._pending_approval.get("remaining_tool_calls", [])

        self.message_history.append({"role": "user", "content": message})

        if decision == "approve":
            # Execute the approved tool and add proper tool-result message
            result = tool.execute(raw_args)
            tool_call_log.append({"name": tool_name, "args": raw_args, "result": result})
            self.message_history.append(
                {"role": "tool", "tool_call_id": tool_call_id, "content": self._normalize_content(result)}
            )

            # Execute remaining tool calls from the same batch
            tool_map = {t.name: t for t in self.tools}
            for tc in remaining:
                tc_tool = tool_map.get(tc["name"])
                if tc_tool:
                    tc_result = tc_tool.execute(tc["arguments"])
                else:
                    tc_result = f"Unknown tool: {tc['name']}"
                tool_call_log.append({"name": tc["name"], "args": tc["arguments"], "result": tc_result})
                self.message_history.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": self._normalize_content(tc_result)}
                )

            self._approval_store.grant(tool_name)
            self._pending_approval = None
            return None  # Signal chat() to resume tool loop

        if decision == "deny":
            # Add error tool results for ALL pending tool calls (OpenAI requires
            # a result for every tool_call in the assistant message)
            error_msg = f"User denied execution of {tool_name}"
            self.message_history.append({"role": "tool", "tool_call_id": tool_call_id, "content": error_msg})
            for tc in remaining:
                self.message_history.append(
                    {"role": "tool", "tool_call_id": tc["id"], "content": f"Skipped: prior tool {tool_name} was denied"}
                )

            self._pending_approval = None
            return None  # Signal chat() to resume — LLM will see the errors

        # Unrecognised input — re-prompt
        prompt = self._build_approval_prompt(tool_name, parsed_args)
        self.message_history.append({"role": "assistant", "content": prompt})
        self._trim_history()
        return ChatResult(text=prompt, tool_calls=tool_call_log, model=self.model)

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
