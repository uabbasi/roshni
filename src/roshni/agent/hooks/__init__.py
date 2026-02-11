"""Built-in after-chat hooks for the agent framework."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from roshni.agent.circuit_breaker import CircuitBreaker
    from roshni.agent.memory import MemoryManager
    from roshni.core.llm.client import LLMClient
    from roshni.core.llm.model_selector import ModelSelector


class MemoryExtractionHook:
    """Extract and save standing instructions from user messages.

    Replaces the inline _fire_memory_extraction() in DefaultAgent.
    Runs in a background daemon thread.
    """

    name = "memory_extraction"

    def __init__(
        self,
        memory_manager: MemoryManager,
        llm: LLMClient,
        model_selector: ModelSelector | None = None,
    ) -> None:
        self._mm = memory_manager
        self._llm = llm
        self._model_selector = model_selector

    def run(
        self,
        *,
        message: str,
        response: str,
        tool_calls: list[dict[str, Any]],
        channel: str | None = None,
    ) -> None:
        if not self._mm.detect_trigger(message):
            return
        # Skip if the agent already called save_memory
        if any(tc.get("name") == "save_memory" for tc in tool_calls):
            return
        self._extract(message)

    def _extract(self, user_message: str) -> None:
        """Extract and save a standing instruction (runs synchronously)."""
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
            response = self._llm.completion(prompt_messages, model=light_model)
            extracted = (response.choices[0].message.content or "").strip()
            if not extracted or extracted.upper() == "NONE":
                return

            msg_lower = user_message.lower()
            if any(kw in msg_lower for kw in ("always", "never", "prefer", "like", "hate")):
                section = "preferences"
            else:
                section = "decisions"

            self._mm.save(section, extracted)
        except Exception as e:
            logger.debug(f"Memory auto-extraction failed (non-fatal): {e}")


class LoggingHook:
    """Wraps a legacy on_chat_complete callback as an AfterChatHook."""

    name = "logging"

    def __init__(self, callback: Callable[[str, str, list[dict[str, Any]]], None]) -> None:
        self._callback = callback

    def run(
        self,
        *,
        message: str,
        response: str,
        tool_calls: list[dict[str, Any]],
        channel: str | None = None,
    ) -> None:
        self._callback(message, response, tool_calls)


class MetricsHook:
    """Records tool outcomes into CircuitBreaker so SystemHealthAdvisor has data."""

    name = "metrics"

    def __init__(self, circuit_breaker: CircuitBreaker) -> None:
        self._cb = circuit_breaker

    def run(
        self,
        *,
        message: str,
        response: str,
        tool_calls: list[dict[str, Any]],
        channel: str | None = None,
    ) -> None:
        for tc in tool_calls:
            name = tc.get("name", "unknown")
            result_str = str(tc.get("result", ""))[:200].lower()
            success = "error" not in result_str
            self._cb.record(name, success=success, duration=tc.get("duration", 0.0))


__all__ = ["LoggingHook", "MemoryExtractionHook", "MetricsHook"]
