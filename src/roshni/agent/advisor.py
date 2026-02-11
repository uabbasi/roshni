"""Advisor and AfterChatHook protocols for agent extensibility.

Advisors inject context into the system prompt before each LLM call.
AfterChatHooks run side-effects after chat() completes.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Advisor(Protocol):
    """Pre-chat context injector.

    Returns text to inject into the system prompt's dynamic section,
    or empty string to skip. Must be fast (<500ms) — no LLM calls.
    """

    name: str

    def advise(self, *, message: str, channel: str | None = None) -> str:
        """Return context to inject, or empty string."""
        ...


@runtime_checkable
class AfterChatHook(Protocol):
    """Post-chat side-effect handler.

    Runs after chat() completes. Failures are logged, never propagated.
    """

    name: str

    def run(
        self,
        *,
        message: str,
        response: str,
        tool_calls: list[dict[str, Any]],
        channel: str | None = None,
    ) -> None:
        """Execute post-chat side-effect."""
        ...


class FunctionAdvisor:
    """Wrap a plain callable as an Advisor.

    The wrapped function is introspected: only kwargs it accepts are passed.
    This lets the roshni app wrap existing enricher functions without creating classes::

        advisor = FunctionAdvisor("daily_quote", get_daily_quote)
    """

    def __init__(self, name: str, fn: Callable[..., str]) -> None:
        self.name = name
        self._fn = fn
        self._params = set(inspect.signature(fn).parameters.keys())

    def advise(self, *, message: str, channel: str | None = None) -> str:
        kwargs: dict[str, Any] = {}
        if "message" in self._params:
            kwargs["message"] = message
        if "channel" in self._params:
            kwargs["channel"] = channel
        return self._fn(**kwargs)


class FunctionAfterChatHook:
    """Wrap a plain callable as an AfterChatHook.

    Same signature introspection as FunctionAdvisor::

        hook = FunctionAfterChatHook("my_hook", lambda message, response: log(message))
    """

    def __init__(self, name: str, fn: Callable[..., None]) -> None:
        self.name = name
        self._fn = fn
        self._params = set(inspect.signature(fn).parameters.keys())

    def run(
        self,
        *,
        message: str,
        response: str,
        tool_calls: list[dict[str, Any]],
        channel: str | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if "message" in self._params:
            kwargs["message"] = message
        if "response" in self._params:
            kwargs["response"] = response
        if "tool_calls" in self._params:
            kwargs["tool_calls"] = tool_calls
        if "channel" in self._params:
            kwargs["channel"] = channel
        self._fn(**kwargs)
