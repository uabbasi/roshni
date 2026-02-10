"""Event bus for loose-coupled extensibility.

Provides a lightweight publish/subscribe system that lets components
communicate without direct dependencies. Hooks can be sync or async.

Usage::

    from roshni.core.events import EventBus, Event, AGENT_CHAT_START

    bus = EventBus()

    async def log_chat(event: Event) -> None:
        print(f"Chat started: {event.payload}")

    bus.on(AGENT_CHAT_START, log_chat)
    await bus.emit(Event(name=AGENT_CHAT_START, payload={"msg": "hi"}, source="agent"))
"""

from __future__ import annotations

import asyncio
import inspect
from collections import defaultdict
from dataclasses import dataclass, field
from time import time
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Well-known event names
# ---------------------------------------------------------------------------

AGENT_CHAT_START = "agent.chat.start"
AGENT_CHAT_COMPLETE = "agent.chat.complete"
AGENT_TOOL_CALLED = "agent.tool.called"
AGENT_TOOL_RESULT = "agent.tool.result"
AGENT_MEMORY_SAVED = "agent.memory.saved"
HEALTH_COLLECTED = "health.collected"
JOURNAL_INDEXED = "journal.indexed"
ETL_COMPLETE = "etl.complete"
STARTUP = "startup"
SHUTDOWN = "shutdown"

# Type alias for hook callables (sync or async)
Hook = Any  # Callable[[Event], None] | Callable[[Event], Awaitable[None]]


# ---------------------------------------------------------------------------
# Event model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Event:
    """An immutable event that flows through the bus."""

    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time)
    source: str = ""


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


class EventBus:
    """Simple pub/sub event bus supporting sync and async hooks."""

    def __init__(self) -> None:
        self._hooks: dict[str, list[Hook]] = defaultdict(list)
        self._wildcard_hooks: list[Hook] = []
        self._background_tasks: set[asyncio.Task] = set()  # prevent GC of fire-and-forget tasks

    def on(self, event_name: str, hook: Hook) -> None:
        """Register *hook* for a specific event name."""
        self._hooks[event_name].append(hook)

    def on_all(self, hook: Hook) -> None:
        """Register *hook* for all events (wildcard)."""
        self._wildcard_hooks.append(hook)

    def off(self, event_name: str, hook: Hook) -> None:
        """Unregister *hook* from a specific event name."""
        try:
            self._hooks[event_name].remove(hook)
        except ValueError:
            pass

    async def emit(self, event: Event) -> None:
        """Emit an event, running all matching hooks (async)."""
        hooks = list(self._hooks.get(event.name, []))
        hooks.extend(self._wildcard_hooks)
        for hook in hooks:
            try:
                if inspect.iscoroutinefunction(hook):
                    await hook(event)
                else:
                    hook(event)
            except Exception as exc:
                logger.warning(f"Event hook failed for {event.name}: {exc}")

    def emit_sync(self, event: Event) -> None:
        """Emit from a sync context.

        If a running event loop exists, schedules async hooks as tasks.
        Otherwise, only runs sync hooks (async hooks are skipped with a warning).
        """
        hooks = list(self._hooks.get(event.name, []))
        hooks.extend(self._wildcard_hooks)

        loop: asyncio.AbstractEventLoop | None = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        for hook in hooks:
            try:
                if inspect.iscoroutinefunction(hook):
                    if loop is not None:
                        task = loop.create_task(hook(event))
                        self._background_tasks.add(task)
                        task.add_done_callback(self._background_tasks.discard)
                    else:
                        logger.debug(f"Skipping async hook {hook!r} — no running event loop")
                else:
                    hook(event)
            except Exception as exc:
                logger.warning(f"Event hook failed for {event.name}: {exc}")
