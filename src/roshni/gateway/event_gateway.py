"""Event gateway — serialized event processing through a priority queue.

Routes all gateway inputs (user messages, heartbeats, scheduled jobs)
through a single ``asyncio.PriorityQueue`` so that only one agent
invocation runs at a time, preventing concurrent mutation of shared
agent state (message history, tool context).

User messages carry an ``asyncio.Future`` for request/response semantics.
Fire-and-forget events (heartbeats, scheduled) route responses through
registered :data:`ResponseHandler` callbacks.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from roshni.agent.base import BaseAgent
from roshni.gateway.events import EventSource, GatewayEvent

ResponseHandler = Callable[[GatewayEvent, str], Awaitable[None]]
"""Async callback ``(event, agent_response) -> None`` for fire-and-forget events."""

_SENTINEL = object()


class EventGateway:
    """Serialized event processor backed by an asyncio priority queue.

    Events are submitted via :meth:`submit` and consumed one at a time
    by a background task.  Priority ordering ensures user messages
    (``HIGH``) are always processed before scheduled jobs (``NORMAL``)
    and heartbeats (``LOW``).

    Args:
        agent: The agent to invoke for each event.
        max_queue_size: Maximum number of pending events.  When full,
            message events have their Future rejected; fire-and-forget
            events are silently dropped.
    """

    def __init__(self, agent: BaseAgent, max_queue_size: int = 100):
        self._agent = agent
        self._queue: asyncio.PriorityQueue[Any] = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._consumer_task: asyncio.Task | None = None
        self._response_handlers: dict[EventSource | None, ResponseHandler] = {}
        self._dead_letters: list[tuple[GatewayEvent, str, float]] = []

    # ── Public API ─────────────────────────────────────────────────

    async def submit(self, event: GatewayEvent) -> None:
        """Add an event to the processing queue.

        If the queue is full:
        - Message events: the Future is rejected with a RuntimeError.
        - Fire-and-forget events: silently dropped with a warning log.
        """
        try:
            self._queue.put_nowait(event)
            logger.debug(f"Queued event {event.id} ({event.source.value}, pri={event.priority})")
        except asyncio.QueueFull:
            if event._response_future and not event._response_future.done():
                event._response_future.set_exception(RuntimeError("Event queue is full — try again later"))
                logger.warning(f"Queue full — rejected message event {event.id}")
            else:
                logger.warning(f"Queue full — dropped {event.source.value} event {event.id}")

    def set_response_handler(
        self,
        handler: ResponseHandler,
        source: EventSource | None = None,
    ) -> None:
        """Register a handler for fire-and-forget event responses.

        Args:
            handler: Async callback receiving ``(event, response_text)``.
            source: If given, handler only fires for events with this source.
                ``None`` registers a default/fallback handler.
        """
        self._response_handlers[source] = handler

    def start(self) -> None:
        """Create the background consumer task.

        Must be called from a running event loop (typically at startup).
        """
        if self._consumer_task and not self._consumer_task.done():
            logger.warning("EventGateway consumer already running")
            return
        self._consumer_task = asyncio.create_task(self._consume_loop(), name="event-gateway-consumer")
        logger.info("EventGateway consumer started")

    async def stop(self) -> None:
        """Gracefully shut down the consumer.

        Enqueues a sentinel that causes the loop to exit, then awaits
        the consumer task.
        """
        if not self._consumer_task:
            return
        # Sentinel with lowest possible priority so current work finishes first
        await self._queue.put(_SENTINEL)
        await self._consumer_task
        self._consumer_task = None
        logger.info("EventGateway consumer stopped")

    # ── Internal ───────────────────────────────────────────────────

    async def _consume_loop(self) -> None:
        """Pull events one at a time and process them."""
        while True:
            item = await self._queue.get()
            if item is _SENTINEL:
                self._queue.task_done()
                break
            try:
                await self._process_event(item)
            except Exception:
                logger.exception(f"Unhandled error processing event {item.id}")
            finally:
                self._queue.task_done()

    async def _process_event(self, event: GatewayEvent, *, is_retry: bool = False) -> None:
        """Invoke the agent and route the response."""
        logger.info(f"Processing event {event.id} ({event.source.value}): {event.message[:80]}")
        try:
            response = await self._agent.invoke(
                event.message,
                call_type=event.call_type,
                channel=event.channel,
                mode=event.mode,
            )
        except Exception as exc:
            logger.error(f"Agent error on event {event.id}: {exc}")
            if event._response_future and not event._response_future.done():
                event._response_future.set_exception(exc)
            elif not is_retry and event.call_type in ("scheduled", "heartbeat"):
                # One retry for scheduled/heartbeat events
                logger.info(f"Retrying scheduled event {event.id}")
                await self._process_event(event, is_retry=True)
            else:
                # Record in dead letter queue for introspection
                self._dead_letters.append((event, str(exc), time.time()))
                logger.warning(f"Event {event.id} moved to dead letter queue ({len(self._dead_letters)} total)")
            return

        # Route the response
        if event._response_future and not event._response_future.done():
            event._response_future.set_result(response)
        else:
            await self._dispatch_response(event, response)

    # ── Dead letter queue introspection ────────────────────────────

    @property
    def dead_letter_count(self) -> int:
        """Number of events in the dead letter queue."""
        return len(self._dead_letters)

    def get_dead_letters(self) -> list[tuple[GatewayEvent, str, float]]:
        """Return all dead letter entries as (event, error_message, timestamp)."""
        return list(self._dead_letters)

    def clear_dead_letters(self) -> None:
        """Clear the dead letter queue."""
        self._dead_letters.clear()

    async def _dispatch_response(self, event: GatewayEvent, response: str) -> None:
        """Call the appropriate response handler for a fire-and-forget event."""
        handler = self._response_handlers.get(event.source) or self._response_handlers.get(None)
        if handler:
            try:
                await handler(event, response)
            except Exception:
                logger.exception(f"Response handler error for event {event.id}")
        else:
            logger.debug(f"No handler for {event.source.value} response (event {event.id}), discarding")
