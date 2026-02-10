"""Tests for roshni.core.events — EventBus and Event."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError

import pytest

from roshni.core.events import Event, EventBus

pytestmark = pytest.mark.smoke


# ---------------------------------------------------------------------------
# 1. on / off / emit lifecycle
# ---------------------------------------------------------------------------


async def test_on_off_emit_lifecycle():
    bus = EventBus()
    received: list[Event] = []

    def hook(event: Event) -> None:
        received.append(event)

    bus.on("test.event", hook)
    evt = Event(name="test.event", payload={"k": "v"}, source="test")
    await bus.emit(evt)

    assert len(received) == 1
    assert received[0] is evt

    bus.off("test.event", hook)
    await bus.emit(evt)

    assert len(received) == 1  # still 1 — hook was removed


# ---------------------------------------------------------------------------
# 2. Wildcard hooks via on_all
# ---------------------------------------------------------------------------


async def test_wildcard_hooks_receive_all_events():
    bus = EventBus()
    received: list[str] = []

    def wildcard(event: Event) -> None:
        received.append(event.name)

    bus.on_all(wildcard)

    await bus.emit(Event(name="alpha"))
    await bus.emit(Event(name="beta"))
    await bus.emit(Event(name="gamma"))

    assert received == ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# 3. emit_sync fires sync hooks
# ---------------------------------------------------------------------------


async def test_emit_sync_fires_sync_hooks():
    bus = EventBus()
    received: list[Event] = []

    def sync_hook(event: Event) -> None:
        received.append(event)

    bus.on("sync.event", sync_hook)
    evt = Event(name="sync.event", source="test")
    bus.emit_sync(evt)

    assert len(received) == 1
    assert received[0] is evt


# ---------------------------------------------------------------------------
# 4. Async hooks are awaited properly
# ---------------------------------------------------------------------------


async def test_async_hooks_awaited():
    bus = EventBus()
    received: list[Event] = []

    async def async_hook(event: Event) -> None:
        await asyncio.sleep(0)
        received.append(event)

    bus.on("async.event", async_hook)
    evt = Event(name="async.event", source="test")
    await bus.emit(evt)

    assert len(received) == 1
    assert received[0] is evt


# ---------------------------------------------------------------------------
# 5. Emit with no listeners does not raise
# ---------------------------------------------------------------------------


async def test_emit_no_listeners():
    bus = EventBus()
    await bus.emit(Event(name="nobody.listening"))


def test_emit_sync_no_listeners():
    bus = EventBus()
    bus.emit_sync(Event(name="nobody.listening"))


# ---------------------------------------------------------------------------
# 6. Hook exceptions don't prevent other hooks from running
# ---------------------------------------------------------------------------


async def test_hook_exception_does_not_block_others():
    bus = EventBus()
    received: list[str] = []

    def bad_hook(event: Event) -> None:
        raise RuntimeError("boom")

    def good_hook(event: Event) -> None:
        received.append("ok")

    bus.on("err.event", bad_hook)
    bus.on("err.event", good_hook)

    await bus.emit(Event(name="err.event"))
    assert received == ["ok"]


async def test_async_hook_exception_does_not_block_others():
    bus = EventBus()
    received: list[str] = []

    async def bad_hook(event: Event) -> None:
        raise RuntimeError("async boom")

    async def good_hook(event: Event) -> None:
        received.append("ok")

    bus.on("err.event", bad_hook)
    bus.on("err.event", good_hook)

    await bus.emit(Event(name="err.event"))
    assert received == ["ok"]


# ---------------------------------------------------------------------------
# 7. Event is frozen (immutable)
# ---------------------------------------------------------------------------


def test_event_is_frozen():
    evt = Event(name="frozen.test", payload={"x": 1}, source="test")
    with pytest.raises(FrozenInstanceError):
        evt.name = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        evt.source = "changed"  # type: ignore[misc]
