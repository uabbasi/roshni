"""Tests for GatewayEvent dataclass and factory methods."""

import asyncio

import pytest

from roshni.gateway.events import EventPriority, EventSource, GatewayEvent


@pytest.mark.smoke
class TestGatewayEvent:
    def test_default_fields(self):
        event = GatewayEvent(source=EventSource.MESSAGE, message="hello")
        assert event.source == EventSource.MESSAGE
        assert event.message == "hello"
        assert len(event.id) == 12
        assert event.timestamp > 0
        assert event.priority == EventPriority.NORMAL
        assert event.call_type is None
        assert event.channel is None
        assert event.mode is None
        assert event.user_id == ""
        assert event.metadata == {}
        assert event._response_future is None

    def test_priority_ordering_by_priority(self):
        high = GatewayEvent(source=EventSource.MESSAGE, message="hi", priority=EventPriority.HIGH, timestamp=100.0)
        normal = GatewayEvent(source=EventSource.SCHEDULED, message="job", priority=EventPriority.NORMAL, timestamp=1.0)
        low = GatewayEvent(source=EventSource.HEARTBEAT, message="hb", priority=EventPriority.LOW, timestamp=1.0)
        assert high < normal
        assert normal < low
        assert high < low

    def test_priority_ordering_fifo_same_priority(self):
        first = GatewayEvent(source=EventSource.MESSAGE, message="a", priority=EventPriority.HIGH, timestamp=1.0)
        second = GatewayEvent(source=EventSource.MESSAGE, message="b", priority=EventPriority.HIGH, timestamp=2.0)
        assert first < second
        assert not second < first

    def test_priority_queue_ordering(self):
        """Verify events sort correctly via a PriorityQueue."""
        q: asyncio.PriorityQueue = asyncio.PriorityQueue()
        hb = GatewayEvent(source=EventSource.HEARTBEAT, message="hb", priority=EventPriority.LOW, timestamp=1.0)
        msg = GatewayEvent(source=EventSource.MESSAGE, message="msg", priority=EventPriority.HIGH, timestamp=2.0)
        sched = GatewayEvent(
            source=EventSource.SCHEDULED, message="sched", priority=EventPriority.NORMAL, timestamp=1.5
        )

        # Add in wrong order
        q.put_nowait(hb)
        q.put_nowait(sched)
        q.put_nowait(msg)

        assert q.get_nowait() is msg
        assert q.get_nowait() is sched
        assert q.get_nowait() is hb


@pytest.mark.smoke
class TestMessageFactory:
    async def test_creates_future(self):
        event = GatewayEvent.message("hello", user_id="u1")
        assert event._response_future is not None
        assert not event._response_future.done()

    async def test_correct_defaults(self):
        event = GatewayEvent.message("test", user_id="u1", channel="telegram")
        assert event.source == EventSource.MESSAGE
        assert event.priority == EventPriority.HIGH
        assert event.call_type is None
        assert event.channel == "telegram"
        assert event.user_id == "u1"

    async def test_metadata_passthrough(self):
        event = GatewayEvent.message("hi", user_id="u1", metadata={"key": "val"})
        assert event.metadata == {"key": "val"}


@pytest.mark.smoke
class TestHeartbeatFactory:
    def test_correct_defaults(self):
        event = GatewayEvent.heartbeat("check in")
        assert event.source == EventSource.HEARTBEAT
        assert event.priority == EventPriority.LOW
        assert event.call_type == "heartbeat"
        assert event.channel == "heartbeat"
        assert event._response_future is None

    def test_heartbeat_type_in_metadata(self):
        event = GatewayEvent.heartbeat("check", heartbeat_type="morning")
        assert event.metadata["heartbeat_type"] == "morning"


@pytest.mark.smoke
class TestScheduledFactory:
    def test_correct_defaults(self):
        event = GatewayEvent.scheduled("run brief", job_id="morning_brief")
        assert event.source == EventSource.SCHEDULED
        assert event.priority == EventPriority.NORMAL
        assert event.call_type == "scheduled"
        assert event.channel == "scheduled"
        assert event.metadata["job_id"] == "morning_brief"
        assert event._response_future is None
