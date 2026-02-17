"""Tests for EventGateway — serialized event processing."""

import asyncio

import pytest

from roshni.agent.base import BaseAgent, ChatResult
from roshni.gateway.event_gateway import EventGateway
from roshni.gateway.events import EventPriority, EventSource, GatewayEvent


class MockAgent(BaseAgent):
    """Records calls and returns canned responses."""

    def __init__(self, response: str = "mock response"):
        super().__init__(name="test")
        self.response = response
        self.calls: list[dict] = []

    def chat(self, message, *, mode=None, call_type=None, channel=None, max_iterations=5, **kwargs):
        self.calls.append({"message": message, "call_type": call_type, "channel": channel, "mode": mode})
        return ChatResult(text=self.response)


class ErrorAgent(BaseAgent):
    """Always raises an exception."""

    def __init__(self):
        super().__init__(name="error")

    def chat(self, message, **kwargs):
        raise RuntimeError("agent exploded")


@pytest.mark.smoke
class TestEventGateway:
    async def test_message_future_resolves(self):
        agent = MockAgent(response="hello back")
        gw = EventGateway(agent=agent)
        gw.start()

        event = GatewayEvent.message("hi", user_id="u1", channel="test")
        await gw.submit(event)
        result = await asyncio.wait_for(event._response_future, timeout=5.0)

        assert result == "hello back"
        assert len(agent.calls) == 1
        assert agent.calls[0]["message"] == "hi"
        assert agent.calls[0]["channel"] == "test"

        await gw.stop()

    async def test_heartbeat_calls_response_handler(self):
        agent = MockAgent(response="heartbeat done")
        gw = EventGateway(agent=agent)

        handled: list[tuple] = []

        async def handler(event, response):
            handled.append((event, response))

        gw.set_response_handler(handler)
        gw.start()

        event = GatewayEvent.heartbeat("check in")
        await gw.submit(event)
        # Give the consumer time to process
        await asyncio.sleep(0.2)

        assert len(handled) == 1
        assert handled[0][0] is event
        assert handled[0][1] == "heartbeat done"
        assert agent.calls[0]["call_type"] == "heartbeat"

        await gw.stop()

    async def test_source_specific_handler(self):
        agent = MockAgent()
        gw = EventGateway(agent=agent)

        heartbeat_responses: list[str] = []
        scheduled_responses: list[str] = []

        async def hb_handler(event, response):
            heartbeat_responses.append(response)

        async def sched_handler(event, response):
            scheduled_responses.append(response)

        gw.set_response_handler(hb_handler, source=EventSource.HEARTBEAT)
        gw.set_response_handler(sched_handler, source=EventSource.SCHEDULED)
        gw.start()

        await gw.submit(GatewayEvent.heartbeat("hb"))
        await gw.submit(GatewayEvent.scheduled("job", job_id="j1"))
        await asyncio.sleep(0.3)

        assert len(heartbeat_responses) == 1
        assert len(scheduled_responses) == 1

        await gw.stop()

    async def test_priority_ordering(self):
        """Message (HIGH) should be processed before heartbeat (LOW) when both are queued."""
        order: list[str] = []

        class OrderTrackingAgent(BaseAgent):
            def __init__(self):
                super().__init__(name="order")

            def chat(self, message, *, call_type=None, **kwargs):
                order.append(message)
                return ChatResult(text="ok")

        agent = OrderTrackingAgent()
        gw = EventGateway(agent=agent)

        # Don't start consumer yet — queue events first
        hb = GatewayEvent(source=EventSource.HEARTBEAT, message="heartbeat", priority=EventPriority.LOW, user_id="sys")
        msg = GatewayEvent(source=EventSource.MESSAGE, message="user_msg", priority=EventPriority.HIGH, user_id="u1")

        await gw.submit(hb)
        await gw.submit(msg)

        # Now start — consumer processes in priority order
        gw.start()
        await asyncio.sleep(0.3)
        await gw.stop()

        assert order == ["user_msg", "heartbeat"]

    async def test_queue_full_rejects_message_future(self):
        agent = MockAgent()
        gw = EventGateway(agent=agent, max_queue_size=1)

        # Fill the queue without starting consumer
        filler = GatewayEvent(source=EventSource.HEARTBEAT, message="filler", priority=EventPriority.LOW)
        await gw.submit(filler)

        # Now submit a message — should reject the future
        msg = GatewayEvent.message("overflow", user_id="u1")
        await gw.submit(msg)

        with pytest.raises(RuntimeError, match="queue is full"):
            await msg._response_future

    async def test_queue_full_drops_fire_and_forget(self):
        agent = MockAgent()
        gw = EventGateway(agent=agent, max_queue_size=1)

        filler = GatewayEvent(source=EventSource.HEARTBEAT, message="filler", priority=EventPriority.LOW)
        await gw.submit(filler)

        # This should be silently dropped (no exception)
        dropped = GatewayEvent.heartbeat("dropped")
        await gw.submit(dropped)

    async def test_agent_exception_resolves_future(self):
        agent = ErrorAgent()
        gw = EventGateway(agent=agent)
        gw.start()

        event = GatewayEvent.message("boom", user_id="u1")
        await gw.submit(event)

        with pytest.raises(RuntimeError, match="agent exploded"):
            await asyncio.wait_for(event._response_future, timeout=5.0)

        await gw.stop()

    async def test_agent_exception_doesnt_kill_consumer(self):
        """After an error, the consumer should keep processing."""
        agent = MockAgent(response="ok")
        gw = EventGateway(agent=agent)
        gw.start()

        # First: inject an event that will cause an error in dispatch
        # (by using a handler that raises)
        async def bad_handler(event, response):
            raise ValueError("handler broke")

        gw.set_response_handler(bad_handler)

        await gw.submit(GatewayEvent.heartbeat("will fail"))
        await asyncio.sleep(0.2)

        # Second: a message should still work
        msg = GatewayEvent.message("still works", user_id="u1")
        await gw.submit(msg)
        result = await asyncio.wait_for(msg._response_future, timeout=5.0)
        assert result == "ok"

        await gw.stop()

    async def test_start_stop_lifecycle(self):
        agent = MockAgent()
        gw = EventGateway(agent=agent)

        gw.start()
        assert gw._consumer_task is not None
        assert not gw._consumer_task.done()

        await gw.stop()
        assert gw._consumer_task is None

    async def test_double_start_is_noop(self):
        agent = MockAgent()
        gw = EventGateway(agent=agent)

        gw.start()
        task1 = gw._consumer_task
        gw.start()  # Should not create a new task
        assert gw._consumer_task is task1

        await gw.stop()
