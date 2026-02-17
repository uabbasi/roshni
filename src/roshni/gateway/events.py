"""Gateway events — unified event types for the event-driven gateway.

All gateway inputs (user messages, heartbeats, scheduled jobs, webhooks)
are normalized into GatewayEvent instances and routed through a single
asyncio.PriorityQueue.  Priority ordering ensures user messages are
always processed before background tasks.
"""

from __future__ import annotations

import asyncio
import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


class EventSource(enum.Enum):
    """Origin of a gateway event."""

    MESSAGE = "message"
    HEARTBEAT = "heartbeat"
    SCHEDULED = "scheduled"
    WEBHOOK = "webhook"
    BOOT = "boot"


class EventPriority(enum.IntEnum):
    """Processing priority — lower value = higher priority."""

    HIGH = 0  # User messages — always first
    NORMAL = 10  # Scheduled jobs (cron)
    LOW = 20  # Heartbeats, background


@dataclass
class GatewayEvent:
    """A single event flowing through the gateway queue.

    Supports ``asyncio.PriorityQueue`` ordering via ``__lt__``:
    events sort by (priority, timestamp) so higher-priority events
    are consumed first, with FIFO within the same priority tier.

    Factory methods provide ergonomic construction for common event
    types and enforce correct defaults (source, priority, call_type).
    """

    source: EventSource
    message: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    priority: EventPriority = EventPriority.NORMAL
    call_type: str | None = None
    channel: str | None = None
    mode: str | None = None
    user_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    _response_future: asyncio.Future | None = field(default=None, repr=False, compare=False)

    def __lt__(self, other: GatewayEvent) -> bool:
        """PriorityQueue ordering: lower priority value first, then earlier timestamp."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp

    # ── Factory methods ────────────────────────────────────────────

    @classmethod
    def message(
        cls,
        text: str,
        user_id: str,
        channel: str | None = None,
        *,
        mode: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GatewayEvent:
        """Create a user-initiated message event with a response Future."""
        loop = asyncio.get_running_loop()
        return cls(
            source=EventSource.MESSAGE,
            message=text,
            priority=EventPriority.HIGH,
            call_type=None,
            channel=channel,
            mode=mode,
            user_id=user_id,
            metadata=metadata or {},
            _response_future=loop.create_future(),
        )

    @classmethod
    def heartbeat(
        cls,
        prompt: str,
        heartbeat_type: str = "heartbeat",
        *,
        channel: str = "heartbeat",
        metadata: dict[str, Any] | None = None,
    ) -> GatewayEvent:
        """Create a fire-and-forget heartbeat event (no Future)."""
        return cls(
            source=EventSource.HEARTBEAT,
            message=prompt,
            priority=EventPriority.LOW,
            call_type="heartbeat",
            channel=channel,
            metadata={"heartbeat_type": heartbeat_type, **(metadata or {})},
        )

    @classmethod
    def scheduled(
        cls,
        prompt: str,
        job_id: str,
        *,
        call_type: str = "scheduled",
        channel: str = "scheduled",
        metadata: dict[str, Any] | None = None,
    ) -> GatewayEvent:
        """Create a scheduled job event (no Future)."""
        return cls(
            source=EventSource.SCHEDULED,
            message=prompt,
            priority=EventPriority.NORMAL,
            call_type=call_type,
            channel=channel,
            metadata={"job_id": job_id, **(metadata or {})},
        )
