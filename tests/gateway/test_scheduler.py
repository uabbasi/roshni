"""Tests for GatewayScheduler — config-driven cron scheduling."""

import asyncio

import pytest

from roshni.gateway.events import EventSource, GatewayEvent
from roshni.gateway.scheduler import GatewayScheduler, ScheduleJob


class FakeConfig:
    """Minimal config mock supporting dot-notation get()."""

    def __init__(self, data: dict):
        self._data = data

    def get(self, key_path: str, default=None):
        parts = key_path.split(".")
        current = self._data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current


@pytest.mark.smoke
class TestScheduleJob:
    def test_defaults(self):
        job = ScheduleJob(id="test", prompt="do thing", cron={"hour": 9})
        assert job.call_type == "scheduled"
        assert job.channel == "scheduled"
        assert job.enabled is True
        assert job.metadata == {}


@pytest.mark.smoke
class TestGatewayScheduler:
    def test_add_heartbeat_requires_prompt(self):
        scheduler = GatewayScheduler(submit_fn=lambda e: None)
        with pytest.raises(ValueError, match="prompt"):
            scheduler.add_heartbeat(cron={"hour": 9})

    def test_add_heartbeat_with_prompt(self):
        scheduler = GatewayScheduler(submit_fn=lambda e: None)
        scheduler.add_heartbeat(cron={"hour": 9}, prompt="check in")
        assert len(scheduler._heartbeats) == 1

    def test_add_heartbeat_with_prompt_fn(self):
        scheduler = GatewayScheduler(submit_fn=lambda e: None)
        scheduler.add_heartbeat(cron={"hour": 9}, prompt_fn=lambda: "dynamic prompt")
        assert len(scheduler._heartbeats) == 1
        assert scheduler._heartbeats[0]["prompt_fn"] is not None

    def test_add_job(self):
        scheduler = GatewayScheduler(submit_fn=lambda e: None)
        job = ScheduleJob(id="brief", prompt="morning brief", cron={"hour": 8, "minute": 55})
        scheduler.add_job(job)
        assert len(scheduler._jobs) == 1

    def test_add_jobs_from_config_full(self):
        config = FakeConfig(
            {
                "scheduler": {
                    "enabled": True,
                    "timezone": "America/Los_Angeles",
                    "heartbeat": {
                        "enabled": True,
                        "cron": {"hour": "9,12", "minute": 0},
                    },
                    "jobs": [
                        {
                            "id": "morning_brief",
                            "prompt": "[BRIEF] Run morning brief.",
                            "cron": {"hour": 8, "minute": 55},
                            "call_type": "scheduled",
                            "channel": "heartbeat",
                        },
                        {
                            "id": "disabled_job",
                            "prompt": "skip me",
                            "cron": {"hour": 12},
                            "enabled": False,
                        },
                    ],
                }
            }
        )
        scheduler = GatewayScheduler(submit_fn=lambda e: None)
        scheduler.add_jobs_from_config(config)

        assert scheduler._timezone == "America/Los_Angeles"
        assert len(scheduler._heartbeats) == 1
        assert len(scheduler._jobs) == 1  # disabled job excluded
        assert scheduler._jobs[0].id == "morning_brief"

    def test_add_jobs_from_config_disabled(self):
        config = FakeConfig({"scheduler": {"enabled": False}})
        scheduler = GatewayScheduler(submit_fn=lambda e: None)
        scheduler.add_jobs_from_config(config)
        assert len(scheduler._heartbeats) == 0
        assert len(scheduler._jobs) == 0

    def test_add_jobs_from_config_no_scheduler_section(self):
        config = FakeConfig({})
        scheduler = GatewayScheduler(submit_fn=lambda e: None)
        scheduler.add_jobs_from_config(config)
        assert len(scheduler._heartbeats) == 0
        assert len(scheduler._jobs) == 0


@pytest.mark.smoke
class TestFireEvents:
    async def test_fire_heartbeat_creates_event(self):
        submitted: list[GatewayEvent] = []

        async def capture(event):
            submitted.append(event)

        scheduler = GatewayScheduler(submit_fn=capture)
        await scheduler._fire_heartbeat({"prompt": "check in", "heartbeat_type": "morning", "metadata": {}})

        assert len(submitted) == 1
        event = submitted[0]
        assert event.source == EventSource.HEARTBEAT
        assert event.message == "check in"
        assert event.call_type == "heartbeat"
        assert event.metadata["heartbeat_type"] == "morning"

    async def test_fire_heartbeat_with_prompt_fn(self):
        submitted: list[GatewayEvent] = []

        async def capture(event):
            submitted.append(event)

        scheduler = GatewayScheduler(submit_fn=capture)
        await scheduler._fire_heartbeat(
            {
                "prompt": "fallback",
                "prompt_fn": lambda: "dynamic prompt text",
                "heartbeat_type": "heartbeat",
                "metadata": {},
            }
        )

        assert submitted[0].message == "dynamic prompt text"

    async def test_fire_job_creates_event(self):
        submitted: list[GatewayEvent] = []

        async def capture(event):
            submitted.append(event)

        scheduler = GatewayScheduler(submit_fn=capture)
        job = ScheduleJob(id="brief", prompt="run brief", cron={}, call_type="scheduled", channel="heartbeat")
        await scheduler._fire_job(job)

        assert len(submitted) == 1
        event = submitted[0]
        assert event.source == EventSource.SCHEDULED
        assert event.message == "run brief"
        assert event.metadata["job_id"] == "brief"
        assert event.call_type == "scheduled"
        assert event.channel == "heartbeat"

    async def test_start_and_shutdown(self):
        """Verify the scheduler can start and shut down without error."""
        submitted: list[GatewayEvent] = []

        async def capture(event):
            submitted.append(event)

        scheduler = GatewayScheduler(submit_fn=capture)
        scheduler.add_heartbeat(cron={"hour": 23, "minute": 59}, prompt="late night")
        scheduler.start()

        assert scheduler.apscheduler is not None
        assert scheduler.apscheduler.running

        scheduler.shutdown()
        # AsyncIOScheduler needs an event loop tick to finalize state
        await asyncio.sleep(0)
        assert not scheduler.apscheduler.running
