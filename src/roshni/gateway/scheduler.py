"""Gateway scheduler — config-driven cron scheduling via APScheduler.

Wraps APScheduler's ``AsyncIOScheduler`` to fire :class:`GatewayEvent`
instances into an event queue.  Supports heartbeats, named scheduled
jobs, and config-driven job loading.

APScheduler is imported lazily (only in :meth:`start`) so the module
can be imported without triggering heavy dependencies at import time.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from roshni.gateway.events import GatewayEvent

SubmitFn = Callable[[GatewayEvent], Awaitable[None]]
"""Async callable that accepts a GatewayEvent — typically ``event_gateway.submit``."""

PromptFn = Callable[[], str]
"""Sync callable returning a prompt string — for dynamic heartbeat prompts."""


@dataclass
class ScheduleJob:
    """A named scheduled job definition."""

    id: str
    prompt: str
    cron: dict[str, Any]
    call_type: str = "scheduled"
    channel: str = "scheduled"
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class GatewayScheduler:
    """Config-driven scheduler that submits GatewayEvents on cron triggers.

    Args:
        submit_fn: Async callable to submit events (typically
            ``event_gateway.submit``).  Dependency-injected so the
            scheduler is decoupled from the gateway.
        timezone: Default timezone for cron triggers.
    """

    def __init__(self, submit_fn: SubmitFn, timezone: str = "UTC"):
        self._submit_fn = submit_fn
        self._timezone = timezone
        self._scheduler: Any = None  # AsyncIOScheduler, lazily created
        self._heartbeats: list[dict[str, Any]] = []
        self._jobs: list[ScheduleJob] = []

    # ── Registration ───────────────────────────────────────────────

    def add_heartbeat(
        self,
        cron: dict[str, Any],
        *,
        prompt: str | None = None,
        prompt_fn: PromptFn | None = None,
        heartbeat_type: str = "heartbeat",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a heartbeat with either a static prompt or a dynamic prompt_fn.

        At least one of ``prompt`` or ``prompt_fn`` must be provided.
        If both are given, ``prompt_fn`` takes precedence at fire time.
        """
        if not prompt and not prompt_fn:
            raise ValueError("Heartbeat requires either prompt or prompt_fn")
        self._heartbeats.append(
            {
                "cron": cron,
                "prompt": prompt,
                "prompt_fn": prompt_fn,
                "heartbeat_type": heartbeat_type,
                "metadata": metadata or {},
            }
        )

    def add_job(self, job: ScheduleJob) -> None:
        """Register a named scheduled job."""
        self._jobs.append(job)

    def add_jobs_from_config(self, config: Any) -> None:
        """Load scheduler configuration from a Config object.

        Expected config layout::

            scheduler:
              enabled: true
              timezone: America/Los_Angeles
              heartbeat:
                enabled: true
                cron:
                  hour: "9,12,15,18,21"
                  minute: 0
              jobs:
                - id: morning_brief
                  prompt: "[MORNING BRIEF] Run the morning brief protocol."
                  cron: { hour: 8, minute: 55 }
                  call_type: scheduled
                  channel: heartbeat
        """
        if not config.get("scheduler.enabled", False):
            logger.info("Scheduler disabled in config")
            return

        tz = config.get("scheduler.timezone")
        if tz:
            self._timezone = tz

        # Heartbeat
        hb_cfg = config.get("scheduler.heartbeat", {})
        if hb_cfg and hb_cfg.get("enabled", True):
            cron = hb_cfg.get("cron", {})
            if cron:
                self.add_heartbeat(
                    cron=cron,
                    prompt=hb_cfg.get("prompt", "[HEARTBEAT] Check in."),
                )

        # Named jobs
        jobs_cfg = config.get("scheduler.jobs", []) or []
        for job_data in jobs_cfg:
            job = ScheduleJob(
                id=job_data["id"],
                prompt=job_data["prompt"],
                cron=job_data.get("cron", {}),
                call_type=job_data.get("call_type", "scheduled"),
                channel=job_data.get("channel", "scheduled"),
                enabled=job_data.get("enabled", True),
                metadata=job_data.get("metadata", {}),
            )
            if job.enabled:
                self.add_job(job)

    # ── Lifecycle ──────────────────────────────────────────────────

    def start(self) -> None:
        """Create the APScheduler instance, add all jobs, and start.

        Must be called from a running asyncio event loop.
        """
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger

        self._scheduler = AsyncIOScheduler(timezone=self._timezone)

        # Register heartbeats
        for i, hb in enumerate(self._heartbeats):
            trigger = CronTrigger(timezone=self._timezone, **hb["cron"])
            job_id = f"heartbeat_{i}"
            self._scheduler.add_job(
                self._fire_heartbeat,
                trigger=trigger,
                id=job_id,
                kwargs={"heartbeat_def": hb},
                replace_existing=True,
            )
            logger.info(f"Registered heartbeat {job_id}: cron={hb['cron']}")

        # Register named jobs
        for job in self._jobs:
            trigger = CronTrigger(timezone=self._timezone, **job.cron)
            self._scheduler.add_job(
                self._fire_job,
                trigger=trigger,
                id=job.id,
                kwargs={"job": job},
                replace_existing=True,
            )
            logger.info(f"Registered job {job.id}: cron={job.cron}")

        self._scheduler.start()
        total = len(self._heartbeats) + len(self._jobs)
        logger.info(f"GatewayScheduler started with {total} job(s), tz={self._timezone}")

    def shutdown(self) -> None:
        """Stop the APScheduler instance."""
        if self._scheduler:
            self._scheduler.shutdown()
            logger.info("GatewayScheduler shut down")

    @property
    def apscheduler(self) -> Any:
        """Expose the raw APScheduler instance for one-off jobs (e.g. boot actuation).

        Returns ``None`` if :meth:`start` hasn't been called yet.
        """
        return self._scheduler

    # ── Internal event firing ──────────────────────────────────────

    async def _fire_heartbeat(self, heartbeat_def: dict[str, Any]) -> None:
        """Create and submit a heartbeat event."""
        prompt_fn = heartbeat_def.get("prompt_fn")
        prompt = prompt_fn() if prompt_fn else heartbeat_def.get("prompt", "[HEARTBEAT]")
        event = GatewayEvent.heartbeat(
            prompt=prompt,
            heartbeat_type=heartbeat_def.get("heartbeat_type", "heartbeat"),
            metadata=heartbeat_def.get("metadata"),
        )
        logger.debug(f"Heartbeat fired: {event.id}")
        await self._submit_fn(event)

    async def _fire_job(self, job: ScheduleJob) -> None:
        """Create and submit a scheduled job event."""
        event = GatewayEvent.scheduled(
            prompt=job.prompt,
            job_id=job.id,
            call_type=job.call_type,
            channel=job.channel,
            metadata=job.metadata,
        )
        logger.debug(f"Job fired: {job.id} -> event {event.id}")
        await self._submit_fn(event)
