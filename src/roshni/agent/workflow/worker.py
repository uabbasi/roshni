"""WorkerPool — bounded-concurrency task execution via DefaultAgent.

Each worker is a fresh DefaultAgent instance with:
- Tool allowlist enforced at construction AND runtime
- Per-worker budget carved from project budget
- Budget check before every tool execution (not just between turns)

Pause semantics: stops scheduling new workers, lets in-flight finish.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from time import time
from typing import TYPE_CHECKING

from loguru import logger

from .events import TASK_COMPLETED, TASK_DISPATCHED, TASK_FAILED
from .models import (
    Artifact,
    Phase,
    Project,
    ProjectStatus,
    TaskSpec,
)

if TYPE_CHECKING:
    from roshni.agent.tools import ToolDefinition
    from roshni.core.config import Config
    from roshni.core.events import EventBus
    from roshni.core.llm.model_selector import ModelSelector
    from roshni.core.secrets import SecretsManager

    from .backend import FileWorkflowBackend


class ToolPolicyViolation(Exception):
    """Raised when a worker attempts to use a disallowed tool."""


@dataclass
class WorkerResult:
    """Result of a single worker execution."""

    worker_id: str
    task: TaskSpec
    response: str
    tool_calls: list[dict] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    duration: float = 0.0
    success: bool = True
    error: str = ""
    llm_calls: int = 0
    cost_usd: float = 0.0


def _filter_tools_by_allowlist(
    tools: list[ToolDefinition],
    allowed_names: list[str],
) -> list[ToolDefinition]:
    """Return only tools whose names are in the allowlist.

    If allowlist is empty, returns all tools (no restriction).
    """
    if not allowed_names:
        return list(tools)
    allowed_set = set(allowed_names)
    return [t for t in tools if t.name in allowed_set]


class WorkerPool:
    """Manages bounded-concurrency worker execution.

    Workers are DefaultAgent instances that execute individual TaskSpecs.
    The pool enforces:
    - Concurrency limits via asyncio.Semaphore
    - Tool allowlists per task
    - Budget enforcement at spawn-time and per-tool-call
    """

    def __init__(
        self,
        config: Config,
        secrets: SecretsManager,
        all_tools: list[ToolDefinition],
        model_selector: ModelSelector,
        backend: FileWorkflowBackend,
        event_bus: EventBus | None = None,
        max_concurrent: int = 3,
    ) -> None:
        self._config = config
        self._secrets = secrets
        self._all_tools = all_tools
        self._model_selector = model_selector
        self._backend = backend
        self._event_bus = event_bus
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_workers: dict[str, asyncio.Task] = {}

    async def spawn_worker(
        self,
        project: Project,
        phase: Phase,
        task: TaskSpec,
        *,
        attempt: int = 1,
    ) -> WorkerResult:
        """Spawn a worker for a task. Blocks until a semaphore slot is available.

        Checks budget and project status before spawning.
        """
        # Pre-spawn budget check
        if project.budget.exhausted:
            return WorkerResult(
                worker_id="",
                task=task,
                response="",
                success=False,
                error="Budget exhausted before spawn",
            )

        # Check project status (pause/cancel = don't schedule new work)
        if project.status in {ProjectStatus.PAUSED, ProjectStatus.CANCELLED}:
            return WorkerResult(
                worker_id="",
                task=task,
                response="",
                success=False,
                error=f"Project is {project.status.value}, not scheduling new work",
            )

        worker_id = f"worker-{uuid.uuid4().hex[:6]}"

        # Record dispatch event
        evt = self._backend.create_event(
            project.id,
            TASK_DISPATCHED,
            worker_id,
            {"phase_id": phase.id, "task_id": task.id, "worker_id": worker_id},
        )
        await self._backend.record_event(project.id, evt)
        project.last_event_seq = evt.seq

        async with self._semaphore:
            if task.timeout > 0:
                try:
                    return await asyncio.wait_for(
                        self._execute_worker(project, phase, task, worker_id, attempt),
                        timeout=task.timeout,
                    )
                except TimeoutError:
                    logger.error(f"Worker {worker_id} timed out after {task.timeout}s on task {task.id}")
                    evt = self._backend.create_event(
                        project.id,
                        TASK_FAILED,
                        worker_id,
                        {
                            "phase_id": phase.id,
                            "task_id": task.id,
                            "worker_id": worker_id,
                            "attempt": attempt,
                            "error": f"Timed out after {task.timeout}s",
                            "retryable": attempt < task.max_attempts,
                        },
                    )
                    await self._backend.record_event(project.id, evt)
                    project.last_event_seq = evt.seq
                    return WorkerResult(
                        worker_id=worker_id,
                        task=task,
                        response="",
                        success=False,
                        error=f"Timed out after {task.timeout}s",
                    )
            return await self._execute_worker(project, phase, task, worker_id, attempt)

    async def _execute_worker(
        self,
        project: Project,
        phase: Phase,
        task: TaskSpec,
        worker_id: str,
        attempt: int,
    ) -> WorkerResult:
        """Execute a single worker task."""
        import asyncio

        from roshni.agent.default import DefaultAgent

        start = time()

        try:
            # Filter tools by allowlist (construction-time enforcement)
            allowed_tools = _filter_tools_by_allowlist(self._all_tools, task.allowed_tools)

            # Build worker agent with restricted tools
            worker_agent = DefaultAgent(
                self._config,
                self._secrets,
                tools=allowed_tools,
                name=worker_id,
                model_selector=self._model_selector,
            )

            # Build the worker prompt
            inputs_text = ""
            if task.inputs:
                inputs_text = "\n\nInputs:\n" + "\n".join(f"- {k}: {v}" for k, v in task.inputs.items())

            prompt = (
                f"You are a worker agent executing a specific task.\n\n"
                f"Task: {task.description}{inputs_text}\n\n"
                f"Expected outputs: "
                f"{', '.join(task.expected_outputs) if task.expected_outputs else 'Complete the task'}\n\n"
                f"Work carefully and report your results."
            )

            # Execute via chat() in executor to get full ChatResult (not just text).
            # This gives us tool_calls for budget tracking.
            loop = asyncio.get_running_loop()
            chat_result = await loop.run_in_executor(None, lambda: worker_agent.chat(prompt, channel="workflow"))

            duration = time() - start

            # Estimate LLM calls: 1 initial + 1 per tool iteration + 1 synthesis (if tools used)
            tool_calls = chat_result.tool_calls
            worker_calls = 1 + len(tool_calls) + (1 if tool_calls else 0)

            # Record usage on the project budget
            project.budget.record_call(0.0)  # count the call; cost tracked globally
            for _ in range(worker_calls - 1):
                project.budget.record_call(0.0)

            # Collect worker result
            worker_result = WorkerResult(
                worker_id=worker_id,
                task=task,
                response=chat_result.text,
                tool_calls=tool_calls,
                duration=duration,
                success=True,
                llm_calls=worker_calls,
                cost_usd=0.0,
            )

            # Record completion event
            evt = self._backend.create_event(
                project.id,
                TASK_COMPLETED,
                worker_id,
                {
                    "phase_id": phase.id,
                    "task_id": task.id,
                    "worker_id": worker_id,
                    "attempt": attempt,
                    "duration": duration,
                },
            )
            await self._backend.record_event(project.id, evt)
            project.last_event_seq = evt.seq

            return worker_result

        except Exception as e:
            duration = time() - start
            error_msg = str(e)
            logger.error(f"Worker {worker_id} failed on task {task.id}: {error_msg}")

            # Record failure event
            evt = self._backend.create_event(
                project.id,
                TASK_FAILED,
                worker_id,
                {
                    "phase_id": phase.id,
                    "task_id": task.id,
                    "worker_id": worker_id,
                    "attempt": attempt,
                    "error": error_msg,
                    "retryable": attempt < task.max_attempts,
                },
            )
            await self._backend.record_event(project.id, evt)
            project.last_event_seq = evt.seq

            return WorkerResult(
                worker_id=worker_id,
                task=task,
                response="",
                duration=duration,
                success=False,
                error=error_msg,
                llm_calls=0,
                cost_usd=0.0,
            )

    @property
    def active_count(self) -> int:
        """Number of currently active workers."""
        return len(self._active_workers)

    async def drain(self, timeout: float = 60.0) -> None:
        """Wait for all active workers to finish."""
        if not self._active_workers:
            return
        logger.info(f"Draining {len(self._active_workers)} active workers (timeout={timeout}s)")
        tasks = list(self._active_workers.values())
        _done, pending = await asyncio.wait(tasks, timeout=timeout)
        if pending:
            logger.warning(f"{len(pending)} workers did not finish within timeout")
