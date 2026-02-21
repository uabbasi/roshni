"""Data models for long-running agentic workflows.

Defines the core dataclasses for projects, phases, tasks, budgets,
terminal conditions, and events. Pure data — no I/O, no dependencies
beyond stdlib.

State machine:
    planning -> awaiting_approval -> executing -> reviewing -> done
    executing/reviewing -> paused -> executing/planning
    done/reviewing -> planning (advance: re-open for new work)
    Any non-terminal -> failed; failed -> planning (retry)
    Any non-terminal -> cancelled (terminal, only truly terminal state)
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal


class ProjectStatus(StrEnum):
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    PAUSED = "paused"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PhaseStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# Valid transitions: from_status -> set of allowed to_statuses
VALID_TRANSITIONS: dict[ProjectStatus, set[ProjectStatus]] = {
    ProjectStatus.PLANNING: {ProjectStatus.AWAITING_APPROVAL, ProjectStatus.FAILED, ProjectStatus.CANCELLED},
    ProjectStatus.AWAITING_APPROVAL: {
        ProjectStatus.EXECUTING,
        ProjectStatus.PLANNING,
        ProjectStatus.FAILED,
        ProjectStatus.CANCELLED,
    },
    ProjectStatus.EXECUTING: {
        ProjectStatus.REVIEWING,
        ProjectStatus.PAUSED,
        ProjectStatus.FAILED,
        ProjectStatus.CANCELLED,
    },
    ProjectStatus.REVIEWING: {
        ProjectStatus.DONE,
        ProjectStatus.PLANNING,  # replan to address unmet conditions
        ProjectStatus.PAUSED,
        ProjectStatus.FAILED,
        ProjectStatus.CANCELLED,
    },
    ProjectStatus.PAUSED: {
        ProjectStatus.EXECUTING,
        ProjectStatus.PLANNING,
        ProjectStatus.FAILED,
        ProjectStatus.CANCELLED,
    },
    ProjectStatus.DONE: {ProjectStatus.PLANNING},  # advance: re-open for new work
    ProjectStatus.FAILED: {ProjectStatus.PLANNING, ProjectStatus.CANCELLED},
    ProjectStatus.CANCELLED: set(),
}

TERMINAL_STATUSES = {ProjectStatus.CANCELLED}  # done is NOT terminal — projects can be advanced


@dataclass
class WorkflowEvent:
    """Append-only event with monotonic sequencing for deterministic replay.

    Replay rule: replay strictly by seq (not timestamp).
    Checkpoint stores last_event_seq as the replay cursor.
    """

    event_id: str  # monotonic: evt-000001, evt-000002, ...
    seq: int  # strict increment per project (replay key)
    type: str  # e.g. "project.transitioned", "phase.started"
    timestamp: str  # ISO 8601
    actor: str  # "orchestrator", "worker-{id}", "user", "system"
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class TerminalCondition:
    """A condition that determines when a project is done."""

    description: str
    type: Literal["artifact_exists", "phase_count", "llm_eval", "check_fn"]
    params: dict[str, Any] = field(default_factory=dict)
    met: bool = False
    met_at: str | None = None  # ISO 8601
    evaluation: dict | None = None  # for llm_eval: {met, rationale, evidence}


@dataclass
class PhaseEntry:
    """A single entry/exit criterion for a phase gate."""

    description: str
    met: bool = False


@dataclass
class ArtifactSpec:
    """Specification for an expected artifact output from a task."""

    name: str
    mime_type: str = "text/markdown"
    description: str = ""


@dataclass
class TaskSpec:
    """Defines what a worker should do and what it's allowed to use.

    Uses stable IDs (task-001, task-002) for event replay references,
    not list position.
    """

    id: str  # stable: task-001, task-002
    description: str
    allowed_tools: list[str] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)
    expected_outputs: list[str] = field(default_factory=list)
    artifact_outputs: list[ArtifactSpec] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)  # v1: ignored
    max_attempts: int = 1
    timeout: float = 300.0  # seconds; 0 = no timeout


@dataclass
class Phase:
    """A project phase with entry/exit criteria and tasks."""

    id: str
    name: str
    description: str = ""
    status: PhaseStatus = PhaseStatus.PENDING
    entry_criteria: list[PhaseEntry] = field(default_factory=list)
    exit_criteria: list[PhaseEntry] = field(default_factory=list)
    tasks: list[TaskSpec] = field(default_factory=list)
    started: str | None = None  # ISO 8601
    completed: str | None = None  # ISO 8601


@dataclass
class JournalEntry:
    """Human-readable log entry for project activity."""

    timestamp: str  # ISO 8601
    actor: str  # "orchestrator", "worker-{id}", "user", "system"
    action: str  # "created", "phase_advanced", "decision", "steering", "error", "budget_warning"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Budget:
    """Resource budget for a project with thread-safe tracking.

    record_call() is the ONLY path to log LLM usage.
    """

    max_cost_usd: float = 5.0
    max_llm_calls: int = 100
    max_wall_seconds: float = 3600.0
    cost_used_usd: float = 0.0
    llm_calls_used: int = 0
    wall_seconds_used: float = 0.0

    # Thread lock — excluded from serialization
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @property
    def exhausted(self) -> bool:
        """True if any budget dimension is exceeded."""
        return (
            self.cost_used_usd >= self.max_cost_usd
            or self.llm_calls_used >= self.max_llm_calls
            or self.wall_seconds_used >= self.max_wall_seconds
        )

    def remaining_fraction(self) -> float:
        """Return the smallest remaining fraction across all dimensions (0.0 = exhausted, 1.0 = unused)."""
        fractions = []
        if self.max_cost_usd > 0:
            fractions.append(max(0.0, 1.0 - self.cost_used_usd / self.max_cost_usd))
        if self.max_llm_calls > 0:
            fractions.append(max(0.0, 1.0 - self.llm_calls_used / self.max_llm_calls))
        if self.max_wall_seconds > 0:
            fractions.append(max(0.0, 1.0 - self.wall_seconds_used / self.max_wall_seconds))
        return min(fractions) if fractions else 1.0

    def record_call(self, cost_usd: float) -> None:
        """Thread-safe increment — the ONLY path to log LLM usage."""
        with self._lock:
            self.cost_used_usd += cost_usd
            self.llm_calls_used += 1

    def update_wall_time(self, started_at: datetime) -> None:
        """Called by checkpoint(). wall_seconds_used = now - started_at."""
        self.wall_seconds_used = (datetime.now() - started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict (excludes lock)."""
        return {
            "max_cost_usd": self.max_cost_usd,
            "max_llm_calls": self.max_llm_calls,
            "max_wall_seconds": self.max_wall_seconds,
            "cost_used_usd": self.cost_used_usd,
            "llm_calls_used": self.llm_calls_used,
            "wall_seconds_used": self.wall_seconds_used,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Budget:
        """Deserialize from dict."""
        return cls(
            max_cost_usd=data.get("max_cost_usd", 5.0),
            max_llm_calls=data.get("max_llm_calls", 100),
            max_wall_seconds=data.get("max_wall_seconds", 3600.0),
            cost_used_usd=data.get("cost_used_usd", 0.0),
            llm_calls_used=data.get("llm_calls_used", 0),
            wall_seconds_used=data.get("wall_seconds_used", 0.0),
        )


@dataclass
class Artifact:
    """A named output produced by a worker."""

    name: str
    path: str  # relative to workspace artifacts/
    mime_type: str = "text/markdown"
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class Project:
    """A long-running multi-phase project."""

    id: str
    goal: str
    status: ProjectStatus = ProjectStatus.PLANNING
    schema_version: int = 1
    terminal_conditions: list[TerminalCondition] = field(default_factory=list)
    phases: list[Phase] = field(default_factory=list)
    journal: list[JournalEntry] = field(default_factory=list)
    budget: Budget = field(default_factory=Budget)
    artifacts: list[Artifact] = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    updated: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    tags: list[str] = field(default_factory=list)
    obsidian_file: str = ""  # relative to projects_dir
    workspace_dir: str = ""
    started_at: str | None = None  # ISO 8601, set when entering EXECUTING
    cancel_requested_at: str | None = None  # ISO 8601, set when cancel requested
    last_orchestrator_update_at: str | None = None  # for Obsidian conflict detection
    last_event_seq: int = 0  # seq of last event applied (replay cursor)
    plan_hash: str = ""  # hash of canonical plan.json


def compute_plan_hash(project: Project) -> str:
    """Compute a deterministic hash of the plan-relevant parts of a project."""
    plan_data = {
        "phases": [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "entry_criteria": [{"description": e.description} for e in p.entry_criteria],
                "exit_criteria": [{"description": e.description} for e in p.exit_criteria],
                "tasks": [
                    {"id": t.id, "description": t.description, "allowed_tools": t.allowed_tools} for t in p.tasks
                ],
            }
            for p in project.phases
        ],
        "terminal_conditions": [
            {"description": tc.description, "type": tc.type, "params": tc.params} for tc in project.terminal_conditions
        ],
    }
    raw = json.dumps(plan_data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def validate_transition(current: ProjectStatus, target: ProjectStatus) -> None:
    """Raise ValueError if the transition is invalid."""
    allowed = VALID_TRANSITIONS.get(current, set())
    if target not in allowed:
        raise ValueError(
            f"Invalid transition: {current.value} -> {target.value}. "
            f"Allowed: {', '.join(s.value for s in sorted(allowed, key=lambda s: s.value))}"
        )


def _phase_to_dict(phase: Phase) -> dict[str, Any]:
    """Serialize a Phase to a dict (avoids asdict on nested dataclasses with Locks)."""
    return {
        "id": phase.id,
        "name": phase.name,
        "description": phase.description,
        "status": phase.status.value if isinstance(phase.status, PhaseStatus) else phase.status,
        "entry_criteria": [{"description": e.description, "met": e.met} for e in phase.entry_criteria],
        "exit_criteria": [{"description": e.description, "met": e.met} for e in phase.exit_criteria],
        "tasks": [
            {
                "id": t.id,
                "description": t.description,
                "allowed_tools": t.allowed_tools,
                "inputs": t.inputs,
                "expected_outputs": t.expected_outputs,
                "artifact_outputs": [
                    {"name": a.name, "mime_type": a.mime_type, "description": a.description} for a in t.artifact_outputs
                ],
                "depends_on": t.depends_on,
                "max_attempts": t.max_attempts,
            }
            for t in phase.tasks
        ],
        "started": phase.started,
        "completed": phase.completed,
    }


def project_to_dict(project: Project) -> dict[str, Any]:
    """Serialize a Project to a JSON-safe dict.

    Cannot use dataclasses.asdict() because Budget contains a threading.Lock
    which is not deepcopy-able.
    """
    return {
        "id": project.id,
        "goal": project.goal,
        "status": project.status.value if isinstance(project.status, ProjectStatus) else project.status,
        "schema_version": project.schema_version,
        "terminal_conditions": [
            {
                "description": tc.description,
                "type": tc.type,
                "params": tc.params,
                "met": tc.met,
                "met_at": tc.met_at,
                "evaluation": tc.evaluation,
            }
            for tc in project.terminal_conditions
        ],
        "phases": [_phase_to_dict(p) for p in project.phases],
        "journal": [
            {
                "timestamp": j.timestamp,
                "actor": j.actor,
                "action": j.action,
                "content": j.content,
                "metadata": j.metadata,
            }
            for j in project.journal
        ],
        "budget": project.budget.to_dict(),
        "artifacts": [
            {"name": a.name, "path": a.path, "mime_type": a.mime_type, "created": a.created} for a in project.artifacts
        ],
        "created": project.created,
        "updated": project.updated,
        "tags": project.tags,
        "obsidian_file": project.obsidian_file,
        "workspace_dir": project.workspace_dir,
        "started_at": project.started_at,
        "cancel_requested_at": project.cancel_requested_at,
        "last_orchestrator_update_at": project.last_orchestrator_update_at,
        "last_event_seq": project.last_event_seq,
        "plan_hash": project.plan_hash,
    }


def project_from_dict(data: dict[str, Any]) -> Project:
    """Deserialize a Project from a dict."""
    data = dict(data)  # shallow copy to avoid mutating caller's dict
    budget_data = data.pop("budget", {})
    phases_data = data.pop("phases", [])
    journal_data = data.pop("journal", [])
    tc_data = data.pop("terminal_conditions", [])
    artifacts_data = data.pop("artifacts", [])

    # Convert status string to enum
    if "status" in data and isinstance(data["status"], str):
        data["status"] = ProjectStatus(data["status"])

    phases = []
    for pd in phases_data:
        pd = dict(pd)  # shallow copy
        tasks = [TaskSpec(**t) for t in pd.pop("tasks", [])]
        entry_criteria = [PhaseEntry(**e) for e in pd.pop("entry_criteria", [])]
        exit_criteria = [PhaseEntry(**e) for e in pd.pop("exit_criteria", [])]
        for t in tasks:
            if isinstance(t.artifact_outputs, list):
                t.artifact_outputs = [ArtifactSpec(**a) if isinstance(a, dict) else a for a in t.artifact_outputs]
        # Convert phase status string to enum
        if "status" in pd and isinstance(pd["status"], str):
            pd["status"] = PhaseStatus(pd["status"])
        phases.append(Phase(**pd, tasks=tasks, entry_criteria=entry_criteria, exit_criteria=exit_criteria))

    journal = [JournalEntry(**j) for j in journal_data]
    terminal_conditions = [TerminalCondition(**tc) for tc in tc_data]
    artifacts = [Artifact(**a) for a in artifacts_data]

    return Project(
        **data,
        budget=Budget.from_dict(budget_data),
        phases=phases,
        journal=journal,
        terminal_conditions=terminal_conditions,
        artifacts=artifacts,
    )
