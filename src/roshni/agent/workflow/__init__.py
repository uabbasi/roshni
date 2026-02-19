"""Long-running agentic workflow system.

Provides orchestrator/worker architecture for multi-phase projects
that span hours or days. Includes event sourcing, checkpointing,
and Obsidian integration.

Quick start::

    from roshni.agent.workflow import ProjectStore, Orchestrator, create_workflow_tools

    store = ProjectStore("~/.weeklies-data/projects")
    # ... configure orchestrator ...
    tools = create_workflow_tools(store, orchestrator)
"""

from .backend import FileWorkflowBackend, PhaseResult, WorkflowBackend
from .events import (
    ALL_EVENT_TYPES,
    BUDGET_EXHAUSTED,
    BUDGET_RECORDED_CALL,
    BUDGET_WARNING,
    CONFLICT_DETECTED,
    CONFLICT_RECONCILED,
    PHASE_COMPLETED,
    PHASE_FAILED,
    PHASE_STARTED,
    PLAN_WRITTEN,
    PROJECT_ADVANCED,
    PROJECT_CREATED,
    PROJECT_STEERED,
    PROJECT_TRANSITIONED,
    TASK_COMPLETED,
    TASK_DISPATCHED,
    TASK_FAILED,
    TERMINAL_CONDITION_EVALUATED,
)
from .models import (
    TERMINAL_STATUSES,
    VALID_TRANSITIONS,
    Artifact,
    ArtifactSpec,
    Budget,
    JournalEntry,
    Phase,
    PhaseEntry,
    PhaseStatus,
    Project,
    ProjectStatus,
    TaskSpec,
    TerminalCondition,
    WorkflowEvent,
    compute_plan_hash,
    project_from_dict,
    project_to_dict,
    validate_transition,
)
from .orchestrator import Orchestrator
from .store import ProjectStore
from .tools import create_workflow_tools
from .worker import ToolPolicyViolation, WorkerPool, WorkerResult

__all__ = [
    "ALL_EVENT_TYPES",
    "BUDGET_EXHAUSTED",
    "BUDGET_RECORDED_CALL",
    "BUDGET_WARNING",
    "CONFLICT_DETECTED",
    "CONFLICT_RECONCILED",
    "PHASE_COMPLETED",
    "PHASE_FAILED",
    "PHASE_STARTED",
    "PLAN_WRITTEN",
    "PROJECT_ADVANCED",
    "PROJECT_CREATED",
    "PROJECT_STEERED",
    "PROJECT_TRANSITIONED",
    "TASK_COMPLETED",
    "TASK_DISPATCHED",
    "TASK_FAILED",
    "TERMINAL_CONDITION_EVALUATED",
    "TERMINAL_STATUSES",
    "VALID_TRANSITIONS",
    "Artifact",
    "ArtifactSpec",
    "Budget",
    "FileWorkflowBackend",
    "JournalEntry",
    "Orchestrator",
    "Phase",
    "PhaseEntry",
    "PhaseResult",
    "PhaseStatus",
    "Project",
    "ProjectStatus",
    "ProjectStore",
    "TaskSpec",
    "TerminalCondition",
    "ToolPolicyViolation",
    "WorkerPool",
    "WorkerResult",
    "WorkflowBackend",
    "WorkflowEvent",
    "compute_plan_hash",
    "create_workflow_tools",
    "project_from_dict",
    "project_to_dict",
    "validate_transition",
]
