"""WorkflowBackend protocol and file-based implementation.

The backend handles durability: event logging, checkpointing, resumption,
and Obsidian rendering. The event log (events.ndjson) is the recovery
source of truth; checkpoint.json is a derived snapshot.

Crash semantics:
1. Every state change is first appended to events.ndjson (fsync'd)
2. checkpoint() writes a full snapshot via atomic temp-file + rename
3. On resume(), replay events with seq > checkpoint.last_event_seq
4. If checkpoint is missing/corrupt, rebuild entirely from events
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from loguru import logger

from .events import CONFLICT_DETECTED
from .models import (
    Phase,
    PhaseStatus,
    Project,
    ProjectStatus,
    WorkflowEvent,
    compute_plan_hash,
    project_from_dict,
    project_to_dict,
)


@dataclass
class PhaseResult:
    """Result of executing a single phase."""

    phase_id: str
    success: bool
    error: str = ""
    artifacts: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = []


@runtime_checkable
class WorkflowBackend(Protocol):
    """Pluggable durability backend for workflow projects.

    File-based v1; Temporal upgrade path for v2.
    """

    async def execute_phase(self, project: Project, phase: Phase) -> PhaseResult: ...

    async def checkpoint(self, project: Project) -> None: ...

    async def resume(self, project_id: str) -> Project: ...

    async def record_event(self, project_id: str, event: WorkflowEvent) -> None: ...

    async def record_llm_call(self, project_id: str, call_record: dict) -> None: ...


# ---------------------------------------------------------------------------
# Obsidian rendering
# ---------------------------------------------------------------------------

_PLAN_OVERRIDE_START = "<!-- ROSHNI:PLAN-OVERRIDE-START -->"
_PLAN_OVERRIDE_END = "<!-- ROSHNI:PLAN-OVERRIDE-END -->"


def render_obsidian(project: Project, projects_dir: str) -> str:
    """Render a project to Obsidian markdown with YAML frontmatter."""
    now = datetime.now().isoformat(timespec="seconds")
    lines = [
        "---",
        f"id: {project.id}",
        f"status: {project.status.value}",
        f"plan_hash: {project.plan_hash}",
        f"last_orchestrator_update_at: {now}",
        f"tags: [{', '.join(project.tags)}]",
        f"created: {project.created}",
        f"updated: {project.updated}",
        "---",
        "",
        f"# {project.goal}",
        "",
        f"**Status:** {project.status.value}",
        f"**Budget:** ${project.budget.cost_used_usd:.2f} / ${project.budget.max_cost_usd:.2f} USD"
        f" | {project.budget.llm_calls_used} / {project.budget.max_llm_calls} calls",
        "",
    ]

    # Phases
    if project.phases:
        lines.append("## Phases")
        lines.append("")
        for phase in project.phases:
            status_icon = {
                PhaseStatus.PENDING: " ",
                PhaseStatus.ACTIVE: "~",
                PhaseStatus.COMPLETED: "x",
                PhaseStatus.FAILED: "!",
                PhaseStatus.SKIPPED: "-",
            }.get(phase.status, " ")
            lines.append(f"### [{status_icon}] {phase.name}")
            if phase.description:
                lines.append(f"{phase.description}")
            lines.append("")

            if phase.entry_criteria:
                lines.append("**Entry criteria:**")
                for ec in phase.entry_criteria:
                    check = "x" if ec.met else " "
                    lines.append(f"- [{check}] {ec.description}")
                lines.append("")

            if phase.tasks:
                lines.append("**Tasks:**")
                for task in phase.tasks:
                    lines.append(f"- `{task.id}`: {task.description}")
                lines.append("")

            if phase.exit_criteria:
                lines.append("**Exit criteria:**")
                for ec in phase.exit_criteria:
                    check = "x" if ec.met else " "
                    lines.append(f"- [{check}] {ec.description}")
                lines.append("")

    # Terminal conditions
    if project.terminal_conditions:
        lines.append("## Terminal Conditions")
        lines.append("")
        for tc in project.terminal_conditions:
            check = "x" if tc.met else " "
            lines.append(f"- [{check}] {tc.description} ({tc.type})")
        lines.append("")

    # Recent journal (last 10)
    if project.journal:
        lines.append("## Journal (recent)")
        lines.append("")
        for entry in project.journal[-10:]:
            lines.append(f"- **{entry.timestamp}** [{entry.actor}] {entry.action}: {entry.content}")
        lines.append("")

    # Artifacts
    if project.artifacts:
        lines.append("## Artifacts")
        lines.append("")
        for artifact in project.artifacts:
            lines.append(f"- [{artifact.name}]({artifact.path}) ({artifact.mime_type})")
        lines.append("")

    return "\n".join(lines)


def parse_obsidian_frontmatter(text: str) -> dict[str, Any]:
    """Extract YAML frontmatter from an Obsidian markdown file."""
    import yaml

    m = re.match(r"\A---\n(.*?)\n---", text, re.DOTALL)
    if not m:
        return {}
    try:
        return yaml.safe_load(m.group(1)) or {}
    except Exception:
        return {}


def check_obsidian_conflict(
    obsidian_path: Path,
    checkpoint_plan_hash: str,
    last_update_at: str | None,
) -> str | None:
    """Check if an Obsidian file was edited externally.

    Returns None if no conflict, or a reason string if conflict detected.
    Uses mtime + plan_hash to reduce false positives:
    - mtime unchanged -> no conflict
    - mtime changed but plan_hash same -> cosmetic edit, no conflict
    - mtime changed and plan_hash different -> real conflict
    """
    if not obsidian_path.exists():
        return None

    if not last_update_at:
        return None

    # Check mtime
    file_mtime = datetime.fromtimestamp(obsidian_path.stat().st_mtime)
    try:
        last_update = datetime.fromisoformat(last_update_at)
    except (ValueError, TypeError):
        return None

    # Allow 2-second tolerance for filesystem timestamp granularity
    if (file_mtime - last_update).total_seconds() <= 2.0:
        return None

    # mtime changed — check if plan_hash changed too
    try:
        text = obsidian_path.read_text(encoding="utf-8")
        fm = parse_obsidian_frontmatter(text)
        obs_plan_hash = fm.get("plan_hash", "")
        obs_status = fm.get("status", "")

        if obs_plan_hash and obs_plan_hash != checkpoint_plan_hash:
            return f"Plan hash changed: checkpoint={checkpoint_plan_hash}, obsidian={obs_plan_hash}"

        # Check for status override
        if obs_status and obs_status != "":
            # We'll check this in the reconcile step, not flag as conflict
            pass

        # plan_hash unchanged -> cosmetic edit, no conflict
        return None
    except Exception as e:
        return f"Could not read Obsidian file: {e}"


# ---------------------------------------------------------------------------
# FileWorkflowBackend
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?(.*)", re.DOTALL)


def _slugify(text: str) -> str:
    """Convert text to a filename-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60] or "project"


class FileWorkflowBackend:
    """File-based workflow backend using JSON + NDJSON.

    Layout per project::

        {base_dir}/{project_id}/
            plan.json           # Canonical plan
            checkpoint.json     # Full snapshot (derived)
            events.ndjson       # Append-only event log
            worker-logs/        # Per-worker JSONL
            llm-calls/          # LLM request/response records
            artifacts/          # Named outputs
    """

    def __init__(self, base_dir: str | Path, obsidian_projects_dir: str | Path = "") -> None:
        self._base = Path(base_dir)
        self._obsidian_dir = Path(obsidian_projects_dir) if obsidian_projects_dir else None
        self._seq_counters: dict[str, int] = {}  # project_id -> next seq

    def _project_dir(self, project_id: str) -> Path:
        return self._base / project_id

    def _ensure_dirs(self, project_id: str) -> Path:
        """Create workspace directories for a project."""
        project_dir = self._project_dir(project_id)
        for sub in ("worker-logs", "llm-calls", "artifacts"):
            (project_dir / sub).mkdir(parents=True, exist_ok=True)
        return project_dir

    def _next_seq(self, project_id: str) -> int:
        """Get the next monotonic sequence number for a project."""
        if project_id not in self._seq_counters:
            # Initialize from event log
            events_path = self._project_dir(project_id) / "events.ndjson"
            max_seq = 0
            if events_path.exists():
                for line in events_path.read_text().splitlines():
                    if line.strip():
                        try:
                            evt = json.loads(line)
                            max_seq = max(max_seq, evt.get("seq", 0))
                        except json.JSONDecodeError:
                            continue
            self._seq_counters[project_id] = max_seq
        self._seq_counters[project_id] += 1
        return self._seq_counters[project_id]

    def _format_event_id(self, seq: int) -> str:
        return f"evt-{seq:06d}"

    def create_event(self, project_id: str, event_type: str, actor: str, payload: dict | None = None) -> WorkflowEvent:
        """Create a new WorkflowEvent with monotonic seq."""
        seq = self._next_seq(project_id)
        return WorkflowEvent(
            event_id=self._format_event_id(seq),
            seq=seq,
            type=event_type,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            actor=actor,
            payload=payload or {},
        )

    async def record_event(self, project_id: str, event: WorkflowEvent) -> None:
        """Append event to NDJSON log with fsync."""
        project_dir = self._ensure_dirs(project_id)
        events_path = project_dir / "events.ndjson"
        event_dict = {
            "event_id": event.event_id,
            "seq": event.seq,
            "type": event.type,
            "timestamp": event.timestamp,
            "actor": event.actor,
            "payload": event.payload,
        }
        line = json.dumps(event_dict, ensure_ascii=False) + "\n"
        fd = os.open(str(events_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)

    async def record_llm_call(self, project_id: str, call_record: dict) -> None:
        """Write an LLM call record to the llm-calls directory."""
        project_dir = self._ensure_dirs(project_id)
        llm_dir = project_dir / "llm-calls"
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        call_id = call_record.get("id", "unknown")[:8]
        filename = f"{timestamp}-call-{call_id}.json"
        path = llm_dir / filename
        path.write_text(json.dumps(call_record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    async def save_plan(self, project: Project) -> None:
        """Write the canonical plan.json."""
        project_dir = self._ensure_dirs(project.id)
        plan_path = project_dir / "plan.json"

        plan_data = {
            "schema_version": project.schema_version,
            "goal": project.goal,
            "phases": [
                {
                    "id": p.id,
                    "name": p.name,
                    "description": p.description,
                    "entry_criteria": [{"description": e.description, "met": e.met} for e in p.entry_criteria],
                    "exit_criteria": [{"description": e.description, "met": e.met} for e in p.exit_criteria],
                    "tasks": [
                        {
                            "id": t.id,
                            "description": t.description,
                            "allowed_tools": t.allowed_tools,
                            "inputs": t.inputs,
                            "expected_outputs": t.expected_outputs,
                            "depends_on": t.depends_on,
                            "max_attempts": t.max_attempts,
                        }
                        for t in p.tasks
                    ],
                }
                for p in project.phases
            ],
            "terminal_conditions": [
                {
                    "description": tc.description,
                    "type": tc.type,
                    "params": tc.params,
                }
                for tc in project.terminal_conditions
            ],
        }
        self._atomic_write(plan_path, json.dumps(plan_data, indent=2, ensure_ascii=False))

        # Update plan_hash
        project.plan_hash = compute_plan_hash(project)

    async def checkpoint(self, project: Project) -> None:
        """Write a full project snapshot (atomic write)."""
        project_dir = self._ensure_dirs(project.id)
        checkpoint_path = project_dir / "checkpoint.json"

        # Update wall time if project has a start time
        if project.started_at:
            try:
                started = datetime.fromisoformat(project.started_at)
                project.budget.update_wall_time(started)
            except (ValueError, TypeError):
                pass

        project.updated = datetime.now().isoformat(timespec="seconds")
        project.last_orchestrator_update_at = project.updated

        data = project_to_dict(project)
        self._atomic_write(checkpoint_path, json.dumps(data, indent=2, ensure_ascii=False, default=str))

        # Re-render Obsidian file ONLY for workflow-managed projects (those with phases).
        # This prevents overwriting hand-crafted Obsidian project docs that exist
        # independently of the workflow system.
        if self._obsidian_dir and project.obsidian_file and project.phases:
            try:
                obs_path = self._obsidian_dir / project.obsidian_file
                obs_path.parent.mkdir(parents=True, exist_ok=True)
                content = render_obsidian(project, str(self._obsidian_dir))
                self._atomic_write(obs_path, content)
            except Exception as e:
                logger.warning(f"Failed to render Obsidian file: {e}")

    async def resume(self, project_id: str) -> Project:
        """Resume a project from checkpoint + event replay.

        1. Load checkpoint.json as base state
        2. Replay events with seq > checkpoint.last_event_seq
        3. Check Obsidian for external edits
        """
        project_dir = self._project_dir(project_id)
        checkpoint_path = project_dir / "checkpoint.json"
        events_path = project_dir / "events.ndjson"

        project: Project | None = None
        base_seq = 0

        # Try loading checkpoint
        if checkpoint_path.exists():
            try:
                data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                project = project_from_dict(data)
                base_seq = project.last_event_seq
                logger.info(f"Loaded checkpoint for {project_id} at seq={base_seq}")
            except Exception as e:
                logger.warning(f"Checkpoint corrupt for {project_id}, will rebuild from events: {e}")
                project = None
                base_seq = 0

        # Replay events
        if events_path.exists():
            events = self._load_events(events_path)
            # Sort by seq for deterministic replay
            events.sort(key=lambda e: e["seq"])

            if project is None:
                # Full rebuild from events
                logger.info(f"Rebuilding {project_id} from {len(events)} events")
                project = self._rebuild_from_events(project_id, events)
            else:
                # Incremental replay
                pending = [e for e in events if e["seq"] > base_seq]
                if pending:
                    logger.info(f"Replaying {len(pending)} events for {project_id} (seq > {base_seq})")
                    self._replay_events(project, pending)

        if project is None:
            raise ValueError(f"No checkpoint or events found for project {project_id}")

        # Check Obsidian conflict
        if self._obsidian_dir and project.obsidian_file:
            obs_path = self._obsidian_dir / project.obsidian_file
            conflict = check_obsidian_conflict(obs_path, project.plan_hash, project.last_orchestrator_update_at)
            if conflict:
                logger.warning(f"Obsidian conflict for {project_id}: {conflict}")
                if project.status not in {ProjectStatus.PAUSED, ProjectStatus.DONE, ProjectStatus.CANCELLED}:
                    project.status = ProjectStatus.PAUSED
                    project.journal.append(
                        _journal("system", "conflict", f"Human edits detected; reconcile required: {conflict}")
                    )
                    # Record conflict event
                    evt = self.create_event(project_id, CONFLICT_DETECTED, "system", {"reason": conflict})
                    await self.record_event(project_id, evt)
                    project.last_event_seq = evt.seq

        # Sync seq counter
        self._seq_counters[project_id] = project.last_event_seq

        return project

    async def execute_phase(self, project: Project, phase: Phase) -> PhaseResult:
        """Placeholder — actual execution is done by the orchestrator + worker pool."""
        raise NotImplementedError("Phase execution is handled by the Orchestrator")

    # --- Obsidian reconciliation ---

    async def reconcile_accept_obsidian(self, project: Project) -> None:
        """Accept structured human edits from Obsidian.

        Only parses: frontmatter (tags, status), and plan override blocks
        between markers. Does NOT parse freeform markdown.
        """
        if not self._obsidian_dir or not project.obsidian_file:
            return

        obs_path = self._obsidian_dir / project.obsidian_file
        if not obs_path.exists():
            return

        text = obs_path.read_text(encoding="utf-8")
        fm = parse_obsidian_frontmatter(text)

        # Accept frontmatter changes
        if fm.get("tags"):
            project.tags = fm["tags"] if isinstance(fm["tags"], list) else [fm["tags"]]
        if fm.get("status"):
            try:
                project.status = ProjectStatus(fm["status"])
            except ValueError:
                pass

        # Check for plan override block
        if _PLAN_OVERRIDE_START in text and _PLAN_OVERRIDE_END in text:
            start_idx = text.index(_PLAN_OVERRIDE_START) + len(_PLAN_OVERRIDE_START)
            end_idx = text.index(_PLAN_OVERRIDE_END)
            override_text = text[start_idx:end_idx].strip()
            if override_text:
                project.journal.append(
                    _journal("user", "steering", f"Plan override from Obsidian: {override_text[:200]}")
                )

    async def reconcile_override_obsidian(self, project: Project) -> None:
        """Re-render Obsidian from checkpoint (safe, always works)."""
        if self._obsidian_dir and project.obsidian_file:
            obs_path = self._obsidian_dir / project.obsidian_file
            obs_path.parent.mkdir(parents=True, exist_ok=True)
            content = render_obsidian(project, str(self._obsidian_dir))
            self._atomic_write(obs_path, content)

    # --- Internal helpers ---

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        """Write atomically via temp file + rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            os.write(fd, content.encode("utf-8"))
            os.fsync(fd)
            os.close(fd)
            os.rename(tmp_path, str(path))
        except Exception:
            os.close(fd) if not os.get_inheritable(fd) else None
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    @staticmethod
    def _load_events(events_path: Path) -> list[dict]:
        """Load all events from NDJSON file."""
        events = []
        for line in events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed event line: {line[:80]}")
        return events

    def _rebuild_from_events(self, project_id: str, events: list[dict]) -> Project | None:
        """Rebuild a Project entirely from events (seq=0 -> end)."""
        if not events:
            return None

        # Find creation event
        created_event = None
        for e in events:
            if e["type"] == "project.created":
                created_event = e
                break

        if not created_event:
            logger.error(f"No project.created event for {project_id}")
            return None

        payload = created_event.get("payload", {})
        project = Project(
            id=project_id,
            goal=payload.get("goal", ""),
            created=created_event.get("timestamp", datetime.now().isoformat()),
        )

        self._replay_events(project, events)
        return project

    @staticmethod
    def _replay_events(project: Project, events: list[dict]) -> None:
        """Apply events to a project in seq order."""
        for evt in sorted(events, key=lambda e: e["seq"]):
            evt_type = evt["type"]
            payload = evt.get("payload", {})
            seq = evt["seq"]

            if evt_type == "project.transitioned":
                try:
                    project.status = ProjectStatus(payload.get("to", project.status))
                except ValueError:
                    pass

            elif evt_type == "phase.started":
                phase_id = payload.get("phase_id")
                for p in project.phases:
                    if p.id == phase_id:
                        p.status = PhaseStatus.ACTIVE
                        p.started = evt.get("timestamp")
                        break

            elif evt_type == "phase.completed":
                phase_id = payload.get("phase_id")
                for p in project.phases:
                    if p.id == phase_id:
                        p.status = PhaseStatus.COMPLETED
                        p.completed = evt.get("timestamp")
                        break

            elif evt_type == "phase.failed":
                phase_id = payload.get("phase_id")
                for p in project.phases:
                    if p.id == phase_id:
                        p.status = PhaseStatus.FAILED
                        break

            elif evt_type == "budget.recorded_call":
                cost = payload.get("cost_usd", 0.0)
                project.budget.cost_used_usd += cost
                project.budget.llm_calls_used += 1

            elif evt_type == "plan.written":
                project.plan_hash = payload.get("plan_hash", "")

            project.last_event_seq = seq

    def get_workspace_path(self, project_id: str) -> Path:
        """Get the workspace directory for a project."""
        return self._project_dir(project_id)

    def get_artifacts_path(self, project_id: str) -> Path:
        """Get the artifacts directory for a project."""
        return self._project_dir(project_id) / "artifacts"

    def list_project_ids(self) -> list[str]:
        """List all project IDs with checkpoint or events."""
        if not self._base.exists():
            return []
        ids = []
        for d in sorted(self._base.iterdir()):
            if d.is_dir() and (d / "checkpoint.json").exists():
                ids.append(d.name)
            elif d.is_dir() and (d / "events.ndjson").exists():
                ids.append(d.name)
        return ids


def _journal(actor: str, action: str, content: str) -> dict:
    """Create a JournalEntry-compatible dict."""
    from .models import JournalEntry

    return JournalEntry(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        actor=actor,
        action=action,
        content=content,
    )
