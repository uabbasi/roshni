"""Task store — markdown-file-backed task management with YAML frontmatter.

Each task is a markdown file with YAML frontmatter containing metadata.
The TaskStore handles CRUD, status transitions, dependency resolution,
and archiving of completed tasks.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

# Use stdlib yaml — available via PyYAML (a core dep)
import yaml


class TaskStatus(StrEnum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"
    CANCELLED = "cancelled"


class TaskPriority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


# Valid transitions: from_status -> set of allowed to_statuses
_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.OPEN: {TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED, TaskStatus.CANCELLED},
    TaskStatus.IN_PROGRESS: {TaskStatus.DONE, TaskStatus.BLOCKED, TaskStatus.OPEN, TaskStatus.CANCELLED},
    TaskStatus.BLOCKED: {TaskStatus.OPEN, TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED},
    TaskStatus.DONE: {TaskStatus.OPEN},
    TaskStatus.CANCELLED: {TaskStatus.OPEN},
}


@dataclass
class Task:
    id: str
    title: str
    status: TaskStatus = TaskStatus.OPEN
    priority: TaskPriority = TaskPriority.MEDIUM
    project: str = ""
    depends_on: list[str] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)
    created: datetime = field(default_factory=datetime.now)
    updated: datetime = field(default_factory=datetime.now)
    due: datetime | None = None
    tags: list[str] = field(default_factory=list)
    body: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?(.*)", re.DOTALL)
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(title: str) -> str:
    """Convert title to a filename-safe slug."""
    slug = _SLUG_RE.sub("-", title.lower()).strip("-")
    return slug[:60] or "task"


def _parse_datetime(value: str | datetime | None) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _format_datetime(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat(timespec="seconds")


def _parse_task(path: Path) -> Task:
    """Read a .md file and parse YAML frontmatter into a Task."""
    text = path.read_text(encoding="utf-8")
    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise ValueError(f"No YAML frontmatter in {path}")
    front = yaml.safe_load(m.group(1)) or {}
    body = m.group(2).strip()

    return Task(
        id=front.get("id", path.stem),
        title=front.get("title", ""),
        status=TaskStatus(front.get("status", "open")),
        priority=TaskPriority(front.get("priority", "medium")),
        project=front.get("project", "") or "",
        depends_on=front.get("depends_on") or [],
        blocks=front.get("blocks") or [],
        created=_parse_datetime(front.get("created")) or datetime.now(),
        updated=_parse_datetime(front.get("updated")) or datetime.now(),
        due=_parse_datetime(front.get("due")),
        tags=front.get("tags") or [],
        body=body,
    )


def _write_task(task: Task, path: Path) -> None:
    """Serialize a Task to a .md file with YAML frontmatter."""
    front = {
        "id": task.id,
        "title": task.title,
        "status": task.status.value,
        "priority": task.priority.value,
        "project": task.project,
        "depends_on": task.depends_on,
        "blocks": task.blocks,
        "created": _format_datetime(task.created),
        "updated": _format_datetime(task.updated),
        "due": _format_datetime(task.due),
        "tags": task.tags,
    }
    lines = ["---", yaml.dump(front, default_flow_style=False, sort_keys=False).rstrip(), "---"]
    if task.body:
        lines.append("")
        lines.append(task.body)
    lines.append("")  # trailing newline
    path.write_text("\n".join(lines), encoding="utf-8")


def _next_id(tasks_dir: Path) -> str:
    """Generate the next task ID: t-YYYYMMDD-NNN."""
    today = datetime.now().strftime("%Y%m%d")
    prefix = f"t-{today}-"
    max_seq = 0
    for p in tasks_dir.glob("*.md"):
        if p.name.startswith("_"):
            continue
        try:
            task = _parse_task(p)
        except Exception:
            continue
        if task.id.startswith(prefix):
            try:
                seq = int(task.id[len(prefix) :])
                max_seq = max(max_seq, seq)
            except ValueError:
                pass
    return f"{prefix}{max_seq + 1:03d}"


# ---------------------------------------------------------------------------
# TaskStore
# ---------------------------------------------------------------------------


class TaskStore:
    """File-backed task store using markdown files with YAML frontmatter."""

    def __init__(self, tasks_dir: str | Path) -> None:
        self.tasks_dir = Path(tasks_dir)
        self._archive_dir = self.tasks_dir / "_archive"

    def _ensure_dir(self) -> None:
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    def _task_path(self, task_id: str) -> Path | None:
        """Find the file for a given task ID."""
        for p in self.tasks_dir.glob("*.md"):
            if p.name.startswith("_"):
                continue
            try:
                t = _parse_task(p)
                if t.id == task_id:
                    return p
            except Exception:
                continue
        return None

    # -- CRUD ---------------------------------------------------------------

    def create(
        self,
        title: str,
        *,
        description: str = "",
        priority: str = "medium",
        project: str = "",
        tags: list[str] | None = None,
        due: str = "",
    ) -> Task:
        """Create a new task and write it to disk."""
        self._ensure_dir()
        task_id = _next_id(self.tasks_dir)
        now = datetime.now()
        task = Task(
            id=task_id,
            title=title,
            status=TaskStatus.OPEN,
            priority=TaskPriority(priority),
            project=project,
            tags=tags or [],
            due=_parse_datetime(due) if due else None,
            created=now,
            updated=now,
            body=description,
        )
        slug = _slugify(title)
        filename = f"{task_id}-{slug}.md"
        _write_task(task, self.tasks_dir / filename)
        return task

    def get(self, task_id: str) -> Task | None:
        """Get a task by ID. Returns None if not found."""
        path = self._task_path(task_id)
        if path is None:
            return None
        return _parse_task(path)

    def update(self, task_id: str, **changes: object) -> Task:
        """Update task fields and rewrite the file. Returns updated task."""
        path = self._task_path(task_id)
        if path is None:
            raise ValueError(f"Task not found: {task_id}")
        task = _parse_task(path)

        for key, value in changes.items():
            if key == "status":
                task.status = TaskStatus(str(value))
            elif key == "priority":
                task.priority = TaskPriority(str(value))
            elif key == "due":
                task.due = _parse_datetime(value) if value else None  # type: ignore[arg-type]
            elif key == "body" or key == "description":
                task.body = str(value)
            elif hasattr(task, key):
                setattr(task, key, value)
            else:
                raise ValueError(f"Unknown task field: {key}")

        task.updated = datetime.now()
        _write_task(task, path)
        return task

    def delete(self, task_id: str) -> bool:
        """Delete a task file. Returns True if deleted."""
        path = self._task_path(task_id)
        if path is None:
            return False
        path.unlink()
        return True

    # -- Query --------------------------------------------------------------

    def list_tasks(
        self,
        *,
        status: str = "",
        project: str = "",
        tag: str = "",
        limit: int = 20,
    ) -> list[Task]:
        """List tasks with optional filters."""
        if not self.tasks_dir.is_dir():
            return []
        tasks: list[Task] = []
        for p in sorted(self.tasks_dir.glob("*.md")):
            if p.name.startswith("_"):
                continue
            try:
                t = _parse_task(p)
            except Exception:
                continue
            if status and t.status.value != status:
                continue
            if project and t.project != project:
                continue
            if tag and tag not in t.tags:
                continue
            tasks.append(t)

        # Sort: urgent first, then by created descending
        priority_order = {TaskPriority.URGENT: 0, TaskPriority.HIGH: 1, TaskPriority.MEDIUM: 2, TaskPriority.LOW: 3}
        tasks.sort(key=lambda t: (priority_order.get(t.priority, 2), t.created), reverse=False)
        # reverse=False because lower priority_order number = higher priority, and earlier created comes first
        # Actually we want urgent first (lowest number) then newest first within same priority
        tasks.sort(key=lambda t: (priority_order.get(t.priority, 2), -t.created.timestamp()))

        return tasks[:limit]

    def search(self, query: str, limit: int = 10) -> list[Task]:
        """Search tasks by keyword in title and body."""
        if not self.tasks_dir.is_dir():
            return []
        query_lower = query.lower()
        results: list[Task] = []
        for p in sorted(self.tasks_dir.glob("*.md")):
            if p.name.startswith("_"):
                continue
            try:
                t = _parse_task(p)
            except Exception:
                continue
            if (
                query_lower in t.title.lower()
                or query_lower in t.body.lower()
                or query_lower in " ".join(t.tags).lower()
            ):
                results.append(t)
            if len(results) >= limit:
                break
        return results

    # -- Status machine -----------------------------------------------------

    def transition(self, task_id: str, new_status: str) -> Task:
        """Transition a task to a new status, validating the state machine."""
        path = self._task_path(task_id)
        if path is None:
            raise ValueError(f"Task not found: {task_id}")
        task = _parse_task(path)
        target = TaskStatus(new_status)

        allowed = _TRANSITIONS.get(task.status, set())
        if target not in allowed:
            raise ValueError(
                f"Invalid transition: {task.status.value} -> {target.value}. "
                f"Allowed: {', '.join(s.value for s in sorted(allowed, key=lambda s: s.value))}"
            )

        task.status = target
        task.updated = datetime.now()
        _write_task(task, path)
        return task

    # -- Dependencies -------------------------------------------------------

    def get_actionable(self) -> list[Task]:
        """Return open tasks whose dependencies are all resolved (done/cancelled)."""
        all_tasks = self.list_tasks(limit=1000)
        # Build a lookup of task status by ID
        status_by_id: dict[str, TaskStatus] = {t.id: t.status for t in all_tasks}

        resolved = {TaskStatus.DONE, TaskStatus.CANCELLED}
        actionable: list[Task] = []
        for t in all_tasks:
            if t.status != TaskStatus.OPEN:
                continue
            # Check if all dependencies are resolved
            deps_resolved = all(status_by_id.get(dep) in resolved for dep in t.depends_on)
            if deps_resolved:
                actionable.append(t)
        return actionable

    # -- Memory decay -------------------------------------------------------

    def summarize_completed(self, older_than_days: int = 30) -> str:
        """Archive done tasks older than N days. Returns summary text."""
        if not self.tasks_dir.is_dir():
            return "No tasks directory."
        self._archive_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        archived: list[str] = []
        for p in list(self.tasks_dir.glob("*.md")):
            if p.name.startswith("_"):
                continue
            try:
                t = _parse_task(p)
            except Exception:
                continue
            if t.status != TaskStatus.DONE:
                continue
            age_days = (now - t.updated).days
            if age_days >= older_than_days:
                shutil.move(str(p), str(self._archive_dir / p.name))
                archived.append(f"- {t.id}: {t.title} (done {age_days}d ago)")

        if not archived:
            return "No completed tasks old enough to archive."
        return f"Archived {len(archived)} task(s):\n" + "\n".join(archived)

    # -- Index --------------------------------------------------------------

    def rebuild_index(self) -> None:
        """Regenerate _index.md with status counts and task list grouped by status."""
        self._ensure_dir()
        all_tasks = self.list_tasks(limit=10000)

        # Group by status
        by_status: dict[str, list[Task]] = {}
        for t in all_tasks:
            by_status.setdefault(t.status.value, []).append(t)

        lines = [f"# Task Index (updated {datetime.now().strftime('%Y-%m-%d %H:%M')})", ""]

        # Counts
        lines.append("## Summary")
        for s in TaskStatus:
            count = len(by_status.get(s.value, []))
            lines.append(f"- **{s.value}**: {count}")
        lines.append(f"- **total**: {len(all_tasks)}")
        lines.append("")

        # Grouped listings
        display_order = [
            TaskStatus.OPEN,
            TaskStatus.IN_PROGRESS,
            TaskStatus.BLOCKED,
            TaskStatus.DONE,
            TaskStatus.CANCELLED,
        ]
        for s in display_order:
            tasks = by_status.get(s.value, [])
            if not tasks:
                continue
            lines.append(f"## {s.value.replace('_', ' ').title()}")
            for t in tasks:
                due_str = f" (due: {_format_datetime(t.due)})" if t.due else ""
                lines.append(f"- [{t.priority.value}] **{t.id}**: {t.title}{due_str}")
            lines.append("")

        index_path = self.tasks_dir / "_index.md"
        index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
