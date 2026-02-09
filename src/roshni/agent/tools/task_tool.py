"""Task management tools — CRUD, status transitions, and project planning."""

from __future__ import annotations

import json
from pathlib import Path

from roshni.agent.permissions import PermissionTier, filter_tools_by_tier
from roshni.agent.task_store import TaskStore
from roshni.agent.tools import ToolDefinition


def _fmt_task(t) -> str:
    """Format a Task for tool output."""
    lines = [
        f"**{t.id}** — {t.title}",
        f"Status: {t.status.value} | Priority: {t.priority.value}",
    ]
    if t.project:
        lines.append(f"Project: {t.project}")
    if t.tags:
        lines.append(f"Tags: {', '.join(t.tags)}")
    if t.due:
        lines.append(f"Due: {t.due.isoformat(timespec='seconds')}")
    if t.depends_on:
        lines.append(f"Depends on: {', '.join(t.depends_on)}")
    if t.body:
        lines.append(f"\n{t.body}")
    return "\n".join(lines)


def create_task_tools(
    tasks_dir: str,
    tier: PermissionTier = PermissionTier.INTERACT,
    projects_dir: str = "",
) -> list[ToolDefinition]:
    """Create task management tools filtered by permission tier."""
    store = TaskStore(tasks_dir)

    # -- OBSERVE (read) -----------------------------------------------------

    def list_tasks(status: str = "", project: str = "", tag: str = "", limit: int = 20) -> str:
        tasks = store.list_tasks(status=status, project=project, tag=tag, limit=limit)
        if not tasks:
            return "No tasks found."
        return "\n\n---\n\n".join(_fmt_task(t) for t in tasks)

    def get_task(task_id: str) -> str:
        t = store.get(task_id)
        if t is None:
            return f"Task not found: {task_id}"
        return _fmt_task(t)

    def search_tasks(query: str, limit: int = 10) -> str:
        tasks = store.search(query, limit=limit)
        if not tasks:
            return f"No tasks matching '{query}'."
        return "\n\n---\n\n".join(_fmt_task(t) for t in tasks)

    # -- INTERACT (write) ---------------------------------------------------

    def create_task(
        title: str,
        description: str = "",
        priority: str = "medium",
        project: str = "",
        tags: str = "",
        due: str = "",
    ) -> str:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
        t = store.create(
            title=title, description=description, priority=priority, project=project, tags=tag_list, due=due
        )
        return f"Created: {t.id} — {t.title}"

    def update_task(
        task_id: str,
        title: str = "",
        description: str = "",
        priority: str = "",
        project: str = "",
        tags: str = "",
        due: str = "",
    ) -> str:
        changes: dict[str, object] = {}
        if title:
            changes["title"] = title
        if description:
            changes["description"] = description
        if priority:
            changes["priority"] = priority
        if project:
            changes["project"] = project
        if tags:
            changes["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
        if due:
            changes["due"] = due
        if not changes:
            return "No changes specified."
        try:
            t = store.update(task_id, **changes)
            return f"Updated: {t.id} — {t.title}"
        except ValueError as e:
            return f"Error: {e}"

    def transition_task(task_id: str, new_status: str) -> str:
        try:
            t = store.transition(task_id, new_status)
            return f"Transitioned {t.id} to {t.status.value}."
        except ValueError as e:
            return f"Error: {e}"

    def plan_project(project_name: str, tasks_json: str) -> str:
        try:
            task_defs = json.loads(tasks_json)
        except json.JSONDecodeError as e:
            return f"Error parsing tasks_json: {e}"

        if not isinstance(task_defs, list):
            return "Error: tasks_json must be a JSON array of task objects."

        # Write project file if projects_dir provided
        p_dir = Path(projects_dir) if projects_dir else Path(tasks_dir).parent / "projects"
        p_dir.mkdir(parents=True, exist_ok=True)
        slug = project_name.lower().replace(" ", "-")[:60]
        project_file = p_dir / f"{slug}.md"
        project_file.write_text(
            f"# {project_name}\n\nCreated with plan_project.\n\n## Tasks\n\n"
            + "\n".join(f"- {td.get('title', 'untitled')}" for td in task_defs)
            + "\n",
            encoding="utf-8",
        )

        # Create tasks and track their IDs for dependency mapping
        placeholder_to_id: dict[str, str] = {}
        created: list[str] = []
        for i, td in enumerate(task_defs):
            t = store.create(
                title=td.get("title", "Untitled"),
                description=td.get("description", ""),
                priority=td.get("priority", "medium"),
                project=project_name,
            )
            # Allow tasks_json to reference tasks by index placeholder like "task-0", "task-1"
            placeholder_to_id[f"task-{i}"] = t.id
            created.append(t.id)

        # Wire up depends_on relationships
        for i, td in enumerate(task_defs):
            raw_deps = td.get("depends_on") or []
            resolved_deps = []
            for dep in raw_deps:
                if dep in placeholder_to_id:
                    resolved_deps.append(placeholder_to_id[dep])
                else:
                    resolved_deps.append(dep)
            if resolved_deps:
                store.update(created[i], depends_on=resolved_deps)

        return f"Project '{project_name}' created with {len(created)} task(s): {', '.join(created)}"

    # -- FULL (admin) -------------------------------------------------------

    def delete_task(task_id: str) -> str:
        if store.delete(task_id):
            return f"Deleted: {task_id}"
        return f"Task not found: {task_id}"

    def archive_completed(older_than_days: int = 30) -> str:
        return store.summarize_completed(older_than_days=older_than_days)

    # -- Build tool list ----------------------------------------------------

    tools = [
        # OBSERVE
        ToolDefinition(
            name="list_tasks",
            description="List tasks with optional filters by status, project, or tag.",
            parameters={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Filter by status (open, in_progress, blocked, done, cancelled)",
                    },
                    "project": {"type": "string", "description": "Filter by project name"},
                    "tag": {"type": "string", "description": "Filter by tag"},
                    "limit": {"type": "integer", "description": "Max results (default 20)"},
                },
                "required": [],
            },
            function=list_tasks,
            permission="read",
        ),
        ToolDefinition(
            name="get_task",
            description="Get full details for a task by ID.",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID (e.g. t-20260208-001)"},
                },
                "required": ["task_id"],
            },
            function=get_task,
            permission="read",
        ),
        ToolDefinition(
            name="search_tasks",
            description="Search tasks by keyword in title, body, and tags.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keyword"},
                    "limit": {"type": "integer", "description": "Max results (default 10)"},
                },
                "required": ["query"],
            },
            function=search_tasks,
            permission="read",
        ),
        # INTERACT
        ToolDefinition(
            name="create_task",
            description="Create a new task.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Task title"},
                    "description": {"type": "string", "description": "Task description (markdown)"},
                    "priority": {"type": "string", "description": "Priority: low, medium, high, urgent"},
                    "project": {"type": "string", "description": "Project name"},
                    "tags": {"type": "string", "description": "Comma-separated tags"},
                    "due": {"type": "string", "description": "Due date (ISO format)"},
                },
                "required": ["title"],
            },
            function=create_task,
            permission="write",
            requires_approval=False,
        ),
        ToolDefinition(
            name="update_task",
            description="Update task fields. Only non-empty values are changed.",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID"},
                    "title": {"type": "string", "description": "New title"},
                    "description": {"type": "string", "description": "New description"},
                    "priority": {"type": "string", "description": "New priority"},
                    "project": {"type": "string", "description": "New project"},
                    "tags": {"type": "string", "description": "New comma-separated tags"},
                    "due": {"type": "string", "description": "New due date (ISO format)"},
                },
                "required": ["task_id"],
            },
            function=update_task,
            permission="write",
            requires_approval=False,
        ),
        ToolDefinition(
            name="transition_task",
            description="Change a task's status. Validates the state machine.",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID"},
                    "new_status": {
                        "type": "string",
                        "description": "Target status (open, in_progress, blocked, done, cancelled)",
                    },
                },
                "required": ["task_id", "new_status"],
            },
            function=transition_task,
            permission="write",
            requires_approval=False,
        ),
        ToolDefinition(
            name="plan_project",
            description=(
                "Create a project with multiple linked tasks atomically. "
                "tasks_json is a JSON array of {title, description, priority, depends_on} objects. "
                "Use 'task-0', 'task-1' etc. in depends_on to reference tasks by creation order."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "project_name": {"type": "string", "description": "Project name"},
                    "tasks_json": {"type": "string", "description": "JSON array of task definitions"},
                },
                "required": ["project_name", "tasks_json"],
            },
            function=plan_project,
            permission="write",
            requires_approval=False,
        ),
        # FULL
        ToolDefinition(
            name="delete_task",
            description="Permanently delete a task.",
            parameters={
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Task ID to delete"},
                },
                "required": ["task_id"],
            },
            function=delete_task,
            permission="admin",
        ),
        ToolDefinition(
            name="archive_completed",
            description="Archive done tasks older than N days to _archive/ directory.",
            parameters={
                "type": "object",
                "properties": {
                    "older_than_days": {"type": "integer", "description": "Age threshold in days (default 30)"},
                },
                "required": [],
            },
            function=archive_completed,
            permission="admin",
        ),
    ]

    return filter_tools_by_tier(tools, tier)
