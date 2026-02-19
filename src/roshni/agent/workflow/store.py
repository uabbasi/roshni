"""ProjectStore — thread-safe CRUD for workflow projects.

Uses Obsidian as the primary project registry (single source of truth)
when obsidian_projects_dir is configured. Obsidian files contain project
metadata (goal, status, tags, human-written notes). The weeklies-data
directory holds internal workflow execution state only (events, checkpoints,
worker logs, artifacts).

Project IDs are Obsidian filename stems (slugs), e.g. "adu-housing",
"medical-action-plan". Legacy proj-YYYYMMDD-NNN IDs are auto-migrated.
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

from loguru import logger

from .backend import FileWorkflowBackend, _slugify, parse_obsidian_frontmatter
from .events import PROJECT_CREATED, PROJECT_TRANSITIONED
from .models import (
    Artifact,
    Budget,
    JournalEntry,
    Project,
    ProjectStatus,
    validate_transition,
)


def _map_obsidian_status(raw: str) -> ProjectStatus:
    """Map Obsidian frontmatter status values to ProjectStatus.

    Handles various formats: 'active', 'executing', quoted strings, etc.
    Default: EXECUTING (assumed active if no status specified).
    """
    if not raw:
        return ProjectStatus.EXECUTING

    clean = raw.strip().lower().strip('"').strip("'")
    mapping = {
        "active": ProjectStatus.EXECUTING,
        "planning": ProjectStatus.PLANNING,
        "awaiting_approval": ProjectStatus.AWAITING_APPROVAL,
        "executing": ProjectStatus.EXECUTING,
        "reviewing": ProjectStatus.REVIEWING,
        "done": ProjectStatus.DONE,
        "paused": ProjectStatus.PAUSED,
        "failed": ProjectStatus.FAILED,
        "cancelled": ProjectStatus.CANCELLED,
    }
    return mapping.get(clean, ProjectStatus.EXECUTING)


class ProjectStore:
    """Thread-safe project store with Obsidian as the primary registry.

    Project discovery: Obsidian projects/*.md files
    Workflow execution state: weeklies-data/projects/{slug}/ (internal only)

    When no obsidian_projects_dir is configured, falls back to weeklies-data
    as the sole registry (backward compat for tests).
    """

    def __init__(
        self,
        base_dir: str | Path,
        obsidian_projects_dir: str | Path = "",
    ) -> None:
        self._base = Path(base_dir)
        self._base.mkdir(parents=True, exist_ok=True)
        self._obsidian_dir = Path(obsidian_projects_dir) if obsidian_projects_dir else None
        self._backend = FileWorkflowBackend(self._base, obsidian_projects_dir)
        self._create_lock = threading.Lock()
        self._project_cache: dict[str, Project] = {}

    @property
    def backend(self) -> FileWorkflowBackend:
        return self._backend

    # -- ID generation -------------------------------------------------------

    def _make_id(self, goal: str) -> str:
        """Generate a project ID.

        With Obsidian: slug from goal (filename-safe, deduplicated).
        Without Obsidian: legacy proj-YYYYMMDD-NNN format.
        """
        if self._obsidian_dir:
            slug = _slugify(goal)
            base_slug = slug
            counter = 1
            while (self._obsidian_dir / f"{slug}.md").exists():
                slug = f"{base_slug}-{counter}"
                counter += 1
            return slug
        return self._next_sequential_id()

    def _next_sequential_id(self) -> str:
        """Generate proj-YYYYMMDD-NNN (fallback when no Obsidian dir)."""
        today = datetime.now().strftime("%Y%m%d")
        prefix = f"proj-{today}-"
        max_seq = 0
        if self._base.exists():
            for d in self._base.iterdir():
                if d.is_dir() and d.name.startswith(prefix):
                    try:
                        seq = int(d.name[len(prefix) :])
                        max_seq = max(max_seq, seq)
                    except ValueError:
                        pass
        return f"{prefix}{max_seq + 1:03d}"

    # -- Obsidian parsing ---------------------------------------------------

    def _parse_obsidian_project(self, path: Path) -> Project:
        """Parse an Obsidian markdown file into a lightweight Project.

        Handles three frontmatter formats:
        1. Old-style: type: project, created, updated (no status/tags)
        2. Hakim-created: title, status: "active", tags: [...]
        3. Workflow-created: id: proj-..., status, plan_hash, tags
        """
        text = path.read_text(encoding="utf-8")
        fm = parse_obsidian_frontmatter(text)
        slug = path.stem

        # --- Goal: title field > first # heading > prettified slug ---
        goal = fm.get("title", "")
        if isinstance(goal, str):
            goal = goal.strip().strip('"').strip("'")
        if not goal:
            # Strip frontmatter block(s) to find body headings
            body = text
            while body.startswith("---\n"):
                end = body.find("\n---", 4)
                if end == -1:
                    break
                body = body[end + 4 :].lstrip("\n")
            for line in body.split("\n"):
                if line.startswith("# "):
                    goal = line[2:].strip()
                    break
        if not goal:
            goal = slug.replace("-", " ").title()

        # --- Status ---
        status = _map_obsidian_status(str(fm.get("status", "")))

        # --- Tags (normalize to list[str]) ---
        tags = fm.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip().strip("'").strip('"') for t in tags.split(",") if t.strip()]
        elif isinstance(tags, list):
            tags = [str(t).strip().strip("'").strip('"') for t in tags]
        else:
            tags = []

        # --- Dates ---
        created = str(fm.get("created", ""))
        updated = str(fm.get("updated", created))

        return Project(
            id=slug,
            goal=goal,
            status=status,
            tags=tags,
            created=created,
            updated=updated,
            obsidian_file=path.name,
        )

    def _migrate_legacy_workspace(self, slug: str, legacy_id: str) -> None:
        """One-time migration: rename legacy proj-YYYYMMDD-NNN workspace to slug."""
        if not legacy_id or legacy_id == slug:
            return
        legacy_dir = self._base / legacy_id
        slug_dir = self._base / slug
        if legacy_dir.exists() and not slug_dir.exists():
            legacy_dir.rename(slug_dir)
            logger.info(f"Migrated workspace {legacy_id} -> {slug}")

    # -- CRUD ----------------------------------------------------------------

    async def create(
        self,
        goal: str,
        *,
        budget: Budget | None = None,
        tags: list[str] | None = None,
    ) -> Project:
        """Create a new project.

        With Obsidian: the markdown file is the primary artifact; workspace
        holds execution state only. Without Obsidian: workspace is everything.
        """
        with self._create_lock:
            project_id = self._make_id(goal)
            now = datetime.now().isoformat(timespec="seconds")

            project = Project(
                id=project_id,
                goal=goal,
                budget=budget or Budget(),
                tags=tags or [],
                created=now,
                updated=now,
                obsidian_file=f"{project_id}.md" if self._obsidian_dir else "",
                workspace_dir=str(self._backend.get_workspace_path(project_id)),
            )

            # Create workspace directories
            self._backend._ensure_dirs(project_id)

            # Write minimal Obsidian file for registry/deduplication.
            # checkpoint() only renders full Obsidian for projects with phases,
            # so we create a minimal entry here.
            if self._obsidian_dir:
                obs_path = self._obsidian_dir / f"{project_id}.md"
                obs_path.parent.mkdir(parents=True, exist_ok=True)
                tag_str = ", ".join(tags or [])
                obs_content = (
                    f"---\n"
                    f'title: "{goal}"\n'
                    f"status: {project.status.value}\n"
                    f"tags: [{tag_str}]\n"
                    f"created: {now}\n"
                    f"updated: {now}\n"
                    f"---\n\n"
                    f"# {goal}\n"
                )
                obs_path.write_text(obs_content, encoding="utf-8")

            # Record creation event
            evt = self._backend.create_event(project_id, PROJECT_CREATED, "system", {"goal": goal})
            await self._backend.record_event(project_id, evt)
            project.last_event_seq = evt.seq

            project.journal.append(
                JournalEntry(
                    timestamp=now,
                    actor="system",
                    action="created",
                    content=f"Project created: {goal}",
                )
            )

            # Checkpoint (renders Obsidian only if project has phases — see backend)
            await self._backend.checkpoint(project)

            self._project_cache[project_id] = project
            logger.info(f"Created project {project_id}: {goal[:60]}")
            return project

    async def get(self, project_id: str) -> Project | None:
        """Get a project by ID (slug or legacy ID).

        Resolution order:
        1. In-memory cache
        2. Obsidian file (primary registry) + merged workflow state
        3. Weeklies-data only (backward compat for orphaned projects)
        """
        if project_id in self._project_cache:
            return self._project_cache[project_id]

        # 1. Try Obsidian (primary registry)
        if self._obsidian_dir:
            obs_path = self._obsidian_dir / f"{project_id}.md"
            if obs_path.exists():
                obs_project = self._parse_obsidian_project(obs_path)
                fm = parse_obsidian_frontmatter(obs_path.read_text(encoding="utf-8"))

                # Migrate legacy workspace dir if frontmatter has old-style ID
                legacy_id = fm.get("id", "")
                if legacy_id and str(legacy_id) != project_id:
                    self._migrate_legacy_workspace(project_id, str(legacy_id))

                # Check for workflow execution state in weeklies-data
                checkpoint_path = self._backend._project_dir(project_id) / "checkpoint.json"
                if checkpoint_path.exists():
                    try:
                        wf_project = await self._backend.resume(project_id)
                        # Merge: Obsidian metadata + workflow execution state
                        wf_project.id = project_id  # canonical ID = slug
                        wf_project.obsidian_file = obs_project.obsidian_file
                        wf_project.goal = obs_project.goal  # Obsidian is authoritative
                        if obs_project.tags:
                            wf_project.tags = obs_project.tags
                        self._project_cache[project_id] = wf_project
                        return wf_project
                    except Exception as e:
                        logger.warning(f"Failed to load workflow state for {project_id}: {e}")

                # No workflow state — return Obsidian-only project
                self._project_cache[project_id] = obs_project
                return obs_project

        # 2. Fallback: weeklies-data only (orphaned workflow projects)
        checkpoint_path = self._backend._project_dir(project_id) / "checkpoint.json"
        if not checkpoint_path.exists():
            return None

        try:
            project = await self._backend.resume(project_id)
            self._project_cache[project_id] = project
            return project
        except Exception as e:
            logger.warning(f"Failed to load project {project_id}: {e}")
            return None

    async def update(self, project: Project) -> None:
        """Persist current project state."""
        project.updated = datetime.now().isoformat(timespec="seconds")
        await self._backend.checkpoint(project)
        self._project_cache[project.id] = project

    async def delete(self, project_id: str) -> bool:
        """Delete a project workspace and optionally its Obsidian file."""
        import shutil

        deleted = False

        # Remove workspace
        project_dir = self._backend._project_dir(project_id)
        if project_dir.exists():
            shutil.rmtree(project_dir)
            deleted = True

        # Remove Obsidian file
        if self._obsidian_dir:
            obs_path = self._obsidian_dir / f"{project_id}.md"
            if obs_path.exists():
                obs_path.unlink()
                deleted = True

        self._project_cache.pop(project_id, None)
        if deleted:
            logger.info(f"Deleted project {project_id}")
        return deleted

    # -- Query ---------------------------------------------------------------

    async def list_projects(
        self,
        *,
        status: str = "",
        tag: str = "",
        limit: int = 20,
    ) -> list[Project]:
        """List projects. Obsidian is primary, weeklies-data is fallback.

        Scans all Obsidian project files first, merging with workflow state
        when available. Then picks up orphaned weeklies-data projects.
        """
        projects: list[Project] = []
        seen_ids: set[str] = set()

        # 1. Scan Obsidian (primary registry)
        if self._obsidian_dir and self._obsidian_dir.exists():
            for md_file in sorted(self._obsidian_dir.glob("*.md")):
                slug = md_file.stem
                seen_ids.add(slug)
                project = await self.get(slug)
                if project is None:
                    continue
                if status and project.status.value != status:
                    continue
                if tag and tag not in project.tags:
                    continue
                projects.append(project)

        # 2. Pick up orphaned weeklies-data projects not in Obsidian
        for pid in self._backend.list_project_ids():
            if pid in seen_ids:
                continue
            project = await self.get(pid)
            if project is None:
                continue
            if status and project.status.value != status:
                continue
            if tag and tag not in project.tags:
                continue
            projects.append(project)

        projects.sort(key=lambda p: p.updated or p.created, reverse=True)
        return projects[:limit]

    # -- State machine -------------------------------------------------------

    async def transition(self, project_id: str, new_status: str, *, actor: str = "system") -> Project:
        """Transition a project to a new status with validation."""
        project = await self.get(project_id)
        if project is None:
            raise ValueError(f"Project not found: {project_id}")

        target = ProjectStatus(new_status)
        validate_transition(project.status, target)

        old_status = project.status
        project.status = target
        project.updated = datetime.now().isoformat(timespec="seconds")

        # Set started_at when entering EXECUTING
        if target == ProjectStatus.EXECUTING and not project.started_at:
            project.started_at = project.updated

        # Set cancel_requested_at when entering CANCELLED
        if target == ProjectStatus.CANCELLED:
            project.cancel_requested_at = project.updated

        # Ensure workspace exists for recording events
        self._backend._ensure_dirs(project.id)

        # Record transition event
        evt = self._backend.create_event(
            project.id,
            PROJECT_TRANSITIONED,
            actor,
            {"from": old_status.value, "to": target.value},
        )
        await self._backend.record_event(project.id, evt)
        project.last_event_seq = evt.seq

        # Journal
        project.journal.append(
            JournalEntry(
                timestamp=project.updated,
                actor=actor,
                action="status_change",
                content=f"Status: {old_status.value} -> {target.value}",
            )
        )

        await self._backend.checkpoint(project)
        logger.info(f"Project {project_id}: {old_status.value} -> {target.value}")
        return project

    # -- Journal -------------------------------------------------------------

    async def append_journal(
        self,
        project_id: str,
        actor: str,
        action: str,
        content: str,
        *,
        metadata: dict | None = None,
    ) -> None:
        """Append a journal entry to a project."""
        project = await self.get(project_id)
        if project is None:
            raise ValueError(f"Project not found: {project_id}")

        entry = JournalEntry(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            actor=actor,
            action=action,
            content=content,
            metadata=metadata or {},
        )
        project.journal.append(entry)
        await self._backend.checkpoint(project)

    # -- Artifacts -----------------------------------------------------------

    async def save_artifact(
        self,
        project_id: str,
        name: str,
        content: str,
        *,
        mime_type: str = "text/markdown",
    ) -> Artifact:
        """Save an artifact to the project workspace."""
        project = await self.get(project_id)
        if project is None:
            raise ValueError(f"Project not found: {project_id}")

        artifacts_dir = self._backend.get_artifacts_path(project_id)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Sanitize filename
        safe_name = _slugify(name)
        ext = ".md" if mime_type == "text/markdown" else ".txt"
        filename = f"{safe_name}{ext}"
        path = artifacts_dir / filename
        path.write_text(content, encoding="utf-8")

        artifact = Artifact(
            name=name,
            path=f"artifacts/{filename}",
            mime_type=mime_type,
        )
        project.artifacts.append(artifact)
        await self._backend.checkpoint(project)

        logger.info(f"Saved artifact '{name}' for project {project_id}")
        return artifact

    # -- Workspace -----------------------------------------------------------

    def workspace_path(self, project_id: str) -> Path:
        """Get the workspace path for a project."""
        return self._backend.get_workspace_path(project_id)
