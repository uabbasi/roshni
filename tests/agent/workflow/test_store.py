"""Tests for ProjectStore — CRUD, state machine, workspace management.

Two fixture modes:
- tmp_store: no Obsidian dir (legacy / test fallback)
- obs_store: with Obsidian dir (primary mode — Obsidian as registry)
"""

import json

import pytest

from roshni.agent.workflow.models import ProjectStatus
from roshni.agent.workflow.store import ProjectStore, _map_obsidian_status


@pytest.fixture
def tmp_store(tmp_path):
    """Create a ProjectStore without Obsidian (legacy mode)."""
    return ProjectStore(tmp_path / "projects")


@pytest.fixture
def obs_store(tmp_path):
    """Create a ProjectStore with Obsidian as primary registry."""
    projects_dir = tmp_path / "weeklies" / "projects"
    obsidian_dir = tmp_path / "obsidian" / "projects"
    obsidian_dir.mkdir(parents=True)
    return ProjectStore(projects_dir, obsidian_dir)


def _write_obsidian_project(obsidian_dir, filename, frontmatter_yaml, body=""):
    """Helper: write an Obsidian project markdown file."""
    path = obsidian_dir / filename
    content = f"---\n{frontmatter_yaml}\n---\n{body}"
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Legacy mode tests (no Obsidian) — existing behavior preserved
# ---------------------------------------------------------------------------


@pytest.mark.smoke
class TestProjectStoreSmoke:
    """Fast smoke tests — no LLM, no Obsidian."""

    async def test_create_and_get(self, tmp_store):
        project = await tmp_store.create("Build something cool")
        assert project.id.startswith("proj-")
        assert project.goal == "Build something cool"
        assert project.status == ProjectStatus.PLANNING

        loaded = await tmp_store.get(project.id)
        assert loaded is not None
        assert loaded.id == project.id
        assert loaded.goal == project.goal

    async def test_id_generation(self, tmp_store):
        p1 = await tmp_store.create("Project 1")
        p2 = await tmp_store.create("Project 2")
        assert p1.id != p2.id
        # Both should be from today
        assert p1.id.split("-")[1] == p2.id.split("-")[1]

    async def test_list_projects(self, tmp_store):
        await tmp_store.create("Project A")
        await tmp_store.create("Project B")

        projects = await tmp_store.list_projects()
        assert len(projects) == 2

    async def test_transition_valid(self, tmp_store):
        project = await tmp_store.create("Test transition")
        updated = await tmp_store.transition(project.id, "awaiting_approval")
        assert updated.status == ProjectStatus.AWAITING_APPROVAL

    async def test_transition_invalid(self, tmp_store):
        project = await tmp_store.create("Test transition")
        with pytest.raises(ValueError, match="Invalid transition"):
            await tmp_store.transition(project.id, "done")

    async def test_delete(self, tmp_store):
        project = await tmp_store.create("Delete me")
        assert await tmp_store.delete(project.id)
        assert await tmp_store.get(project.id) is None

    async def test_journal_append(self, tmp_store):
        project = await tmp_store.create("Journal test")
        await tmp_store.append_journal(project.id, "user", "test", "Test entry")
        loaded = await tmp_store.get(project.id)
        # Journal should have creation entry + our entry
        assert len(loaded.journal) >= 2
        assert loaded.journal[-1].content == "Test entry"

    async def test_workspace_path(self, tmp_store):
        project = await tmp_store.create("Workspace test")
        path = tmp_store.workspace_path(project.id)
        assert path.exists()


class TestProjectStoreWorkspace:
    """Test workspace directory creation."""

    async def test_workspace_dirs_created(self, tmp_store):
        project = await tmp_store.create("Dir test")
        ws = tmp_store.workspace_path(project.id)
        assert (ws / "worker-logs").exists()
        assert (ws / "llm-calls").exists()
        assert (ws / "artifacts").exists()

    async def test_save_artifact(self, tmp_store):
        project = await tmp_store.create("Artifact test")
        artifact = await tmp_store.save_artifact(
            project.id,
            "research-report",
            "# Research Report\n\nContent here.",
            mime_type="text/markdown",
        )
        assert artifact.name == "research-report"
        loaded = await tmp_store.get(project.id)
        assert len(loaded.artifacts) == 1


class TestProjectStoreFiltering:
    """Test list filtering."""

    async def test_filter_by_tag(self, tmp_store):
        await tmp_store.create("Tagged", tags=["health"])
        await tmp_store.create("Untagged")

        tagged = await tmp_store.list_projects(tag="health")
        assert len(tagged) == 1
        assert tagged[0].tags == ["health"]


class TestProjectStoreThreadSafety:
    """Test thread safety of create."""

    async def test_concurrent_creates(self, tmp_store):
        import asyncio

        tasks = [tmp_store.create(f"Project {i}") for i in range(5)]
        projects = await asyncio.gather(*tasks)
        ids = [p.id for p in projects]
        assert len(set(ids)) == 5  # All unique


class TestProjectStoreEventLog:
    """Test that events are recorded to NDJSON."""

    async def test_events_file_created(self, tmp_store):
        project = await tmp_store.create("Event test")
        events_path = tmp_store.workspace_path(project.id) / "events.ndjson"
        assert events_path.exists()
        lines = events_path.read_text().strip().split("\n")
        assert len(lines) >= 1  # At least creation event

    async def test_transition_records_event(self, tmp_store):
        project = await tmp_store.create("Transition event test")
        await tmp_store.transition(project.id, "awaiting_approval")

        events_path = tmp_store.workspace_path(project.id) / "events.ndjson"

        lines = events_path.read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]
        types = [e["type"] for e in events]
        assert "project.created" in types
        assert "project.transitioned" in types


# ---------------------------------------------------------------------------
# Obsidian-primary mode tests
# ---------------------------------------------------------------------------


class TestObsidianStatusMapping:
    """Test _map_obsidian_status handles all frontmatter formats."""

    def test_active_maps_to_executing(self):
        assert _map_obsidian_status("active") == ProjectStatus.EXECUTING
        assert _map_obsidian_status('"active"') == ProjectStatus.EXECUTING
        assert _map_obsidian_status("Active") == ProjectStatus.EXECUTING

    def test_empty_maps_to_executing(self):
        assert _map_obsidian_status("") == ProjectStatus.EXECUTING

    def test_workflow_statuses(self):
        assert _map_obsidian_status("planning") == ProjectStatus.PLANNING
        assert _map_obsidian_status("executing") == ProjectStatus.EXECUTING
        assert _map_obsidian_status("done") == ProjectStatus.DONE
        assert _map_obsidian_status("paused") == ProjectStatus.PAUSED

    def test_unknown_maps_to_executing(self):
        assert _map_obsidian_status("some-random-status") == ProjectStatus.EXECUTING


@pytest.mark.smoke
class TestObsidianPrimarySmoke:
    """Obsidian as primary registry — smoke tests."""

    async def test_list_sees_obsidian_projects(self, obs_store):
        """Core test: list_projects returns Obsidian-only projects."""
        obsidian_dir = obs_store._obsidian_dir
        _write_obsidian_project(
            obsidian_dir,
            "adu-housing.md",
            "type: project\ncreated: 2026-02-08\nupdated: 2026-02-15",
            "\n# ADU Housing (Master)\nSome content here.",
        )
        _write_obsidian_project(
            obsidian_dir,
            "umbrella-insurance.md",
            'title: "Umbrella Insurance Upgrade"\nstatus: "active"\ntags: [finance, insurance]',
        )

        projects = await obs_store.list_projects()
        assert len(projects) == 2
        ids = {p.id for p in projects}
        assert "adu-housing" in ids
        assert "umbrella-insurance" in ids

    async def test_get_obsidian_only_project(self, obs_store):
        """Get returns a lightweight Project from Obsidian markdown."""
        _write_obsidian_project(
            obs_store._obsidian_dir,
            "medical-action-plan.md",
            'title: "Medical Action Plan"\nstatus: "active"\ntags: [health, medical]',
            "\n# Phases\n## Phase 1: Urgent\n- Oral surgeon\n",
        )

        project = await obs_store.get("medical-action-plan")
        assert project is not None
        assert project.id == "medical-action-plan"
        assert project.goal == "Medical Action Plan"
        assert project.status == ProjectStatus.EXECUTING  # "active" -> EXECUTING
        assert "health" in project.tags
        assert "medical" in project.tags
        assert len(project.phases) == 0  # Obsidian-only: no workflow phases

    async def test_get_old_style_frontmatter(self, obs_store):
        """Old-style frontmatter (type: project) extracts goal from heading."""
        _write_obsidian_project(
            obs_store._obsidian_dir,
            "adu-housing.md",
            "type: project\ncreated: 2026-02-08",
            "\n# ADU Housing (Master)\nVendor shortlist here.",
        )

        project = await obs_store.get("adu-housing")
        assert project is not None
        assert project.goal == "ADU Housing (Master)"
        assert project.status == ProjectStatus.EXECUTING  # no status -> EXECUTING

    async def test_get_nonexistent_returns_none(self, obs_store):
        assert await obs_store.get("does-not-exist") is None

    async def test_create_uses_slug_id(self, obs_store):
        """With Obsidian, create() generates slug-based IDs."""
        project = await obs_store.create("Build a meditation practice")
        assert project.id == "build-a-meditation-practice"
        assert project.obsidian_file == "build-a-meditation-practice.md"

    async def test_create_deduplicates_slug(self, obs_store):
        """Duplicate goals get numbered slugs."""
        p1 = await obs_store.create("Health plan")
        p2 = await obs_store.create("Health plan")
        assert p1.id == "health-plan"
        assert p2.id == "health-plan-1"


class TestObsidianWorkflowMerge:
    """Test merging Obsidian metadata with workflow execution state."""

    async def test_workflow_project_merges_with_obsidian(self, obs_store):
        """Workflow-created project: checkpoint provides phases/budget, Obsidian provides goal/tags."""
        # 1. Create a project via the workflow system
        project = await obs_store.create("Research solar panels", tags=["energy"])
        project_id = project.id

        # 2. Clear cache to force re-read from disk
        obs_store._project_cache.clear()

        # 3. Get should merge Obsidian (goal/tags) with checkpoint (workflow state)
        loaded = await obs_store.get(project_id)
        assert loaded is not None
        assert loaded.id == project_id
        assert loaded.goal == "Research solar panels"
        assert loaded.status == ProjectStatus.PLANNING
        assert loaded.last_event_seq > 0  # has workflow events

    async def test_obsidian_goal_is_authoritative(self, obs_store):
        """If Obsidian goal differs from checkpoint, Obsidian wins."""
        # Create workflow project
        project = await obs_store.create("Research solar panels")
        project_id = project.id

        # Manually edit the Obsidian file to change the title
        obs_path = obs_store._obsidian_dir / f"{project_id}.md"
        # The file may not exist yet (no phases), so create it
        obs_path.write_text(
            '---\ntitle: "Solar Panel Deep Dive"\nstatus: planning\n---\n\n# Solar Panel Deep Dive\n',
            encoding="utf-8",
        )

        # Clear cache
        obs_store._project_cache.clear()

        loaded = await obs_store.get(project_id)
        assert loaded is not None
        assert loaded.goal == "Solar Panel Deep Dive"

    async def test_obsidian_only_plus_workflow_projects(self, obs_store):
        """list_projects shows both Obsidian-only and workflow-managed projects."""
        # Add an Obsidian-only project
        _write_obsidian_project(
            obs_store._obsidian_dir,
            "adu-housing.md",
            "type: project\ncreated: 2026-02-08",
            "\n# ADU Housing\n",
        )

        # Create a workflow-managed project
        await obs_store.create("Meditation research", tags=["health"])

        projects = await obs_store.list_projects()
        assert len(projects) == 2
        ids = {p.id for p in projects}
        assert "adu-housing" in ids
        assert "meditation-research" in ids

        # The Obsidian-only project has no phases
        adu = next(p for p in projects if p.id == "adu-housing")
        assert len(adu.phases) == 0
        assert adu.last_event_seq == 0

        # The workflow project has events
        med = next(p for p in projects if p.id == "meditation-research")
        assert med.last_event_seq > 0


class TestObsidianLegacyMigration:
    """Test migration of legacy proj-YYYYMMDD-NNN workspace dirs."""

    async def test_legacy_workspace_migrated_to_slug(self, obs_store):
        """When Obsidian frontmatter has an old-style id, workspace dir is renamed."""
        # Simulate a legacy workspace
        legacy_id = "proj-20260216-002"
        legacy_dir = obs_store._base / legacy_id
        (legacy_dir / "worker-logs").mkdir(parents=True)
        (legacy_dir / "checkpoint.json").write_text("{}", encoding="utf-8")

        # Create an Obsidian file that references the legacy workspace
        _write_obsidian_project(
            obs_store._obsidian_dir,
            "meditation-research.md",
            f"id: {legacy_id}\nstatus: executing\ntags: [health]",
            "\n# Meditation Research\n",
        )

        # get() should migrate the workspace
        project = await obs_store.get("meditation-research")
        assert project is not None
        assert project.id == "meditation-research"

        # Legacy dir should be renamed to slug
        assert not legacy_dir.exists()
        slug_dir = obs_store._base / "meditation-research"
        assert slug_dir.exists()


class TestObsidianDeleteAndFilter:
    """Test delete and filter with Obsidian primary."""

    async def test_delete_removes_obsidian_file(self, obs_store):
        """Delete removes both the Obsidian file and workspace."""
        _write_obsidian_project(
            obs_store._obsidian_dir,
            "temp-project.md",
            'title: "Temp"\nstatus: active',
        )

        # Verify it exists
        project = await obs_store.get("temp-project")
        assert project is not None

        # Delete
        assert await obs_store.delete("temp-project")
        assert await obs_store.get("temp-project") is None
        assert not (obs_store._obsidian_dir / "temp-project.md").exists()

    async def test_filter_by_tag_obsidian(self, obs_store):
        """Tag filtering works with Obsidian frontmatter tags."""
        _write_obsidian_project(
            obs_store._obsidian_dir,
            "health-project.md",
            'title: "Health"\nstatus: active\ntags: [health, wellness]',
        )
        _write_obsidian_project(
            obs_store._obsidian_dir,
            "finance-project.md",
            'title: "Finance"\nstatus: active\ntags: [finance]',
        )

        health = await obs_store.list_projects(tag="health")
        assert len(health) == 1
        assert health[0].id == "health-project"


class TestObsidianNoOverwrite:
    """Verify that checkpoint() does NOT overwrite hand-crafted Obsidian files."""

    async def test_transition_preserves_obsidian_content(self, obs_store):
        """Transitioning an Obsidian-only project does not overwrite its markdown."""
        original_content = (
            "---\ntype: project\ncreated: 2026-02-08\n---\n\n"
            "# ADU Housing (Master)\n\n"
            "## Vendor Shortlist\n- Studio Shed\n- Type Five\n"
        )
        obs_path = obs_store._obsidian_dir / "adu-housing.md"
        obs_path.write_text(original_content, encoding="utf-8")

        # Transition creates workflow state but should NOT touch the Obsidian file
        # (project has no phases, so the guard in checkpoint() skips rendering)
        await obs_store.transition("adu-housing", "paused", actor="user")

        # Obsidian file should be unchanged
        assert obs_path.read_text(encoding="utf-8") == original_content

        # But workflow state should exist
        workspace = obs_store._base / "adu-housing"
        assert (workspace / "checkpoint.json").exists()
        assert (workspace / "events.ndjson").exists()
