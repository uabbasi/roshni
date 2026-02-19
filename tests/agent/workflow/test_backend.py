"""Tests for FileWorkflowBackend — checkpointing, events, replay, conflicts."""

import json

import pytest

from roshni.agent.workflow.backend import (
    FileWorkflowBackend,
    check_obsidian_conflict,
    parse_obsidian_frontmatter,
    render_obsidian,
)
from roshni.agent.workflow.models import (
    Phase,
    PhaseEntry,
    Project,
    ProjectStatus,
    TaskSpec,
    WorkflowEvent,
)


@pytest.fixture
def backend(tmp_path):
    return FileWorkflowBackend(tmp_path / "projects", tmp_path / "obsidian")


@pytest.fixture
def sample_project():
    return Project(
        id="proj-20260215-001",
        goal="Test project",
        phases=[
            Phase(
                id="phase-1",
                name="Research",
                tasks=[TaskSpec(id="task-001", description="Gather data")],
                entry_criteria=[PhaseEntry(description="Requirements defined")],
                exit_criteria=[PhaseEntry(description="Data gathered")],
            )
        ],
        tags=["test"],
    )


class TestEventRecording:
    async def test_append_event(self, backend, sample_project):
        evt = backend.create_event(sample_project.id, "test.event", "test", {"key": "value"})
        await backend.record_event(sample_project.id, evt)

        events_path = backend._project_dir(sample_project.id) / "events.ndjson"
        assert events_path.exists()
        lines = events_path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["type"] == "test.event"
        assert data["seq"] == 1

    async def test_monotonic_seq(self, backend, sample_project):
        for i in range(5):
            evt = backend.create_event(sample_project.id, f"test.event.{i}", "test")
            await backend.record_event(sample_project.id, evt)

        events_path = backend._project_dir(sample_project.id) / "events.ndjson"
        lines = events_path.read_text().strip().split("\n")
        seqs = [json.loads(line)["seq"] for line in lines]
        assert seqs == [1, 2, 3, 4, 5]


class TestCheckpointResume:
    async def test_checkpoint_roundtrip(self, backend, sample_project):
        sample_project.budget.record_call(0.5)
        await backend.checkpoint(sample_project)

        checkpoint_path = backend._project_dir(sample_project.id) / "checkpoint.json"
        assert checkpoint_path.exists()

        # Resume
        loaded = await backend.resume(sample_project.id)
        assert loaded.id == sample_project.id
        assert loaded.goal == sample_project.goal
        assert loaded.budget.cost_used_usd == pytest.approx(0.5)

    async def test_resume_from_events_only(self, backend, sample_project):
        """If checkpoint is missing, rebuild from events."""
        pid = sample_project.id

        # Record creation event
        evt = backend.create_event(pid, "project.created", "system", {"goal": sample_project.goal})
        await backend.record_event(pid, evt)

        # Record a transition
        evt2 = backend.create_event(
            pid,
            "project.transitioned",
            "system",
            {"from": "planning", "to": "awaiting_approval"},
        )
        await backend.record_event(pid, evt2)

        # No checkpoint file — should rebuild
        project = await backend.resume(pid)
        assert project.status == ProjectStatus.AWAITING_APPROVAL

    async def test_resume_replays_events_after_checkpoint(self, backend, sample_project):
        """Events with seq > checkpoint.last_event_seq should be replayed."""
        pid = sample_project.id

        # Write checkpoint at seq 1
        evt1 = backend.create_event(
            pid,
            "project.created",
            "system",
            {"goal": sample_project.goal},
        )
        await backend.record_event(pid, evt1)
        sample_project.last_event_seq = 1
        await backend.checkpoint(sample_project)

        # Write more events after checkpoint
        evt2 = backend.create_event(
            pid,
            "project.transitioned",
            "system",
            {"from": "planning", "to": "awaiting_approval"},
        )
        await backend.record_event(pid, evt2)

        # Resume should replay evt2
        loaded = await backend.resume(pid)
        assert loaded.status == ProjectStatus.AWAITING_APPROVAL
        assert loaded.last_event_seq == 2


class TestEventReplayDeterminism:
    """Events must replay by seq, not timestamp."""

    async def test_replay_by_seq_not_timestamp(self, backend):
        pid = "proj-replay-test"

        # Create events with non-monotonic timestamps but correct seq
        events = [
            WorkflowEvent(
                event_id="evt-000001",
                seq=1,
                type="project.created",
                timestamp="2026-01-01T12:00:00",
                actor="system",
                payload={"goal": "test"},
            ),
            WorkflowEvent(
                event_id="evt-000002",
                seq=2,
                type="project.transitioned",
                timestamp="2026-01-01T11:00:00",
                actor="system",
                payload={"from": "planning", "to": "awaiting_approval"},
            ),
            WorkflowEvent(
                event_id="evt-000003",
                seq=3,
                type="project.transitioned",
                timestamp="2026-01-01T10:00:00",
                actor="system",
                payload={"from": "awaiting_approval", "to": "executing"},
            ),
        ]

        for evt in events:
            await backend.record_event(pid, evt)

        # Resume — should replay by seq regardless of timestamp
        project = await backend.resume(pid)
        assert project.status == ProjectStatus.EXECUTING


class TestResumeFuzz:
    """Test resume with partial/corrupt files."""

    async def test_truncated_checkpoint(self, backend, sample_project):
        """Truncated checkpoint should fall back to event replay."""
        pid = sample_project.id

        # Write a creation event
        evt = backend.create_event(pid, "project.created", "system", {"goal": "test"})
        await backend.record_event(pid, evt)

        # Write a corrupt checkpoint
        checkpoint_path = backend._project_dir(pid) / "checkpoint.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text('{"id": "proj-test", "goal": "test"', encoding="utf-8")  # truncated JSON

        # Should recover from events
        project = await backend.resume(pid)
        assert project is not None
        assert project.goal == "test"

    async def test_missing_checkpoint_with_events(self, backend):
        """Missing checkpoint + events should rebuild."""
        pid = "proj-missing-cp"
        evt = backend.create_event(pid, "project.created", "system", {"goal": "rebuilt"})
        await backend.record_event(pid, evt)

        project = await backend.resume(pid)
        assert project.goal == "rebuilt"

    async def test_no_checkpoint_no_events(self, backend):
        """No checkpoint and no events should raise."""
        with pytest.raises(ValueError, match="No checkpoint or events"):
            await backend.resume("proj-nonexistent")


class TestObsidianConflict:
    def test_no_conflict_when_unchanged(self, tmp_path):
        obs_path = tmp_path / "test.md"
        obs_path.write_text("---\nplan_hash: abc123\n---\n# Test")
        # mtime will match what we pass
        mtime = obs_path.stat().st_mtime
        from datetime import datetime

        last_update = datetime.fromtimestamp(mtime).isoformat(timespec="seconds")
        result = check_obsidian_conflict(obs_path, "abc123", last_update)
        assert result is None

    def test_cosmetic_edit_no_conflict(self, tmp_path):
        """If plan_hash is the same, cosmetic edits don't trigger conflict."""
        obs_path = tmp_path / "test.md"
        obs_path.write_text("---\nplan_hash: abc123\n---\n# Test with notes added")

        # Set last_update to before the file mtime
        result = check_obsidian_conflict(obs_path, "abc123", "2020-01-01T00:00:00")
        assert result is None  # plan_hash unchanged -> no conflict

    def test_plan_change_triggers_conflict(self, tmp_path):
        """If plan_hash changed, it's a real conflict."""
        obs_path = tmp_path / "test.md"
        obs_path.write_text("---\nplan_hash: different_hash\n---\n# Modified plan")

        result = check_obsidian_conflict(obs_path, "abc123", "2020-01-01T00:00:00")
        assert result is not None
        assert "Plan hash changed" in result

    def test_missing_file_no_conflict(self, tmp_path):
        obs_path = tmp_path / "nonexistent.md"
        result = check_obsidian_conflict(obs_path, "abc123", "2026-01-01T00:00:00")
        assert result is None


class TestObsidianRendering:
    def test_render_basic(self, sample_project):
        md = render_obsidian(sample_project, "/tmp/obsidian")
        assert "---" in md
        assert sample_project.goal in md
        assert "phase-1" in md or "Research" in md

    def test_frontmatter_parsing(self):
        text = "---\nid: test\nstatus: executing\nplan_hash: abc\n---\n# Content"
        fm = parse_obsidian_frontmatter(text)
        assert fm["id"] == "test"
        assert fm["status"] == "executing"
        assert fm["plan_hash"] == "abc"


class TestLLMCallRecording:
    async def test_record_llm_call(self, backend, sample_project):
        await backend.record_llm_call(
            sample_project.id,
            {
                "id": "call-123",
                "model": "test-model",
                "tokens": 100,
            },
        )
        llm_dir = backend._project_dir(sample_project.id) / "llm-calls"
        assert llm_dir.exists()
        files = list(llm_dir.glob("*.json"))
        assert len(files) == 1
