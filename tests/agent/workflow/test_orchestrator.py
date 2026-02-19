"""Tests for Orchestrator — mocked LLM, no API keys needed.

Patches DefaultAgent.invoke() to return canned responses, and
WorkerPool.spawn_worker() to return controlled WorkerResults.
Uses real ProjectStore + FileWorkflowBackend with tmp_path.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from roshni.agent.workflow.backend import FileWorkflowBackend
from roshni.agent.workflow.models import (
    Artifact,
    Budget,
    PhaseStatus,
    ProjectStatus,
    TaskSpec,
    TerminalCondition,
)
from roshni.agent.workflow.orchestrator import Orchestrator
from roshni.agent.workflow.store import ProjectStore
from roshni.agent.workflow.worker import WorkerPool, WorkerResult

# ---------------------------------------------------------------------------
# Canned LLM responses
# ---------------------------------------------------------------------------

GOOD_PLAN_JSON = json.dumps(
    {
        "phases": [
            {
                "id": "phase-1",
                "name": "Research",
                "description": "Gather background info",
                "tasks": [
                    {"id": "task-001", "description": "Search the web"},
                    {"id": "task-002", "description": "Summarize findings"},
                ],
                "entry_criteria": ["Goal defined"],
                "exit_criteria": ["Summary written"],
            },
            {
                "id": "phase-2",
                "name": "Execute",
                "description": "Implement the plan",
                "tasks": [{"id": "task-003", "description": "Write the code"}],
                "entry_criteria": ["Research complete"],
                "exit_criteria": ["Code working"],
            },
        ]
    }
)

GOOD_EVAL_JSON = json.dumps({"met": True, "rationale": "All outputs produced", "evidence": ["artifact found"]})
BAD_EVAL_JSON = json.dumps({"met": False, "rationale": "Missing artifact", "evidence": []})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    return ProjectStore(tmp_path / "projects")


@pytest.fixture
def backend(tmp_path):
    return FileWorkflowBackend(tmp_path / "projects")


@pytest.fixture
def mock_config():
    cfg = MagicMock()
    cfg.get.return_value = {}
    return cfg


@pytest.fixture
def mock_secrets():
    return MagicMock()


@pytest.fixture
def send_fn():
    return MagicMock()


def _make_worker_result(task: TaskSpec, *, success: bool = True, error: str = "") -> WorkerResult:
    return WorkerResult(
        worker_id="worker-abc123",
        task=task,
        response="Done" if success else "",
        success=success,
        error=error,
        duration=0.5,
    )


@pytest.fixture
def orchestrator(mock_config, mock_secrets, store, backend, send_fn):
    """Orchestrator with mocked worker pool — workers don't actually run."""
    pool = MagicMock(spec=WorkerPool)
    pool.spawn_worker = AsyncMock(side_effect=lambda proj, phase, task, **kw: _make_worker_result(task))
    return Orchestrator(
        mock_config,
        mock_secrets,
        store,
        backend,
        send_fn=send_fn,
        worker_pool=pool,
    )


# ---------------------------------------------------------------------------
# start_project
# ---------------------------------------------------------------------------


class TestStartProject:
    @patch("roshni.agent.default.DefaultAgent")
    async def test_creates_plan_from_llm(self, MockAgent, orchestrator):
        MockAgent.return_value.invoke = AsyncMock(return_value=GOOD_PLAN_JSON)

        project = await orchestrator.start_project("Build a dashboard")

        assert project.status == ProjectStatus.AWAITING_APPROVAL
        assert len(project.phases) == 2
        assert project.phases[0].name == "Research"
        assert project.phases[1].name == "Execute"
        assert len(project.phases[0].tasks) == 2
        assert project.plan_hash != ""

    @patch("roshni.agent.default.DefaultAgent")
    async def test_fallback_on_bad_json(self, MockAgent, orchestrator):
        MockAgent.return_value.invoke = AsyncMock(return_value="This is not JSON at all")

        project = await orchestrator.start_project("Do something")

        assert project.status == ProjectStatus.AWAITING_APPROVAL
        # Fallback: single phase with the goal as description
        assert len(project.phases) == 1
        assert project.phases[0].name == "Execute goal"
        assert any("Fallback" in j.content for j in project.journal)

    @patch("roshni.agent.default.DefaultAgent")
    async def test_handles_markdown_fenced_json(self, MockAgent, orchestrator):
        fenced = f"```json\n{GOOD_PLAN_JSON}\n```"
        MockAgent.return_value.invoke = AsyncMock(return_value=fenced)

        project = await orchestrator.start_project("Build it")

        assert project.status == ProjectStatus.AWAITING_APPROVAL
        assert len(project.phases) == 2

    @patch("roshni.agent.default.DefaultAgent")
    async def test_failure_transitions_to_failed(self, MockAgent, orchestrator):
        MockAgent.return_value.invoke = AsyncMock(side_effect=RuntimeError("LLM down"))

        project = await orchestrator.start_project("Impossible task")

        assert project.status == ProjectStatus.FAILED
        assert any("Planning failed" in j.content for j in project.journal)

    @patch("roshni.agent.default.DefaultAgent")
    async def test_reports_to_user(self, MockAgent, orchestrator, send_fn):
        MockAgent.return_value.invoke = AsyncMock(return_value=GOOD_PLAN_JSON)

        await orchestrator.start_project("Build a thing")

        # Should have called send_fn at least twice: creation + plan ready
        assert send_fn.call_count >= 2
        messages = [call.args[0] for call in send_fn.call_args_list]
        assert any("created" in m.lower() for m in messages)
        assert any("plan ready" in m.lower() for m in messages)

    @patch("roshni.agent.default.DefaultAgent")
    async def test_events_recorded(self, MockAgent, orchestrator, store):
        MockAgent.return_value.invoke = AsyncMock(return_value=GOOD_PLAN_JSON)

        project = await orchestrator.start_project("Track events")

        # Event seq should have advanced (plan.written + at least one transition)
        assert project.last_event_seq >= 2
        # Verify events on disk
        events_path = store.workspace_path(project.id) / "events.ndjson"
        assert events_path.exists()
        lines = [l for l in events_path.read_text().splitlines() if l.strip()]
        assert len(lines) >= 3  # project.created + plan.written + project.transitioned

    @patch("roshni.agent.default.DefaultAgent")
    async def test_tags_persisted(self, MockAgent, orchestrator, store):
        MockAgent.return_value.invoke = AsyncMock(return_value=GOOD_PLAN_JSON)

        project = await orchestrator.start_project("Tagged project", tags=["health", "urgent"])

        assert project.tags == ["health", "urgent"]
        reloaded = await store.get(project.id)
        assert reloaded.tags == ["health", "urgent"]


# ---------------------------------------------------------------------------
# approve_and_execute
# ---------------------------------------------------------------------------


class TestApproveAndExecute:
    @patch("roshni.agent.default.DefaultAgent")
    async def test_happy_path_all_phases_done(self, MockAgent, orchestrator, send_fn):
        MockAgent.return_value.invoke = AsyncMock(return_value=GOOD_PLAN_JSON)

        project = await orchestrator.start_project("E2E test")
        await orchestrator.approve_and_execute(project.id)

        # Reload from store
        updated = await orchestrator._store.get(project.id)
        assert updated.status == ProjectStatus.DONE
        assert all(p.status == PhaseStatus.COMPLETED for p in updated.phases)

    @patch("roshni.agent.default.DefaultAgent")
    async def test_wrong_status_raises(self, MockAgent, orchestrator):
        MockAgent.return_value.invoke = AsyncMock(return_value=GOOD_PLAN_JSON)
        project = await orchestrator.start_project("Test")

        # Manually transition to FAILED so approve is invalid
        project.status = ProjectStatus.FAILED
        await orchestrator._store.update(project)

        with pytest.raises(ValueError, match="not awaiting_approval"):
            await orchestrator.approve_and_execute(project.id)

    @patch("roshni.agent.default.DefaultAgent")
    async def test_nonexistent_project_raises(self, MockAgent, orchestrator):
        with pytest.raises(ValueError, match="not found"):
            await orchestrator.approve_and_execute("proj-99999-001")

    @patch("roshni.agent.default.DefaultAgent")
    async def test_worker_failure_fails_project(self, MockAgent, orchestrator):
        MockAgent.return_value.invoke = AsyncMock(return_value=GOOD_PLAN_JSON)
        project = await orchestrator.start_project("Fail test")

        # Make worker pool fail on the first task
        orchestrator._worker_pool.spawn_worker = AsyncMock(
            side_effect=lambda proj, phase, task, **kw: _make_worker_result(task, success=False, error="Boom")
        )

        await orchestrator.approve_and_execute(project.id)

        updated = await orchestrator._store.get(project.id)
        assert updated.status == ProjectStatus.FAILED
        assert any("failed" in j.action for j in updated.journal)

    @patch("roshni.agent.default.DefaultAgent")
    async def test_task_retry_on_failure(self, MockAgent, orchestrator):
        MockAgent.return_value.invoke = AsyncMock(
            return_value=json.dumps(
                {
                    "phases": [
                        {
                            "id": "phase-1",
                            "name": "Retryable",
                            "tasks": [{"id": "task-001", "description": "Flaky task"}],
                            "entry_criteria": [],
                            "exit_criteria": [],
                        }
                    ]
                }
            )
        )

        project = await orchestrator.start_project("Retry test")
        # Give the task max_attempts=2
        project.phases[0].tasks[0].max_attempts = 2
        await orchestrator._store.update(project)

        # First call fails, second succeeds
        call_count = 0

        async def flaky_worker(proj, phase, task, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_worker_result(task, success=False, error="Transient error")
            return _make_worker_result(task, success=True)

        orchestrator._worker_pool.spawn_worker = AsyncMock(side_effect=flaky_worker)

        await orchestrator.approve_and_execute(project.id)

        updated = await orchestrator._store.get(project.id)
        assert updated.status == ProjectStatus.DONE
        assert call_count == 2

    @patch("roshni.agent.default.DefaultAgent")
    async def test_budget_exhaustion_pauses(self, MockAgent, orchestrator):
        MockAgent.return_value.invoke = AsyncMock(return_value=GOOD_PLAN_JSON)

        project = await orchestrator.start_project("Budget test", budget=Budget(max_cost_usd=0.01, max_llm_calls=1))
        # Exhaust the budget before execution
        project.budget.record_call(0.01)
        await orchestrator._store.update(project)

        await orchestrator.approve_and_execute(project.id)

        updated = await orchestrator._store.get(project.id)
        assert updated.status == ProjectStatus.PAUSED
        assert any("budget" in j.content.lower() for j in updated.journal)

    @patch("roshni.agent.default.DefaultAgent")
    async def test_skips_completed_phases(self, MockAgent, orchestrator):
        MockAgent.return_value.invoke = AsyncMock(return_value=GOOD_PLAN_JSON)
        project = await orchestrator.start_project("Skip test")

        # Pre-complete phase-1
        project.phases[0].status = PhaseStatus.COMPLETED
        await orchestrator._store.update(project)

        await orchestrator.approve_and_execute(project.id)

        # Worker should only be called for phase-2 tasks
        updated = await orchestrator._store.get(project.id)
        assert updated.status == ProjectStatus.DONE
        # Phase-2 has 1 task, phase-1 was skipped
        assert orchestrator._worker_pool.spawn_worker.call_count == 1

    @patch("roshni.agent.default.DefaultAgent")
    async def test_no_worker_pool_fails_gracefully(self, MockAgent, mock_config, mock_secrets, store, backend, send_fn):
        """Orchestrator with no worker pool logs error and fails."""
        MockAgent.return_value.invoke = AsyncMock(return_value=GOOD_PLAN_JSON)

        orch = Orchestrator(mock_config, mock_secrets, store, backend, send_fn=send_fn, worker_pool=None)
        project = await orch.start_project("No pool")
        await orch.approve_and_execute(project.id)

        updated = await store.get(project.id)
        assert updated.status == ProjectStatus.FAILED
        assert any("worker pool" in j.content.lower() for j in updated.journal)

    @patch("roshni.agent.default.DefaultAgent")
    async def test_phase_events_recorded(self, MockAgent, orchestrator, store):
        MockAgent.return_value.invoke = AsyncMock(return_value=GOOD_PLAN_JSON)
        project = await orchestrator.start_project("Events test")
        await orchestrator.approve_and_execute(project.id)

        updated = await store.get(project.id)
        events_path = store.workspace_path(updated.id) / "events.ndjson"
        lines = [l for l in events_path.read_text().splitlines() if l.strip()]
        event_types = [json.loads(l)["type"] for l in lines]

        assert "phase.started" in event_types
        assert "phase.completed" in event_types
        # Should have transitions: planning->awaiting, awaiting->executing, executing->reviewing, reviewing->done
        transitions = [json.loads(l)["payload"] for l in lines if json.loads(l)["type"] == "project.transitioned"]
        transition_targets = [t["to"] for t in transitions]
        assert "awaiting_approval" in transition_targets
        assert "executing" in transition_targets
        assert "done" in transition_targets


# ---------------------------------------------------------------------------
# Terminal conditions
# ---------------------------------------------------------------------------


class TestTerminalConditions:
    @patch("roshni.agent.default.DefaultAgent")
    async def test_no_conditions_means_done(self, MockAgent, orchestrator):
        """If no terminal conditions are defined, completing all phases = DONE."""
        MockAgent.return_value.invoke = AsyncMock(return_value=GOOD_PLAN_JSON)

        project = await orchestrator.start_project("No conditions")
        assert project.terminal_conditions == []
        await orchestrator.approve_and_execute(project.id)

        updated = await orchestrator._store.get(project.id)
        assert updated.status == ProjectStatus.DONE

    @patch("roshni.agent.default.DefaultAgent")
    async def test_artifact_exists_condition(self, MockAgent, orchestrator, store):
        MockAgent.return_value.invoke = AsyncMock(return_value=GOOD_PLAN_JSON)

        project = await orchestrator.start_project("Artifact test")
        project.terminal_conditions = [
            TerminalCondition(description="Report exists", type="artifact_exists", params={"name": "report"})
        ]
        # Add the artifact so the condition is met
        project.artifacts.append(Artifact(name="report", path="artifacts/report.md"))
        await store.update(project)

        await orchestrator.approve_and_execute(project.id)

        updated = await store.get(project.id)
        assert updated.status == ProjectStatus.DONE
        assert updated.terminal_conditions[0].met is True
        assert updated.terminal_conditions[0].met_at is not None

    @patch("roshni.agent.default.DefaultAgent")
    async def test_artifact_missing_stays_reviewing(self, MockAgent, orchestrator, store):
        MockAgent.return_value.invoke = AsyncMock(return_value=GOOD_PLAN_JSON)

        project = await orchestrator.start_project("Missing artifact")
        project.terminal_conditions = [
            TerminalCondition(description="Report exists", type="artifact_exists", params={"name": "report"})
        ]
        # No artifact added
        await store.update(project)

        await orchestrator.approve_and_execute(project.id)

        updated = await store.get(project.id)
        # Not DONE because artifact_exists condition is not met
        assert updated.status == ProjectStatus.REVIEWING
        assert updated.terminal_conditions[0].met is False

    @patch("roshni.agent.default.DefaultAgent")
    async def test_phase_count_condition(self, MockAgent, orchestrator, store):
        MockAgent.return_value.invoke = AsyncMock(return_value=GOOD_PLAN_JSON)

        project = await orchestrator.start_project("Phase count test")
        project.terminal_conditions = [
            TerminalCondition(description="At least 2 phases done", type="phase_count", params={"min_completed": 2})
        ]
        await store.update(project)

        await orchestrator.approve_and_execute(project.id)

        updated = await store.get(project.id)
        # Plan has 2 phases, both completed -> condition met -> DONE
        assert updated.status == ProjectStatus.DONE
        assert updated.terminal_conditions[0].met is True

    @patch("roshni.agent.default.DefaultAgent")
    async def test_llm_eval_condition(self, MockAgent, orchestrator, store):
        # First call: planning. Second call: evaluation.
        call_count = 0

        async def mock_invoke(prompt, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return GOOD_PLAN_JSON
            return GOOD_EVAL_JSON

        MockAgent.return_value.invoke = AsyncMock(side_effect=mock_invoke)

        project = await orchestrator.start_project("LLM eval test")
        project.terminal_conditions = [TerminalCondition(description="Quality check", type="llm_eval")]
        await store.update(project)

        await orchestrator.approve_and_execute(project.id)

        updated = await store.get(project.id)
        assert updated.status == ProjectStatus.DONE
        assert updated.terminal_conditions[0].met is True
        assert updated.terminal_conditions[0].evaluation is not None
        assert updated.terminal_conditions[0].evaluation["rationale"] == "All outputs produced"

    @patch("roshni.agent.default.DefaultAgent")
    async def test_llm_eval_failure_not_met(self, MockAgent, orchestrator, store):
        call_count = 0

        async def mock_invoke(prompt, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return GOOD_PLAN_JSON
            return BAD_EVAL_JSON

        MockAgent.return_value.invoke = AsyncMock(side_effect=mock_invoke)

        project = await orchestrator.start_project("Bad eval test")
        project.terminal_conditions = [TerminalCondition(description="Quality check", type="llm_eval")]
        await store.update(project)

        await orchestrator.approve_and_execute(project.id)

        updated = await store.get(project.id)
        assert updated.status == ProjectStatus.REVIEWING
        assert updated.terminal_conditions[0].met is False

    @patch("roshni.agent.default.DefaultAgent")
    async def test_llm_eval_error_not_met(self, MockAgent, orchestrator, store):
        """If LLM eval crashes, condition is not met (safe default)."""
        call_count = 0

        async def mock_invoke(prompt, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return GOOD_PLAN_JSON
            raise RuntimeError("LLM is down")

        MockAgent.return_value.invoke = AsyncMock(side_effect=mock_invoke)

        project = await orchestrator.start_project("Eval crash test")
        project.terminal_conditions = [TerminalCondition(description="Quality check", type="llm_eval")]
        await store.update(project)

        await orchestrator.approve_and_execute(project.id)

        updated = await store.get(project.id)
        assert updated.status == ProjectStatus.REVIEWING
        assert updated.terminal_conditions[0].met is False

    @patch("roshni.agent.default.DefaultAgent")
    async def test_terminal_condition_events_recorded(self, MockAgent, orchestrator, store):
        call_count = 0

        async def mock_invoke(prompt, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return GOOD_PLAN_JSON
            return GOOD_EVAL_JSON

        MockAgent.return_value.invoke = AsyncMock(side_effect=mock_invoke)

        project = await orchestrator.start_project("TC events")
        project.terminal_conditions = [TerminalCondition(description="Check", type="llm_eval")]
        await store.update(project)

        await orchestrator.approve_and_execute(project.id)

        updated = await store.get(project.id)
        events_path = store.workspace_path(updated.id) / "events.ndjson"
        lines = [l for l in events_path.read_text().splitlines() if l.strip()]
        event_types = [json.loads(l)["type"] for l in lines]
        assert "terminal_condition.evaluated" in event_types


# ---------------------------------------------------------------------------
# steer
# ---------------------------------------------------------------------------


class TestSteer:
    @patch("roshni.agent.default.DefaultAgent")
    async def test_steer_records_journal_and_event(self, MockAgent, orchestrator, store):
        MockAgent.return_value.invoke = AsyncMock(return_value=GOOD_PLAN_JSON)
        project = await orchestrator.start_project("Steer test")

        await orchestrator.steer(project.id, "Focus on security first")

        updated = await store.get(project.id)
        assert any(j.action == "steering" and "security" in j.content for j in updated.journal)

        events_path = store.workspace_path(updated.id) / "events.ndjson"
        lines = [l for l in events_path.read_text().splitlines() if l.strip()]
        event_types = [json.loads(l)["type"] for l in lines]
        assert "project.steered" in event_types

    @patch("roshni.agent.default.DefaultAgent")
    async def test_steer_nonexistent_raises(self, MockAgent, orchestrator):
        with pytest.raises(ValueError, match="not found"):
            await orchestrator.steer("proj-99999-001", "something")


# ---------------------------------------------------------------------------
# reconcile
# ---------------------------------------------------------------------------


class TestReconcile:
    @patch("roshni.agent.default.DefaultAgent")
    async def test_reconcile_override(self, MockAgent, orchestrator, store):
        MockAgent.return_value.invoke = AsyncMock(return_value=GOOD_PLAN_JSON)
        project = await orchestrator.start_project("Reconcile test")

        await orchestrator.reconcile(project.id, accept_obsidian=False)

        updated = await store.get(project.id)
        assert any("override" in j.content for j in updated.journal)

        events_path = store.workspace_path(updated.id) / "events.ndjson"
        lines = [l for l in events_path.read_text().splitlines() if l.strip()]
        event_types = [json.loads(l)["type"] for l in lines]
        assert "conflict.reconciled" in event_types

    @patch("roshni.agent.default.DefaultAgent")
    async def test_reconcile_accept(self, MockAgent, orchestrator, store):
        MockAgent.return_value.invoke = AsyncMock(return_value=GOOD_PLAN_JSON)
        project = await orchestrator.start_project("Accept test")

        await orchestrator.reconcile(project.id, accept_obsidian=True)

        updated = await store.get(project.id)
        assert any("accept_obsidian" in j.content for j in updated.journal)

    @patch("roshni.agent.default.DefaultAgent")
    async def test_reconcile_nonexistent_raises(self, MockAgent, orchestrator):
        with pytest.raises(ValueError, match="not found"):
            await orchestrator.reconcile("proj-99999-001")


# ---------------------------------------------------------------------------
# advance
# ---------------------------------------------------------------------------

ADVANCE_PLAN_JSON = json.dumps(
    {
        "phases": [
            {
                "id": "phase-3",
                "name": "Follow-up Research",
                "description": "Continue with new research",
                "tasks": [{"id": "task-004", "description": "Research latest developments"}],
                "entry_criteria": ["Previous phases complete"],
                "exit_criteria": ["New findings documented"],
            }
        ]
    }
)


class TestAdvance:
    @patch("roshni.agent.default.DefaultAgent")
    async def test_advance_done_project(self, MockAgent, orchestrator, store):
        """Advancing a done project creates new phase and executes it."""
        call_count = 0

        async def mock_invoke(prompt, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return GOOD_PLAN_JSON
            return ADVANCE_PLAN_JSON

        MockAgent.return_value.invoke = AsyncMock(side_effect=mock_invoke)

        project = await orchestrator.start_project("Long-term goal")
        await orchestrator.approve_and_execute(project.id)

        # Verify it's done
        updated = await store.get(project.id)
        assert updated.status == ProjectStatus.DONE

        # Now advance it
        advanced = await orchestrator.advance(project.id, "Research latest developments")

        assert advanced.status == ProjectStatus.DONE  # completes the new phase too
        assert len(advanced.phases) == 3  # original 2 + 1 new
        assert advanced.phases[2].name == "Follow-up Research"
        assert any("advanced" in j.action for j in advanced.journal)

    @patch("roshni.agent.default.DefaultAgent")
    async def test_advance_reviewing_project(self, MockAgent, orchestrator, store):
        """Advancing a reviewing project (unmet conditions) creates new work."""
        call_count = 0

        async def mock_invoke(prompt, **kw):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return GOOD_PLAN_JSON if call_count == 1 else BAD_EVAL_JSON
            return ADVANCE_PLAN_JSON

        MockAgent.return_value.invoke = AsyncMock(side_effect=mock_invoke)

        project = await orchestrator.start_project("Conditional goal")
        project.terminal_conditions = [TerminalCondition(description="Quality check", type="llm_eval")]
        await store.update(project)

        await orchestrator.approve_and_execute(project.id)
        updated = await store.get(project.id)
        assert updated.status == ProjectStatus.REVIEWING

        # Advance to address unmet conditions
        advanced = await orchestrator.advance(project.id, "Fix quality issues")
        assert len(advanced.phases) == 3

    @patch("roshni.agent.default.DefaultAgent")
    async def test_advance_executing_steers(self, MockAgent, orchestrator, store):
        """Advancing an executing project just steers it."""
        MockAgent.return_value.invoke = AsyncMock(return_value=GOOD_PLAN_JSON)
        project = await orchestrator.start_project("Running project")

        # Manually set to executing
        project.status = ProjectStatus.EXECUTING
        await store.update(project)

        advanced = await orchestrator.advance(project.id, "Focus on X")
        # Should have steering journal entry
        assert any(j.action == "steering" for j in advanced.journal)

    @patch("roshni.agent.default.DefaultAgent")
    async def test_advance_nonexistent_raises(self, MockAgent, orchestrator):
        with pytest.raises(ValueError, match="not found"):
            await orchestrator.advance("proj-99999-001")

    @patch("roshni.agent.default.DefaultAgent")
    async def test_advance_cancelled_raises(self, MockAgent, orchestrator, store):
        MockAgent.return_value.invoke = AsyncMock(return_value=GOOD_PLAN_JSON)
        project = await orchestrator.start_project("Cancelled project")
        project.status = ProjectStatus.CANCELLED
        await store.update(project)

        with pytest.raises(ValueError, match="Cannot advance"):
            await orchestrator.advance(project.id)

    @patch("roshni.agent.default.DefaultAgent")
    async def test_advance_events_recorded(self, MockAgent, orchestrator, store):
        """Advance records project.advanced event."""
        call_count = 0

        async def mock_invoke(prompt, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return GOOD_PLAN_JSON
            return ADVANCE_PLAN_JSON

        MockAgent.return_value.invoke = AsyncMock(side_effect=mock_invoke)

        project = await orchestrator.start_project("Event test")
        await orchestrator.approve_and_execute(project.id)
        await orchestrator.advance(project.id, "Do more")

        updated = await store.get(project.id)
        events_path = store.workspace_path(updated.id) / "events.ndjson"
        lines = [l for l in events_path.read_text().splitlines() if l.strip()]
        event_types = [json.loads(l)["type"] for l in lines]
        assert "project.advanced" in event_types


# ---------------------------------------------------------------------------
# review_projects
# ---------------------------------------------------------------------------


class TestReviewProjects:
    @patch("roshni.agent.default.DefaultAgent")
    async def test_review_no_projects(self, MockAgent, orchestrator):
        result = await orchestrator.review_projects()
        assert "No projects" in result

    @patch("roshni.agent.default.DefaultAgent")
    async def test_review_returns_synthesis(self, MockAgent, orchestrator, store):
        """Review synthesizes across multiple projects."""
        call_count = 0

        async def mock_invoke(prompt, **kw):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return GOOD_PLAN_JSON
            return "Here is a synthesis of your projects..."

        MockAgent.return_value.invoke = AsyncMock(side_effect=mock_invoke)

        await orchestrator.start_project("Health tracking", tags=["health"])
        await orchestrator.start_project("Financial planning", tags=["finance"])

        result = await orchestrator.review_projects()
        assert "synthesis" in result.lower()

    @patch("roshni.agent.default.DefaultAgent")
    async def test_review_filters_by_tags(self, MockAgent, orchestrator, store):
        """Review respects tag filter."""
        call_count = 0

        async def mock_invoke(prompt, **kw):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return GOOD_PLAN_JSON
            return "Health project synthesis"

        MockAgent.return_value.invoke = AsyncMock(side_effect=mock_invoke)

        await orchestrator.start_project("Health tracking", tags=["health"])
        await orchestrator.start_project("Financial planning", tags=["finance"])

        await orchestrator.review_projects(tags=["health"])
        # Should have called the reviewer (3rd invoke call)
        assert call_count == 3

    @patch("roshni.agent.default.DefaultAgent")
    async def test_review_filters_by_query(self, MockAgent, orchestrator, store):
        """Review respects query filter."""
        call_count = 0

        async def mock_invoke(prompt, **kw):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return GOOD_PLAN_JSON
            return "Matching project synthesis"

        MockAgent.return_value.invoke = AsyncMock(side_effect=mock_invoke)

        await orchestrator.start_project("Health tracking", tags=["health"])
        await orchestrator.start_project("Financial planning", tags=["finance"])

        await orchestrator.review_projects(query="health")
        assert call_count == 3

    @patch("roshni.agent.default.DefaultAgent")
    async def test_review_no_matches(self, MockAgent, orchestrator, store):
        MockAgent.return_value.invoke = AsyncMock(return_value=GOOD_PLAN_JSON)

        await orchestrator.start_project("Health tracking", tags=["health"])

        result = await orchestrator.review_projects(query="nonexistent")
        assert "No projects matching" in result
