"""Tests for workflow data models — no LLM, fast."""

import threading

import pytest

from roshni.agent.workflow.models import (
    VALID_TRANSITIONS,
    Budget,
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


class TestProjectStatus:
    def test_all_values(self):
        values = {s.value for s in ProjectStatus}
        assert "planning" in values
        assert "executing" in values
        assert "done" in values
        assert "cancelled" in values

    def test_terminal_statuses(self):
        from roshni.agent.workflow.models import TERMINAL_STATUSES

        assert ProjectStatus.CANCELLED in TERMINAL_STATUSES
        # done is NOT terminal — projects can be advanced
        assert ProjectStatus.DONE not in TERMINAL_STATUSES
        assert ProjectStatus.EXECUTING not in TERMINAL_STATUSES


class TestValidTransitions:
    def test_planning_to_awaiting(self):
        validate_transition(ProjectStatus.PLANNING, ProjectStatus.AWAITING_APPROVAL)

    def test_awaiting_to_executing(self):
        validate_transition(ProjectStatus.AWAITING_APPROVAL, ProjectStatus.EXECUTING)

    def test_executing_to_paused(self):
        validate_transition(ProjectStatus.EXECUTING, ProjectStatus.PAUSED)

    def test_paused_to_executing(self):
        validate_transition(ProjectStatus.PAUSED, ProjectStatus.EXECUTING)

    def test_executing_to_cancelled(self):
        validate_transition(ProjectStatus.EXECUTING, ProjectStatus.CANCELLED)

    def test_failed_to_planning(self):
        validate_transition(ProjectStatus.FAILED, ProjectStatus.PLANNING)

    def test_done_to_planning(self):
        """Done projects can be advanced back to planning."""
        validate_transition(ProjectStatus.DONE, ProjectStatus.PLANNING)

    def test_done_to_executing_invalid(self):
        """Done projects cannot jump directly to executing."""
        with pytest.raises(ValueError, match="Invalid transition"):
            validate_transition(ProjectStatus.DONE, ProjectStatus.EXECUTING)

    def test_reviewing_to_planning(self):
        """Reviewing projects can be replanned to address unmet conditions."""
        validate_transition(ProjectStatus.REVIEWING, ProjectStatus.PLANNING)

    def test_cancelled_is_terminal(self):
        with pytest.raises(ValueError, match="Invalid transition"):
            validate_transition(ProjectStatus.CANCELLED, ProjectStatus.PLANNING)

    def test_invalid_transition(self):
        with pytest.raises(ValueError, match="Invalid transition"):
            validate_transition(ProjectStatus.PLANNING, ProjectStatus.DONE)


class TestBudget:
    def test_not_exhausted_initially(self):
        b = Budget()
        assert not b.exhausted
        assert b.remaining_fraction() == 1.0

    def test_exhausted_by_cost(self):
        b = Budget(max_cost_usd=1.0)
        b.record_call(1.0)
        assert b.exhausted

    def test_exhausted_by_calls(self):
        b = Budget(max_llm_calls=2)
        b.record_call(0.0)
        b.record_call(0.0)
        assert b.exhausted

    def test_exhausted_by_wall_time(self):
        b = Budget(max_wall_seconds=10.0, wall_seconds_used=10.0)
        assert b.exhausted

    def test_remaining_fraction(self):
        b = Budget(max_cost_usd=10.0, max_llm_calls=100)
        b.record_call(5.0)
        # Cost: 50% remaining, calls: 99% remaining -> min = 50%
        assert b.remaining_fraction() == pytest.approx(0.5, abs=0.01)

    def test_record_call_thread_safe(self):
        b = Budget(max_cost_usd=1000.0, max_llm_calls=10000)
        errors = []

        def record_many():
            try:
                for _ in range(100):
                    b.record_call(0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_many) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert b.llm_calls_used == 1000
        assert b.cost_used_usd == pytest.approx(10.0, abs=0.01)

    def test_to_dict_from_dict_roundtrip(self):
        b = Budget(max_cost_usd=3.0, max_llm_calls=50)
        b.record_call(1.5)
        d = b.to_dict()
        b2 = Budget.from_dict(d)
        assert b2.max_cost_usd == 3.0
        assert b2.cost_used_usd == pytest.approx(1.5)
        assert b2.llm_calls_used == 1


class TestTaskSpec:
    def test_stable_id(self):
        t = TaskSpec(id="task-001", description="test")
        assert t.id == "task-001"

    def test_default_max_attempts(self):
        t = TaskSpec(id="task-001", description="test")
        assert t.max_attempts == 1

    def test_unique_ids(self):
        tasks = [TaskSpec(id=f"task-{i:03d}", description=f"Task {i}") for i in range(5)]
        ids = [t.id for t in tasks]
        assert len(ids) == len(set(ids))


class TestTerminalCondition:
    def test_types(self):
        for tc_type in ("artifact_exists", "phase_count", "llm_eval", "check_fn"):
            tc = TerminalCondition(description="test", type=tc_type)
            assert tc.type == tc_type
            assert not tc.met


class TestWorkflowEvent:
    def test_monotonic_seq(self):
        events = [
            WorkflowEvent(
                event_id=f"evt-{i:06d}",
                seq=i,
                type="test",
                timestamp="2026-01-01T00:00:00",
                actor="test",
            )
            for i in range(1, 6)
        ]
        seqs = [e.seq for e in events]
        assert seqs == list(range(1, 6))
        assert all(seqs[i] < seqs[i + 1] for i in range(len(seqs) - 1))


class TestProjectSerialization:
    def test_roundtrip(self):
        project = Project(
            id="proj-20260215-001",
            goal="Test project",
            phases=[
                Phase(
                    id="phase-1",
                    name="Phase 1",
                    tasks=[TaskSpec(id="task-001", description="Do something")],
                    entry_criteria=[PhaseEntry(description="Ready")],
                    exit_criteria=[PhaseEntry(description="Done", met=True)],
                )
            ],
            terminal_conditions=[TerminalCondition(description="All done", type="phase_count")],
            tags=["test"],
        )
        project.budget.record_call(0.5)

        d = project_to_dict(project)
        p2 = project_from_dict(d)

        assert p2.id == project.id
        assert p2.goal == project.goal
        assert len(p2.phases) == 1
        assert p2.phases[0].tasks[0].id == "task-001"
        assert p2.budget.cost_used_usd == pytest.approx(0.5)
        assert p2.tags == ["test"]

    def test_plan_hash_stability(self):
        project = Project(
            id="test",
            goal="Test",
            phases=[Phase(id="p1", name="P1", tasks=[TaskSpec(id="t1", description="Task 1")])],
        )
        h1 = compute_plan_hash(project)
        h2 = compute_plan_hash(project)
        assert h1 == h2

    def test_plan_hash_changes_with_tasks(self):
        p1 = Project(
            id="test",
            goal="Test",
            phases=[Phase(id="p1", name="P1", tasks=[TaskSpec(id="t1", description="Task 1")])],
        )
        p2 = Project(
            id="test",
            goal="Test",
            phases=[Phase(id="p1", name="P1", tasks=[TaskSpec(id="t1", description="Task 2")])],
        )
        assert compute_plan_hash(p1) != compute_plan_hash(p2)


@pytest.mark.smoke
class TestSmoke:
    """Smoke tests for workflow models — fast, no LLM."""

    def test_project_creation(self):
        p = Project(id="test-001", goal="Test goal")
        assert p.status == ProjectStatus.PLANNING
        assert not p.budget.exhausted

    def test_phase_creation(self):
        phase = Phase(id="p1", name="Phase 1")
        assert phase.status == PhaseStatus.PENDING

    def test_transitions_map_complete(self):
        for status in ProjectStatus:
            assert status in VALID_TRANSITIONS
