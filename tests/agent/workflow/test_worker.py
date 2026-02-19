"""Tests for WorkerPool — tool allowlist, budget enforcement."""

import pytest

from roshni.agent.workflow.models import Budget, TaskSpec
from roshni.agent.workflow.worker import ToolPolicyViolation, WorkerResult, _filter_tools_by_allowlist


class MockToolDefinition:
    """Minimal mock for ToolDefinition."""

    def __init__(self, name):
        self.name = name


class TestToolAllowlist:
    def test_empty_allowlist_returns_all(self):
        tools = [MockToolDefinition("a"), MockToolDefinition("b")]
        result = _filter_tools_by_allowlist(tools, [])
        assert len(result) == 2

    def test_allowlist_filters(self):
        tools = [MockToolDefinition("a"), MockToolDefinition("b"), MockToolDefinition("c")]
        result = _filter_tools_by_allowlist(tools, ["a", "c"])
        assert len(result) == 2
        assert {t.name for t in result} == {"a", "c"}

    def test_allowlist_with_unknown_names(self):
        tools = [MockToolDefinition("a")]
        result = _filter_tools_by_allowlist(tools, ["a", "nonexistent"])
        assert len(result) == 1
        assert result[0].name == "a"


class TestWorkerResult:
    def test_success_result(self):
        task = TaskSpec(id="task-001", description="Test")
        result = WorkerResult(
            worker_id="worker-abc123",
            task=task,
            response="Done",
            success=True,
            duration=1.5,
        )
        assert result.success
        assert result.worker_id == "worker-abc123"

    def test_failure_result(self):
        task = TaskSpec(id="task-001", description="Test")
        result = WorkerResult(
            worker_id="worker-abc123",
            task=task,
            response="",
            success=False,
            error="Budget exhausted",
        )
        assert not result.success
        assert "Budget" in result.error


class TestBudgetEnforcement:
    def test_exhausted_budget_blocks_spawn(self):
        """A project with exhausted budget should refuse new workers."""
        budget = Budget(max_cost_usd=1.0)
        budget.record_call(1.0)
        assert budget.exhausted
        # The actual spawn check happens in WorkerPool.spawn_worker,
        # but we can verify the budget state that drives it.

    def test_mid_loop_exhaustion(self):
        """Budget exhausting mid-loop should be detectable."""
        budget = Budget(max_llm_calls=2)
        budget.record_call(0.01)
        assert not budget.exhausted
        budget.record_call(0.01)
        assert budget.exhausted


class TestToolPolicyViolation:
    def test_exception(self):
        with pytest.raises(ToolPolicyViolation):
            raise ToolPolicyViolation("Tool 'admin_tool' not in allowlist")
