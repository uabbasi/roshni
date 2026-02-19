"""Tests for workflow tools — tool count, names, permissions."""

from unittest.mock import MagicMock

import pytest

from roshni.agent.workflow.tools import create_workflow_tools


@pytest.fixture
def mock_store():
    return MagicMock()


@pytest.fixture
def mock_orchestrator():
    return MagicMock()


class TestWorkflowTools:
    def test_tool_count(self, mock_store, mock_orchestrator):
        tools = create_workflow_tools(mock_store, mock_orchestrator)
        assert len(tools) == 10

    def test_tool_names(self, mock_store, mock_orchestrator):
        tools = create_workflow_tools(mock_store, mock_orchestrator)
        names = {t.name for t in tools}
        expected = {
            "create_project",
            "check_project",
            "steer_project",
            "approve_project",
            "pause_project",
            "resume_project",
            "cancel_project",
            "reconcile_project",
            "advance_project",
            "review_projects",
        }
        assert names == expected

    def test_read_permissions(self, mock_store, mock_orchestrator):
        tools = create_workflow_tools(mock_store, mock_orchestrator)
        read_tools = [t for t in tools if t.permission == "read"]
        assert len(read_tools) == 2
        read_names = {t.name for t in read_tools}
        assert read_names == {"check_project", "review_projects"}

    def test_write_permissions(self, mock_store, mock_orchestrator):
        tools = create_workflow_tools(mock_store, mock_orchestrator)
        write_tools = [t for t in tools if t.permission == "write"]
        assert len(write_tools) == 8

    def test_cancel_requires_approval(self, mock_store, mock_orchestrator):
        tools = create_workflow_tools(mock_store, mock_orchestrator)
        cancel = next(t for t in tools if t.name == "cancel_project")
        assert cancel.requires_approval is True
        assert cancel.needs_approval()

    def test_tools_have_parameters(self, mock_store, mock_orchestrator):
        tools = create_workflow_tools(mock_store, mock_orchestrator)
        for tool in tools:
            assert tool.parameters is not None
            assert "type" in tool.parameters

    def test_tools_have_descriptions(self, mock_store, mock_orchestrator):
        tools = create_workflow_tools(mock_store, mock_orchestrator)
        for tool in tools:
            assert tool.description
            assert len(tool.description) > 10

    def test_litellm_schema_generation(self, mock_store, mock_orchestrator):
        """Verify tools can be converted to litellm format."""
        tools = create_workflow_tools(mock_store, mock_orchestrator)
        for tool in tools:
            schema = tool.to_litellm_schema()
            assert schema["type"] == "function"
            assert "name" in schema["function"]
