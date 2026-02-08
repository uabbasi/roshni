"""Tests for ToolDefinition.from_function() and retry logic."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from roshni.agent.tools import ToolDefinition


# ------------------------------------------------------------------
# from_function()
# ------------------------------------------------------------------


class TestFromFunction:
    def test_requires_pydantic(self):
        pydantic = pytest.importorskip("pydantic")

        class EchoArgs(pydantic.BaseModel):
            text: str
            loud: bool = False

        def echo_fn(text: str, loud: bool = False) -> str:
            return text.upper() if loud else text

        tool = ToolDefinition.from_function(
            func=echo_fn,
            name="echo",
            description="Echo the input",
            args_schema=EchoArgs,
        )

        assert tool.name == "echo"
        assert tool.description == "Echo the input"
        assert tool.permission == "read"
        assert tool.requires_approval is None

        # Schema should have properties, no title at top or property level
        assert "title" not in tool.parameters
        assert "$defs" not in tool.parameters
        assert "text" in tool.parameters["properties"]
        assert "title" not in tool.parameters["properties"]["text"]

        # Actually callable
        assert tool.execute({"text": "hello"}) == "hello"
        assert tool.execute({"text": "hello", "loud": True}) == "HELLO"

    def test_write_permission(self):
        pydantic = pytest.importorskip("pydantic")

        class WriteArgs(pydantic.BaseModel):
            path: str
            data: str

        tool = ToolDefinition.from_function(
            func=lambda path, data: "ok",
            name="write_file",
            description="Write a file",
            args_schema=WriteArgs,
            permission="write",
            requires_approval=True,
        )

        assert tool.permission == "write"
        assert tool.requires_approval is True
        assert tool.needs_approval() is True

    def test_litellm_schema_roundtrip(self):
        pydantic = pytest.importorskip("pydantic")

        class Args(pydantic.BaseModel):
            query: str

        tool = ToolDefinition.from_function(
            func=lambda query: query,
            name="search",
            description="Search for things",
            args_schema=Args,
        )

        schema = tool.to_litellm_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        assert "query" in schema["function"]["parameters"]["properties"]


# ------------------------------------------------------------------
# Retry logic
# ------------------------------------------------------------------


class TestExecuteRetry:
    def test_no_retry_on_success(self):
        calls = []

        def fn(x: str) -> str:
            calls.append(1)
            return x

        tool = ToolDefinition(
            name="t", description="t", parameters={}, function=fn
        )
        result = tool.execute({"x": "ok"})
        assert result == "ok"
        assert len(calls) == 1

    @patch("roshni.agent.tools._time.sleep")
    def test_retries_transient_errors(self, mock_sleep):
        attempts = []

        def fn() -> str:
            attempts.append(1)
            if len(attempts) < 3:
                raise ConnectionError("network blip")
            return "recovered"

        tool = ToolDefinition(
            name="flaky", description="flaky", parameters={}, function=fn
        )
        result = tool.execute({})
        assert result == "recovered"
        assert len(attempts) == 3
        # Verify backoff: sleep(1), sleep(2)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1)
        mock_sleep.assert_any_call(2)

    @patch("roshni.agent.tools._time.sleep")
    def test_transient_exhaustion(self, mock_sleep):
        def fn() -> str:
            raise TimeoutError("always times out")

        tool = ToolDefinition(
            name="broken", description="broken", parameters={}, function=fn
        )
        result = tool.execute({}, max_attempts=2)
        assert "failed after 2 attempts" in result
        assert mock_sleep.call_count == 1

    def test_non_transient_fails_immediately(self):
        calls = []

        def fn() -> str:
            calls.append(1)
            raise ValueError("bad input")

        tool = ToolDefinition(
            name="bad", description="bad", parameters={}, function=fn
        )
        result = tool.execute({})
        assert "Error executing bad" in result
        assert "bad input" in result
        assert len(calls) == 1  # No retry

    def test_json_parse_error(self):
        tool = ToolDefinition(
            name="t", description="t", parameters={}, function=lambda: "ok"
        )
        result = tool.execute("not json{{{")
        assert "could not parse" in result

    @patch("roshni.agent.tools._time.sleep")
    def test_os_error_is_transient(self, mock_sleep):
        attempts = []

        def fn() -> str:
            attempts.append(1)
            if len(attempts) < 2:
                raise OSError("disk full")
            return "ok"

        tool = ToolDefinition(
            name="disk", description="disk", parameters={}, function=fn
        )
        result = tool.execute({})
        assert result == "ok"
        assert len(attempts) == 2
