"""Python execution ToolDefinition factory (adapter-based)."""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, Field

from roshni.agent.tools import ToolDefinition


class PythonExecInput(BaseModel):
    code: str = Field(description="Python code to execute. Must set 'result' variable.")


def create_python_exec_tools(execute_python_fn: Callable[..., str]) -> list[ToolDefinition]:
    return [
        ToolDefinition.from_function(
            func=execute_python_fn,
            name="execute_python",
            description=(
                "Execute Python code for calculations. Available: math, numpy, pandas, datetime, statistics. "
                "MUST set 'result' variable. Example: 'result = sum([7.5, 6.2, 8.0]) / 3' for average."
            ),
            args_schema=PythonExecInput,
        )
    ]


__all__ = ["PythonExecInput", "create_python_exec_tools"]
