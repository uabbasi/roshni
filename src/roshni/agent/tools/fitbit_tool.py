"""Fitbit ToolDefinition factory.

This adapter is intentionally thin: the caller supplies the execution function
so app layers can migrate incrementally while keeping existing auth/config paths.
"""

from __future__ import annotations

from collections.abc import Callable

from roshni.agent.tools import ToolDefinition


def create_fitbit_tools(get_fitbit_data_fn: Callable[..., str]) -> list[ToolDefinition]:
    """Create a ``get_fitbit_data`` tool around an injected callable."""
    return [
        ToolDefinition(
            name="get_fitbit_data",
            description=(
                "Get Fitbit health data for today or a specific date. Returns: steps, active minutes, "
                "calories, heart rate zones, resting HR, sleep (hours, stages, efficiency), HRV, SpO2, "
                "breathing rate. Step pace assessment is included for today only. "
                "Use for wellness check-ins, step queries, or comparing days."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date to fetch: 'today' (default), 'yesterday', or YYYY-MM-DD.",
                    }
                },
                "required": [],
            },
            function=get_fitbit_data_fn,
        )
    ]


__all__ = ["create_fitbit_tools"]
