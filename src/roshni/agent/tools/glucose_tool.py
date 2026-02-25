"""Libre/CGM ToolDefinition factory."""

from __future__ import annotations

from collections.abc import Callable

from roshni.agent.tools import ToolDefinition


def create_glucose_tools(get_glucose_now_fn: Callable[..., str]) -> list[ToolDefinition]:
    """Create a ``get_glucose_now`` tool around an injected callable."""
    return [
        ToolDefinition(
            name="get_glucose_now",
            description=(
                "Get current continuous glucose monitor (CGM) reading from LibreView. "
                "Returns: current glucose value (mg/dL), trend arrow "
                "(falling_fast/falling/stable/rising/rising_fast), and recent ~12h stats "
                "(mean, min, max, time-in-range %). "
                "Use for glucose check-ins, blood sugar questions, or fasting queries."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            function=get_glucose_now_fn,
        )
    ]


__all__ = ["create_glucose_tools"]

