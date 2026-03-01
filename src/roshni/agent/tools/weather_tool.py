"""Weather ToolDefinition factory."""

from __future__ import annotations

from collections.abc import Callable

from roshni.agent.tools import ToolDefinition


def create_weather_tools(get_weather_now_fn: Callable[..., str]) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="get_weather_now",
            description=(
                "Get current weather and hourly forecast. Returns: temperature, "
                "conditions, wind speed, humidity, today's high/low, precipitation "
                "probability, sunrise/sunset times, 6-hour hourly forecast with UV index, "
                "and outdoor activity assessment. "
                "Use for weather questions, outdoor activity planning, or clothing suggestions."
            ),
            parameters={"type": "object", "properties": {}, "required": []},
            function=get_weather_now_fn,
        )
    ]


__all__ = ["create_weather_tools"]
