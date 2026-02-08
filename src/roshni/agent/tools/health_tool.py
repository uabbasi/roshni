"""Health tools backed by Apple Health export data."""

from __future__ import annotations

from datetime import date, datetime

from roshni.agent.tools import ToolDefinition
from roshni.health.plugins.apple_health_export import AppleHealthExportCollector


def _parse_date_or_none(value: str) -> date | None:
    v = (value or "").strip()
    if not v:
        return None
    try:
        return datetime.strptime(v, "%Y-%m-%d").date()
    except ValueError:
        return None


def _format_day(item) -> str:
    parts = [item.date.isoformat()]
    if item.activity:
        if item.activity.steps is not None:
            parts.append(f"steps={item.activity.steps}")
        if item.activity.distance_km is not None:
            parts.append(f"distance_km={round(item.activity.distance_km, 2)}")
        if item.activity.calories_burned is not None:
            parts.append(f"active_kcal={item.activity.calories_burned}")
    if item.sleep and item.sleep.total_minutes is not None:
        parts.append(f"sleep_min={round(item.sleep.total_minutes, 1)}")
    if item.heart_rate and item.heart_rate.resting_hr is not None:
        parts.append(f"resting_hr={item.heart_rate.resting_hr}")
    if item.body and item.body.weight_kg is not None:
        parts.append(f"weight_kg={item.body.weight_kg}")
    return "- " + ", ".join(parts)


def _get_health_summary(export_path: str, start_date: str = "", end_date: str = "") -> str:
    start = _parse_date_or_none(start_date)
    end = _parse_date_or_none(end_date)

    if start_date and start is None:
        return "Invalid start_date. Use YYYY-MM-DD."
    if end_date and end is None:
        return "Invalid end_date. Use YYYY-MM-DD."

    today = date.today()
    if start is None and end is None:
        start = today
        end = today
    elif start is None and end is not None:
        start = end
    elif start is not None and end is None:
        end = start

    if start is None or end is None:
        return "Unable to determine date range."

    collector = AppleHealthExportCollector(export_path=export_path)
    try:
        items = collector.collect(start, end)
    except FileNotFoundError:
        return f"Health export file not found or invalid: {export_path}"
    except Exception as e:
        return f"Failed to load health data: {e}"

    if not items:
        return f"No health data found from {start.isoformat()} to {end.isoformat()}."

    lines = [f"Health summary ({start.isoformat()} to {end.isoformat()}):"]

    total_steps = 0
    total_sleep = 0.0
    hr_values: list[int] = []
    for item in items:
        lines.append(_format_day(item))
        if item.activity and item.activity.steps:
            total_steps += item.activity.steps
        if item.sleep and item.sleep.total_minutes:
            total_sleep += item.sleep.total_minutes
        if item.heart_rate and item.heart_rate.resting_hr:
            hr_values.append(item.heart_rate.resting_hr)

    lines.append("")
    lines.append(f"Days: {len(items)}")
    lines.append(f"Total steps: {total_steps}")
    lines.append(f"Total sleep (minutes): {round(total_sleep, 1)}")
    if hr_values:
        lines.append(f"Average resting HR: {round(sum(hr_values) / len(hr_values), 1)}")

    return "\n".join(lines)


def create_health_tools(export_path: str) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="get_health_summary",
            description=(
                "Get a daily summary from Apple Health export data for a date or date range. "
                "Dates must be YYYY-MM-DD."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"},
                },
                "required": [],
            },
            function=lambda start_date="", end_date="": _get_health_summary(export_path, start_date, end_date),
            permission="read",
        )
    ]
