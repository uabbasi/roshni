"""
Health data models.

Generic building blocks for health data pipelines.  These are intentionally
minimal — consumers extend them with domain-specific fields.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Any

# ── Column types (used by BaseETL for schema enforcement) ────────────


class ColumnType(StrEnum):
    """SQLite-compatible column types for schema definitions."""

    INTEGER = "INTEGER"
    FLOAT = "FLOAT"
    TEXT = "TEXT"
    DATE = "DATE"
    DATETIME = "DATETIME"


# ── Generic health records ───────────────────────────────────────────


@dataclass
class SleepRecord:
    """A single night's sleep data."""

    date: date
    total_minutes: float | None = None
    deep_minutes: float | None = None
    light_minutes: float | None = None
    rem_minutes: float | None = None
    wake_minutes: float | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    efficiency: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActivityRecord:
    """A single day's activity data."""

    date: date
    steps: int | None = None
    calories_burned: int | None = None
    active_minutes: int | None = None
    distance_km: float | None = None
    floors: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class HeartRateRecord:
    """A single day's heart rate data."""

    date: date
    resting_hr: int | None = None
    min_hr: int | None = None
    max_hr: int | None = None
    zones: dict[str, int] = field(default_factory=dict)  # zone_name -> minutes
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class BodyRecord:
    """Body measurements for a day."""

    date: date
    weight_kg: float | None = None
    bmi: float | None = None
    body_fat_pct: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class DailyHealth:
    """Aggregated daily health snapshot.

    The generic fields cover common metrics.  Use ``extra`` for
    source-specific data that doesn't fit the standard fields.
    """

    date: date
    sleep: SleepRecord | None = None
    activity: ActivityRecord | None = None
    heart_rate: HeartRateRecord | None = None
    body: BodyRecord | None = None
    mood: str | None = None
    notes: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
