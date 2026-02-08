"""
Health data collection and ETL framework.

Core module (no heavy deps).  Install extras for specific features:
  - ``roshni[health]``  — pandas, requests (BaseETL, data processing)
  - ``roshni[fitbit]``  — Fitbit API collector
"""

from .collector import BaseCollector, HealthCollector
from .models import (
    ActivityRecord,
    BodyRecord,
    ColumnType,
    DailyHealth,
    HeartRateRecord,
    SleepRecord,
)
from .registry import HealthCollectorRegistry

__all__ = [
    "ActivityRecord",
    "BaseCollector",
    "BodyRecord",
    "ColumnType",
    "DailyHealth",
    "HealthCollector",
    "HealthCollectorRegistry",
    "HeartRateRecord",
    "SleepRecord",
]
