"""Apple Health export collector (HealthKit-compatible import path).

Reads Apple Health XML exports (``export.xml``) and aggregates daily metrics.
This is the most portable way to integrate HealthKit data outside iOS.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from roshni.health.collector import BaseCollector
from roshni.health.models import ActivityRecord, BodyRecord, DailyHealth, HeartRateRecord, SleepRecord


class AppleHealthExportCollector(BaseCollector):
    """Collect daily health metrics from an Apple Health export XML file."""

    name = "apple_health_export"

    def __init__(self, export_path: str, **config: Any):
        super().__init__(export_path=export_path, **config)
        self.export_path = Path(export_path).expanduser()

    def validate(self) -> bool:
        if not self.export_path.exists() or not self.export_path.is_file():
            return False
        try:
            ET.parse(self.export_path)
            return True
        except Exception:
            return False

    def get_config_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "export_path": {
                    "type": "string",
                    "description": "Path to Apple Health export.xml file",
                }
            },
            "required": ["export_path"],
        }

    def collect(self, start_date: date, end_date: date) -> list[DailyHealth]:
        if end_date < start_date:
            return []
        if not self.validate():
            raise FileNotFoundError(f"Invalid Apple Health export: {self.export_path}")

        root = ET.parse(self.export_path).getroot()

        daily: dict[date, DailyHealth] = {}
        resting_hr_sums: dict[date, tuple[float, int]] = {}
        weight_latest: dict[date, tuple[datetime, float]] = {}
        sleep_minutes: dict[date, float] = {}

        for rec in root.iter("Record"):
            record_type = rec.attrib.get("type", "")
            start = self._parse_health_datetime(rec.attrib.get("startDate", ""))
            end = self._parse_health_datetime(rec.attrib.get("endDate", ""))
            if start is None:
                continue

            day = start.date()
            if day < start_date or day > end_date:
                continue

            value_raw = rec.attrib.get("value", "")
            value_num = self._to_float(value_raw)
            unit = (rec.attrib.get("unit", "") or "").strip()

            item = daily.setdefault(day, DailyHealth(date=day))

            if record_type == "HKQuantityTypeIdentifierStepCount":
                item.activity = item.activity or ActivityRecord(date=day)
                current = item.activity.steps or 0
                item.activity.steps = current + round(value_num or 0.0)

            elif record_type == "HKQuantityTypeIdentifierDistanceWalkingRunning":
                item.activity = item.activity or ActivityRecord(date=day)
                km = self._to_km(value_num, unit)
                current = item.activity.distance_km or 0.0
                item.activity.distance_km = current + km

            elif record_type == "HKQuantityTypeIdentifierActiveEnergyBurned":
                item.activity = item.activity or ActivityRecord(date=day)
                kcal = self._to_kcal(value_num, unit)
                current = item.activity.calories_burned or 0
                item.activity.calories_burned = current + round(kcal)

            elif record_type == "HKQuantityTypeIdentifierFlightsClimbed":
                item.activity = item.activity or ActivityRecord(date=day)
                current = item.activity.floors or 0
                item.activity.floors = current + round(value_num or 0.0)

            elif record_type == "HKQuantityTypeIdentifierRestingHeartRate":
                item.heart_rate = item.heart_rate or HeartRateRecord(date=day)
                current_sum, current_count = resting_hr_sums.get(day, (0.0, 0))
                resting_hr_sums[day] = (current_sum + (value_num or 0.0), current_count + 1)

            elif record_type == "HKQuantityTypeIdentifierHeartRate":
                item.heart_rate = item.heart_rate or HeartRateRecord(date=day)
                hr = round(value_num or 0.0)
                if item.heart_rate.min_hr is None or hr < item.heart_rate.min_hr:
                    item.heart_rate.min_hr = hr
                if item.heart_rate.max_hr is None or hr > item.heart_rate.max_hr:
                    item.heart_rate.max_hr = hr

            elif record_type == "HKQuantityTypeIdentifierBodyMass":
                item.body = item.body or BodyRecord(date=day)
                kg = self._to_kg(value_num, unit)
                prev = weight_latest.get(day)
                if prev is None or (end and end > prev[0]):
                    weight_latest[day] = (end or start, kg)

            elif record_type == "HKCategoryTypeIdentifierSleepAnalysis":
                if self._is_sleep_asleep(value_raw) and end is not None and end > start:
                    minutes = (end - start).total_seconds() / 60.0
                    sleep_minutes[day] = sleep_minutes.get(day, 0.0) + minutes

        # Finalize aggregates
        for day, (total, count) in resting_hr_sums.items():
            item = daily.setdefault(day, DailyHealth(date=day))
            item.heart_rate = item.heart_rate or HeartRateRecord(date=day)
            if count > 0:
                item.heart_rate.resting_hr = round(total / count)

        for day, (_when, kg) in weight_latest.items():
            item = daily.setdefault(day, DailyHealth(date=day))
            item.body = item.body or BodyRecord(date=day)
            item.body.weight_kg = round(kg, 2)

        for day, minutes in sleep_minutes.items():
            item = daily.setdefault(day, DailyHealth(date=day))
            item.sleep = item.sleep or SleepRecord(date=day)
            item.sleep.total_minutes = round((item.sleep.total_minutes or 0.0) + minutes, 1)

        return [daily[d] for d in sorted(daily.keys())]

    @staticmethod
    def _parse_health_datetime(value: str) -> datetime | None:
        if not value:
            return None
        formats = [
            "%Y-%m-%d %H:%M:%S %z",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None

    @staticmethod
    def _to_float(value: str) -> float | None:
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _to_km(value: float | None, unit: str) -> float:
        if value is None:
            return 0.0
        u = unit.lower()
        if u == "km":
            return value
        if u in {"m", "meter", "meters"}:
            return value / 1000.0
        if u in {"mi", "mile", "miles"}:
            return value * 1.60934
        return value

    @staticmethod
    def _to_kcal(value: float | None, unit: str) -> float:
        if value is None:
            return 0.0
        u = unit.lower()
        if u in {"kcal", "calorie", "calories"}:
            return value
        if u in {"cal", "smallcalorie"}:
            return value / 1000.0
        return value

    @staticmethod
    def _to_kg(value: float | None, unit: str) -> float:
        if value is None:
            return 0.0
        u = unit.lower()
        if u == "kg":
            return value
        if u in {"lb", "lbs", "pound", "pounds"}:
            return value * 0.45359237
        if u in {"g", "gram", "grams"}:
            return value / 1000.0
        return value

    @staticmethod
    def _is_sleep_asleep(value: str) -> bool:
        normalized = (value or "").lower()
        asleep_terms = [
            "asleep",
            "asleepcore",
            "asleepdeep",
            "asleeprem",
            "asleepunspecified",
        ]
        return any(term in normalized for term in asleep_terms)
