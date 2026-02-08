"""Tests for health.models — data structures and ColumnType."""

from datetime import date

from roshni.health.models import (
    ActivityRecord,
    BodyRecord,
    ColumnType,
    DailyHealth,
    HeartRateRecord,
    SleepRecord,
)


class TestColumnType:
    def test_values(self):
        assert ColumnType.INTEGER == "INTEGER"
        assert ColumnType.FLOAT == "FLOAT"
        assert ColumnType.TEXT == "TEXT"
        assert ColumnType.DATE == "DATE"
        assert ColumnType.DATETIME == "DATETIME"

    def test_is_string_enum(self):
        assert isinstance(ColumnType.INTEGER, str)


class TestSleepRecord:
    def test_defaults(self):
        rec = SleepRecord(date=date(2025, 1, 1))
        assert rec.total_minutes is None
        assert rec.raw == {}

    def test_with_data(self):
        rec = SleepRecord(date=date(2025, 1, 1), total_minutes=420, deep_minutes=90)
        assert rec.total_minutes == 420


class TestActivityRecord:
    def test_defaults(self):
        rec = ActivityRecord(date=date(2025, 1, 1))
        assert rec.steps is None

    def test_with_data(self):
        rec = ActivityRecord(date=date(2025, 1, 1), steps=10000, calories_burned=2500)
        assert rec.steps == 10000


class TestDailyHealth:
    def test_empty(self):
        d = DailyHealth(date=date(2025, 6, 15))
        assert d.sleep is None
        assert d.extra == {}

    def test_with_records(self):
        d = DailyHealth(
            date=date(2025, 6, 15),
            sleep=SleepRecord(date=date(2025, 6, 15), total_minutes=450),
            activity=ActivityRecord(date=date(2025, 6, 15), steps=8000),
            heart_rate=HeartRateRecord(date=date(2025, 6, 15), resting_hr=62),
            body=BodyRecord(date=date(2025, 6, 15), weight_kg=75.5),
            mood="good",
        )
        assert d.sleep.total_minutes == 450
        assert d.activity.steps == 8000
        assert d.heart_rate.resting_hr == 62
        assert d.body.weight_kg == 75.5
        assert d.mood == "good"

    def test_extra_fields(self):
        d = DailyHealth(date=date(2025, 1, 1), extra={"spo2": 97, "hrv": 45})
        assert d.extra["spo2"] == 97
