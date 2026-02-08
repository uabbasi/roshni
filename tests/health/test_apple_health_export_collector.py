"""Tests for AppleHealthExportCollector."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from roshni.health.plugins.apple_health_export import AppleHealthExportCollector


def _write_export(path: Path) -> None:
    xml = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<HealthData>
  <Record
    type=\"HKQuantityTypeIdentifierStepCount\"
    sourceName=\"iPhone\"
    unit=\"count\"
    value=\"1000\"
    startDate=\"2026-01-01 08:00:00 -0800\"
    endDate=\"2026-01-01 08:05:00 -0800\"
  />
  <Record
    type=\"HKQuantityTypeIdentifierStepCount\"
    sourceName=\"iPhone\"
    unit=\"count\"
    value=\"500\"
    startDate=\"2026-01-01 18:00:00 -0800\"
    endDate=\"2026-01-01 18:05:00 -0800\"
  />
  <Record
    type=\"HKQuantityTypeIdentifierDistanceWalkingRunning\"
    unit=\"m\"
    value=\"2400\"
    startDate=\"2026-01-01 10:00:00 -0800\"
    endDate=\"2026-01-01 10:30:00 -0800\"
  />
  <Record
    type=\"HKQuantityTypeIdentifierActiveEnergyBurned\"
    unit=\"kcal\"
    value=\"350\"
    startDate=\"2026-01-01 10:00:00 -0800\"
    endDate=\"2026-01-01 10:30:00 -0800\"
  />
  <Record
    type=\"HKQuantityTypeIdentifierRestingHeartRate\"
    unit=\"count/min\"
    value=\"58\"
    startDate=\"2026-01-01 07:00:00 -0800\"
    endDate=\"2026-01-01 07:00:00 -0800\"
  />
  <Record
    type=\"HKQuantityTypeIdentifierBodyMass\"
    unit=\"kg\"
    value=\"72.4\"
    startDate=\"2026-01-01 07:05:00 -0800\"
    endDate=\"2026-01-01 07:05:00 -0800\"
  />
  <Record
    type=\"HKCategoryTypeIdentifierSleepAnalysis\"
    value=\"HKCategoryValueSleepAnalysisAsleep\"
    startDate=\"2026-01-01 00:00:00 -0800\"
    endDate=\"2026-01-01 07:00:00 -0800\"
  />
</HealthData>
"""
    path.write_text(xml, encoding="utf-8")


def test_validate_and_collect(tmp_dir):
    export_path = Path(tmp_dir) / "export.xml"
    _write_export(export_path)

    collector = AppleHealthExportCollector(export_path=str(export_path))
    assert collector.validate() is True

    rows = collector.collect(date(2026, 1, 1), date(2026, 1, 1))
    assert len(rows) == 1
    day = rows[0]

    assert day.activity is not None
    assert day.activity.steps == 1500
    assert round(day.activity.distance_km or 0.0, 2) == 2.4
    assert day.activity.calories_burned == 350

    assert day.heart_rate is not None
    assert day.heart_rate.resting_hr == 58

    assert day.body is not None
    assert day.body.weight_kg == 72.4

    assert day.sleep is not None
    assert day.sleep.total_minutes == 420.0


def test_validate_missing_file(tmp_dir):
    collector = AppleHealthExportCollector(export_path=str(Path(tmp_dir) / "missing.xml"))
    assert collector.validate() is False


def test_config_schema(tmp_dir):
    export_path = Path(tmp_dir) / "export.xml"
    _write_export(export_path)

    collector = AppleHealthExportCollector(export_path=str(export_path))
    schema = collector.get_config_schema()
    assert schema["type"] == "object"
    assert "export_path" in schema["properties"]
