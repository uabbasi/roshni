"""Tests for health.registry — plugin discovery."""

from datetime import date
from unittest.mock import MagicMock

import pytest

from roshni.health.collector import BaseCollector
from roshni.health.models import DailyHealth
from roshni.health.registry import HealthCollectorRegistry


class FakeCollector(BaseCollector):
    name = "fake"

    def collect(self, start_date: date, end_date: date) -> list[DailyHealth]:
        return [DailyHealth(date=start_date)]


class TestRegistry:
    def test_manual_register(self):
        reg = HealthCollectorRegistry()
        reg.register("fake", FakeCollector)
        assert "fake" in reg.list_names()

    def test_get(self):
        reg = HealthCollectorRegistry()
        reg.register("fake", FakeCollector)
        assert reg.get("fake") is FakeCollector
        assert reg.get("nonexistent") is None

    def test_create(self):
        reg = HealthCollectorRegistry()
        reg.register("fake", FakeCollector)
        instance = reg.create("fake")
        assert instance.name == "fake"

    def test_create_missing_raises(self):
        reg = HealthCollectorRegistry()
        with pytest.raises(KeyError, match="No collector"):
            reg.create("missing")

    def test_discover_returns_dict(self):
        reg = HealthCollectorRegistry()
        result = reg.discover()
        assert isinstance(result, dict)

    def test_discover_accepts_collector_class(self, monkeypatch):
        ep = MagicMock()
        ep.name = "fake_ep"
        ep.load.return_value = FakeCollector
        monkeypatch.setattr("roshni.health.registry.entry_points", lambda group: [ep])

        reg = HealthCollectorRegistry()
        result = reg.discover()
        assert result["fake_ep"] is FakeCollector

    def test_discover_skips_invalid_entry(self, monkeypatch):
        class NotACollector:
            pass

        ep = MagicMock()
        ep.name = "invalid"
        ep.load.return_value = NotACollector
        monkeypatch.setattr("roshni.health.registry.entry_points", lambda group: [ep])

        reg = HealthCollectorRegistry()
        result = reg.discover()
        assert "invalid" not in result
