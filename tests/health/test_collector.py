"""Tests for health.collector — Protocol + BaseCollector."""

from datetime import date

from roshni.health.collector import BaseCollector, HealthCollector
from roshni.health.models import DailyHealth


class StubCollector(BaseCollector):
    name = "stub"

    def collect(self, start_date: date, end_date: date) -> list[DailyHealth]:
        return [DailyHealth(date=d) for d in self._date_range(start_date, end_date)]


class TestBaseCollector:
    def test_collect(self):
        c = StubCollector()
        results = c.collect(date(2025, 1, 1), date(2025, 1, 3))
        assert len(results) == 3
        assert results[0].date == date(2025, 1, 1)
        assert results[2].date == date(2025, 1, 3)

    def test_validate_default(self):
        assert StubCollector().validate() is True

    def test_config_schema_default(self):
        assert StubCollector().get_config_schema() == {}

    def test_stats(self):
        c = StubCollector()
        assert c.stats["api_calls"] == 0

    def test_date_range(self):
        c = StubCollector()
        dates = c._date_range(date(2025, 3, 1), date(2025, 3, 1))
        assert dates == [date(2025, 3, 1)]


class TestProtocol:
    def test_stub_satisfies_protocol(self):
        assert isinstance(StubCollector(), HealthCollector)

    def test_duck_typed_class(self):
        class DuckCollector:
            name = "duck"

            def collect(self, start_date, end_date):
                return []

            def validate(self):
                return True

            def get_config_schema(self):
                return {}

        assert isinstance(DuckCollector(), HealthCollector)
