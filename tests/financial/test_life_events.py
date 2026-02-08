"""Tests for roshni.financial.calculators.life_events."""

import pytest

from roshni.financial.calculators.life_events import (
    EventType,
    LifeEvent,
    ScenarioConfig,
    spending_phases_to_events,
)


class TestLifeEvent:
    def test_create_withdrawal(self):
        event = LifeEvent(year=5, event_type=EventType.LUMP_WITHDRAWAL, amount=100_000, description="Tuition")
        assert str(event) == "Tuition: -$100,000 (year 5)"

    def test_create_spending_reduction(self):
        event = LifeEvent(year=10, event_type=EventType.SPENDING_CHANGE, amount=-30_000, description="Mortgage payoff")
        assert "reduced" in str(event)

    def test_negative_year_raises(self):
        with pytest.raises(ValueError, match="year must be >= 0"):
            LifeEvent(year=-1, event_type=EventType.LUMP_WITHDRAWAL, amount=1000)

    def test_non_positive_withdrawal_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            LifeEvent(year=0, event_type=EventType.LUMP_WITHDRAWAL, amount=-100)

    def test_year_zero_displays_now(self):
        event = LifeEvent(year=0, event_type=EventType.LUMP_WITHDRAWAL, amount=5000)
        assert "now" in str(event)

    def test_format_amount_withdrawal(self):
        event = LifeEvent(year=1, event_type=EventType.LUMP_WITHDRAWAL, amount=50_000)
        assert event.format_amount() == "-$50,000"

    def test_format_amount_spending_increase(self):
        event = LifeEvent(year=1, event_type=EventType.SPENDING_CHANGE, amount=10_000)
        assert event.format_amount() == "+$10,000/year"


class TestScenarioConfig:
    def test_chaining(self):
        scenario = (
            ScenarioConfig(name="Test")
            .add_lump_withdrawal(100_000, year=0, description="Tuition")
            .add_spending_change(-30_000, year=5, description="Mortgage")
        )
        assert len(scenario.events) == 2
        assert scenario.total_lump_withdrawals == 100_000
        assert scenario.final_spending_adjustment == -30_000

    def test_clear(self):
        scenario = ScenarioConfig().add_lump_withdrawal(50_000)
        scenario.clear()
        assert len(scenario.events) == 0

    def test_events_for_year(self):
        scenario = (
            ScenarioConfig()
            .add_lump_withdrawal(100_000, year=5)
            .add_spending_change(-30_000, year=5)
            .add_lump_withdrawal(50_000, year=10)
        )
        year5_events = scenario.events_for_year(5)
        assert len(year5_events) == 2

    def test_str_no_events(self):
        scenario = ScenarioConfig(name="Empty")
        assert "No events" in str(scenario)

    def test_str_with_events(self):
        scenario = ScenarioConfig(name="Test").add_lump_withdrawal(100_000, year=1, description="School")
        assert "Test:" in str(scenario)
        assert "School" in str(scenario)


class TestSpendingPhasesToEvents:
    def test_basic_phases(self):
        phases = [
            {"year": 2026, "spending": 180_000, "description": "Current"},
            {"year": 2030, "spending": 130_000, "description": "Post-education"},
        ]
        events = spending_phases_to_events(phases, start_year=2026)

        assert len(events) == 1
        assert events[0].event_type == EventType.SPENDING_CHANGE
        assert events[0].amount == -50_000  # 130k - 180k
        assert events[0].year == 4  # 2030 - 2026

    def test_with_lump_withdrawal(self):
        phases = [
            {"year": 2026, "spending": 180_000, "description": "Current"},
            {"year": 2031, "spending": None, "lump_withdrawal": 500_000, "description": "Payoff"},
        ]
        events = spending_phases_to_events(phases, start_year=2026)

        lump_events = [e for e in events if e.event_type == EventType.LUMP_WITHDRAWAL]
        assert len(lump_events) == 1
        assert lump_events[0].amount == 500_000

    def test_empty_phases(self):
        assert spending_phases_to_events([]) == []
