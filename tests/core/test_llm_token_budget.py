"""Tests for core.llm.token_budget — daily token tracking."""

import json

import pytest

from roshni.core.llm import token_budget


@pytest.fixture(autouse=True)
def _isolate_budget(tmp_path, monkeypatch):
    """Point token budget at a temp directory for every test."""
    usage_file = str(tmp_path / "token_usage.json")
    monkeypatch.setattr(token_budget, "_usage_path", usage_file)
    monkeypatch.setattr(token_budget, "_DEFAULT_DAILY_LIMIT", 500_000)
    monkeypatch.setattr(token_budget, "_last_known_budget", None)
    monkeypatch.setattr(token_budget, "_degraded_warned", set())


class TestRecordUsage:
    def test_records_tokens(self, tmp_path):
        token_budget.record_usage(100, 50)
        summary = token_budget.get_usage_summary()
        assert summary["input_tokens"] == 100
        assert summary["output_tokens"] == 50
        assert summary["calls"] == 1
        assert summary["total_tokens"] == 150

    def test_accumulates(self):
        token_budget.record_usage(100, 50)
        token_budget.record_usage(200, 100)
        summary = token_budget.get_usage_summary()
        assert summary["input_tokens"] == 300
        assert summary["output_tokens"] == 150
        assert summary["calls"] == 2


class TestCheckBudget:
    def test_within_budget(self):
        within, remaining = token_budget.check_budget(daily_limit=1000)
        assert within is True
        assert remaining == 1000

    def test_over_budget(self):
        token_budget.record_usage(400, 200)
        within, remaining = token_budget.check_budget(daily_limit=500)
        assert within is False
        assert remaining == -100

    def test_custom_limit(self):
        token_budget.record_usage(80, 20)
        within, remaining = token_budget.check_budget(daily_limit=200)
        assert within is True
        assert remaining == 100


class TestUsageSummary:
    def test_empty_summary(self):
        summary = token_budget.get_usage_summary()
        assert summary["total_tokens"] == 0
        assert summary["pct_used"] == 0.0

    def test_percentage(self):
        token_budget.record_usage(250_000, 0)
        summary = token_budget.get_usage_summary()
        assert summary["pct_used"] == 50.0


class TestConfigure:
    def test_configure_path(self, tmp_path):
        new_dir = str(tmp_path / "custom")
        token_budget.configure(data_dir=new_dir)
        assert "custom" in token_budget._get_path()

    def test_configure_limit(self):
        token_budget.configure(daily_limit=1_000_000)
        _within, remaining = token_budget.check_budget()
        assert remaining == 1_000_000


class TestAtomicWrite:
    def test_file_created(self):
        token_budget.record_usage(1, 1)
        path = token_budget._get_path()
        with open(path) as f:
            data = json.load(f)
        assert data["calls"] == 1


class TestDegradedMode:
    def test_lock_contention_fails_closed_by_default(self, monkeypatch):
        class _NoLock:
            def acquire(self, timeout=0):
                return False

            def release(self):
                return None

        monkeypatch.setattr(token_budget, "_FAIL_OPEN_ON_ERROR", False)
        monkeypatch.setattr(token_budget, "_lock", _NoLock())
        within, remaining = token_budget.check_budget(daily_limit=1000)
        assert within is False
        assert remaining == 0

    def test_lock_contention_can_fail_open_when_configured(self, monkeypatch):
        class _NoLock:
            def acquire(self, timeout=0):
                return False

            def release(self):
                return None

        monkeypatch.setattr(token_budget, "_FAIL_OPEN_ON_ERROR", True)
        monkeypatch.setattr(token_budget, "_lock", _NoLock())
        within, remaining = token_budget.check_budget(daily_limit=1000)
        assert within is True
        assert remaining == 1000
