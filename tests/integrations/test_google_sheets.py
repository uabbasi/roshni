"""Tests for roshni.integrations.google_sheets."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from roshni.integrations.google_sheets import (
    CacheEntry,
    GoogleSheetsBase,
    SheetsTimeoutError,
    _with_timeout,
    is_row_empty,
    is_row_empty_or_zero,
)

# -- CacheEntry ------------------------------------------------------------


class TestCacheEntry:
    def test_fresh_entry_not_expired(self):
        entry = CacheEntry(data="test", cached_at=time.time(), spreadsheet_modified_time=None)
        assert not entry.is_expired(6.0)

    def test_old_entry_expired(self):
        eight_hours_ago = time.time() - (8 * 3600)
        entry = CacheEntry(data="test", cached_at=eight_hours_ago, spreadsheet_modified_time=None)
        assert entry.is_expired(6.0)

    def test_age_hours(self):
        two_hours_ago = time.time() - (2 * 3600)
        entry = CacheEntry(data="test", cached_at=two_hours_ago, spreadsheet_modified_time=None)
        assert 1.9 < entry.age_hours() < 2.1


# -- _with_timeout ----------------------------------------------------------


class TestWithTimeout:
    def test_returns_result(self):
        result = _with_timeout(lambda: 42, timeout=5)
        assert result == 42

    def test_timeout_raises(self):
        import time as t

        def slow():
            t.sleep(5)

        with pytest.raises(SheetsTimeoutError, match="timed out"):
            _with_timeout(slow, timeout=1, operation="test op")


# -- Row helpers ------------------------------------------------------------


class TestRowHelpers:
    def test_empty_row(self):
        assert is_row_empty(["", " ", float("nan")])

    def test_non_empty_row(self):
        assert not is_row_empty(["hello", "", float("nan")])

    def test_empty_or_zero_row(self):
        assert is_row_empty_or_zero(["", "0", float("nan")])

    def test_non_zero_row(self):
        assert not is_row_empty_or_zero(["5", "0", ""])


# -- GoogleSheetsBase -------------------------------------------------------


class TestGoogleSheetsBase:
    def test_init_defaults(self):
        sheet = GoogleSheetsBase(sheet_name="Test Sheet")
        assert sheet.sheet_name == "Test Sheet"
        assert sheet.cache_ttl_hours == 6.0
        assert sheet.timeout == 30
        assert sheet.client is None

    def test_init_custom_cache_dir(self, tmp_path):
        sheet = GoogleSheetsBase(sheet_name="Test", cache_dir=str(tmp_path / "cache"))
        assert sheet.cache_dir == tmp_path / "cache"

    def test_setup_without_auth_raises(self):
        sheet = GoogleSheetsBase(sheet_name="Test")
        with pytest.raises(RuntimeError, match="No auth provided"):
            sheet.setup_client()

    def test_setup_with_auth(self):
        mock_auth = MagicMock()
        mock_client = MagicMock()
        mock_sh = MagicMock()
        mock_auth.get_gspread_client.return_value = mock_client
        mock_client.open.return_value = mock_sh

        sheet = GoogleSheetsBase(sheet_name="Test", auth=mock_auth)
        sheet.setup_client()

        assert sheet.client is mock_client
        assert sheet.sh is mock_sh
        mock_client.open.assert_called_once_with("Test")

    def test_cache_valid_none_entry(self):
        sheet = GoogleSheetsBase(sheet_name="Test")
        assert not sheet.is_cache_valid(None)

    def test_cache_valid_fresh_entry(self):
        sheet = GoogleSheetsBase(sheet_name="Test")
        entry = CacheEntry(data="x", cached_at=time.time(), spreadsheet_modified_time=None)
        assert sheet.is_cache_valid(entry)

    def test_cache_stale_checks_modified(self):
        sheet = GoogleSheetsBase(sheet_name="Test")
        old = time.time() - (8 * 3600)
        entry = CacheEntry(data="x", cached_at=old, spreadsheet_modified_time="2025-01-01T00:00:00Z")

        # Mock get_spreadsheet_modified_time to return same value
        sheet.get_spreadsheet_modified_time = MagicMock(return_value="2025-01-01T00:00:00Z")
        assert sheet.is_cache_valid(entry)

        # Different modified time = invalid
        sheet.get_spreadsheet_modified_time = MagicMock(return_value="2025-01-02T00:00:00Z")
        assert not sheet.is_cache_valid(entry)

    def test_pull_uses_cache(self, tmp_path):
        """When cache is fresh, pull_sheet_as_df returns cached data without API call."""
        sheet = GoogleSheetsBase(sheet_name="Test", cache_dir=str(tmp_path))

        # Pre-populate cache
        from roshni.integrations.google_sheets import _cache_path, _save_cache

        cp = _cache_path(tmp_path, "Test", "Sheet1")
        entry = CacheEntry(data="cached_df", cached_at=time.time(), spreadsheet_modified_time=None)
        _save_cache(cp, entry)

        result = sheet.pull_sheet_as_df("Sheet1")
        assert result == "cached_df"
        # No API call was made (client is still None)
        assert sheet.client is None
