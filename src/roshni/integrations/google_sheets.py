"""Google Sheets base class with smart caching and timeout protection.

Provides ``GoogleSheetsBase`` -- a base class for spreadsheet access that:
  - Caches worksheet data on disk with TTL + Drive modified-time validation
  - Wraps all API calls with timeout protection (prevents indefinite hangs)
  - Converts worksheets to/from pandas DataFrames

Note: pickle is used for DataFrame cache serialization.  This is local
trusted data only (cached spreadsheet content), not for untrusted input.

Requires ``roshni[google]``.

Example::

    class BudgetSheet(GoogleSheetsBase):
        def __init__(self):
            super().__init__(
                sheet_name="Household Budget",
                auth=ServiceAccountAuth(key_path="~/.secrets/sa.json"),
            )

    sheet = BudgetSheet()
    df = sheet.pull_sheet_as_df("Expenses")
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")

# Defaults
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_CACHE_TTL_HOURS = 6.0

log = logging.getLogger(__name__)


class SheetsTimeoutError(TimeoutError):
    """A Google Sheets API call exceeded the configured timeout."""


def _with_timeout(
    func: Callable[[], T],
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    operation: str = "Google Sheets call",
):
    """Execute *func* (zero-arg callable) with a wall-clock timeout."""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func)
    timed_out = False
    try:
        return future.result(timeout=timeout)
    except FuturesTimeout:
        timed_out = True
        cancelled = future.cancel()
        log.warning("%s timed out after %ss (cancelled=%s)", operation, timeout, cancelled)
        executor.shutdown(wait=False, cancel_futures=True)
        raise SheetsTimeoutError(
            f"{operation} timed out after {timeout}s. Check network connection or try again later."
        )
    finally:
        if not timed_out:
            executor.shutdown(wait=True, cancel_futures=False)


# -- Smart cache -----------------------------------------------------------

# Note: pickle is used here for local DataFrame caching (trusted data only).
# This is NOT for untrusted/external input.


@dataclass
class CacheEntry:
    """Cached worksheet data with metadata for smart invalidation."""

    data: Any
    cached_at: float  # time.time() epoch
    spreadsheet_modified_time: str | None  # ISO timestamp from Drive API

    def is_expired(self, ttl_hours: float) -> bool:
        return (time.time() - self.cached_at) / 3600 > ttl_hours

    def age_hours(self) -> float:
        return (time.time() - self.cached_at) / 3600


def _cache_path(cache_dir: Path, sheet_name: str, worksheet_name: str) -> Path:
    safe = f"{sheet_name}_{worksheet_name}".replace(" ", "_").lower()
    return cache_dir / f"{safe}.cache"


def _load_cache(path: Path) -> CacheEntry | None:
    if not path.exists():
        return None
    try:
        import pickle as _pkl

        with open(path, "rb") as f:
            return _pkl.load(f)
    except Exception as e:
        log.warning("Failed to load cache %s: %s", path, e)
        return None


def _save_cache(path: Path, entry: CacheEntry) -> None:
    try:
        import pickle as _pkl

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            _pkl.dump(entry, f)
    except Exception as e:
        log.warning("Failed to save cache %s: %s", path, e)


# -- Row helpers ------------------------------------------------------------


def is_row_empty(row) -> bool:
    """True if every cell is NaN or whitespace."""
    import pandas as pd

    return all(pd.isna(x) or str(x).strip() == "" for x in row)


def is_row_empty_or_zero(row) -> bool:
    """True if every cell is NaN, whitespace, or zero."""
    import pandas as pd

    return all(pd.isna(x) or str(x).strip() in ("", "0") for x in row)


# -- Base class -------------------------------------------------------------


class GoogleSheetsBase:
    """Base class for Google Sheets access with smart caching.

    Args:
        sheet_name: Spreadsheet title (as it appears in Google Drive).
        auth: A :class:`~roshni.core.auth.ServiceAccountAuth` instance.
        cache_dir: Directory for disk cache.  Defaults to ``~/.roshni-data/cache/sheets``.
        cache_ttl_hours: Hours before a cached worksheet is considered stale.
        timeout: Seconds before an API call is killed.
    """

    def __init__(
        self,
        sheet_name: str,
        auth=None,
        cache_dir: str | Path | None = None,
        cache_ttl_hours: float = DEFAULT_CACHE_TTL_HOURS,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ):
        self.sheet_name = sheet_name
        self.auth = auth
        self.cache_dir = Path(cache_dir or "~/.roshni-data/cache/sheets").expanduser()
        self.cache_ttl_hours = cache_ttl_hours
        self.timeout = timeout

        # Populated lazily by setup_client()
        self.client = None
        self.sh = None

    def setup_client(self) -> None:
        """Initialise the gspread client and open the spreadsheet.

        Raises:
            SheetsTimeoutError: If connection takes too long.
        """
        if self.auth is None:
            raise RuntimeError("No auth provided. Pass a ServiceAccountAuth to the constructor.")

        def _connect():
            self.client = self.auth.get_gspread_client()
            self.sh = self.client.open(self.sheet_name)

        _with_timeout(_connect, timeout=self.timeout, operation=f"Connecting to '{self.sheet_name}'")

    def _ensure_client(self) -> None:
        if self.client is None or self.sh is None:
            self.setup_client()

    # -- Modified-time check ------------------------------------------------

    def get_spreadsheet_modified_time(self) -> str | None:
        """Lightweight Drive API call (~200ms) to get last-modified time."""
        self._ensure_client()
        try:
            spreadsheet_id = self.sh.id  # type: ignore[union-attr]

            def _fetch():
                return (
                    self.client.http_client.request(  # type: ignore[union-attr]
                        "get",
                        f"https://www.googleapis.com/drive/v3/files/{spreadsheet_id}",
                        params={"fields": "modifiedTime"},
                    )
                    .json()
                    .get("modifiedTime")
                )

            return _with_timeout(_fetch, timeout=10, operation=f"Checking modified time for '{self.sheet_name}'")
        except SheetsTimeoutError:
            log.warning("Timeout checking modified time for %s", self.sheet_name)
            return None
        except Exception as e:
            log.warning("Failed to get modified time for %s: %s", self.sheet_name, e)
            return None

    def is_cache_valid(self, entry: CacheEntry | None) -> bool:
        """Two-tier validation: TTL check (free) then Drive modified-time check (~200ms)."""
        if entry is None:
            return False
        if not entry.is_expired(self.cache_ttl_hours):
            return True

        current = self.get_spreadsheet_modified_time()
        if current is None:
            return False
        return entry.spreadsheet_modified_time == current

    # -- Core data operations -----------------------------------------------

    def pull_sheet_as_df(
        self,
        name: str,
        skiprows: int = 0,
        head: int = 0,
        force_refresh: bool = False,
        **options,
    ):
        """Pull a worksheet as a pandas DataFrame with smart caching.

        Args:
            name: Worksheet tab name.
            skiprows: Rows to skip at the top.
            head: Row number to use as column names.
            force_refresh: Bypass cache.
            **options: Extra args passed to ``gspread_dataframe.get_as_dataframe``.

        Returns:
            ``pandas.DataFrame``.
        """
        try:
            from gspread_dataframe import get_as_dataframe
        except ImportError:
            raise ImportError("Install with: pip install roshni[google]")

        cp = _cache_path(self.cache_dir, self.sheet_name, name)
        entry = _load_cache(cp)

        if not force_refresh and self.is_cache_valid(entry):
            log.debug("Using cached %s/%s (%.1fh old)", self.sheet_name, name, entry.age_hours())  # type: ignore[union-attr]
            return entry.data  # type: ignore[union-attr]

        log.info("Fetching %s/%s from Google Sheets...", self.sheet_name, name)
        self._ensure_client()

        def _fetch():
            ws = self.sh.worksheet(name)  # type: ignore[union-attr]
            return get_as_dataframe(
                ws,
                evaluate_formulas=True,
                parse_dates=True,
                skiprows=skiprows,
                header=head,
                **options,
            )

        df = _with_timeout(_fetch, timeout=self.timeout, operation=f"Fetching '{self.sheet_name}/{name}'")

        # Clean up empty rows/columns
        df = df.dropna(how="all")
        df = df[~df.apply(is_row_empty_or_zero, axis=1)]
        df = df.dropna(axis=1, how="all")

        # Save to cache
        modified = self.get_spreadsheet_modified_time()
        _save_cache(cp, CacheEntry(data=df, cached_at=time.time(), spreadsheet_modified_time=modified))
        return df

    def save_df_to_sheet(
        self,
        name: str,
        df,
        *,
        skip_if_unchanged: bool = True,
        **options,
    ) -> bool:
        """Write a DataFrame to a worksheet.

        Args:
            name: Worksheet tab name.
            df: pandas DataFrame.
            skip_if_unchanged: If True, compare SHA-256 to skip redundant writes.
            **options: Extra args passed to ``gspread_dataframe.set_with_dataframe``.

        Returns:
            True if data was written, False if skipped.
        """
        try:
            from gspread_dataframe import set_with_dataframe
        except ImportError:
            raise ImportError("Install with: pip install roshni[google]")

        self._ensure_client()
        ws = self.sh.worksheet(name)  # type: ignore[union-attr]

        if skip_if_unchanged:
            new_hash = hashlib.sha256(df.to_json(orient="records").encode()).hexdigest()
            try:
                existing = ws.acell("A1").note
                if existing == new_hash:
                    log.debug("Data unchanged for %s/%s, skipping write", self.sheet_name, name)
                    return False
            except Exception:
                pass

        set_with_dataframe(ws, df, resize=True, **options)

        if skip_if_unchanged:
            try:
                ws.update_note("A1", new_hash)  # type: ignore[possibly-undefined]
            except Exception:
                pass

        log.info("Wrote %d rows to %s/%s", len(df), self.sheet_name, name)
        return True

    def save_csv_to_sheet(self, csv_path: str | Path, worksheet_name: str) -> bool:
        """Read a CSV and write it to a worksheet."""
        import pandas as pd

        df = pd.read_csv(csv_path)
        return self.save_df_to_sheet(worksheet_name, df)
