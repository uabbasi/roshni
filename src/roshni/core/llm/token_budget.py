"""
Daily token budget tracker.

Stores a running total in a JSON file (default: ~/.roshni-data/token_usage.json).
Auto-resets when the date changes.  Thread-safe for executor threads.
"""

import json
import os
import tempfile
import threading
from datetime import date

_lock = threading.Lock()
_LOCK_TIMEOUT = 5  # seconds — don't block LLM calls forever on hung fs
_DEFAULT_DAILY_LIMIT = 500_000

# Module-level path — set via configure() or defaults to ~/.roshni-data/token_usage.json
_usage_path: str | None = None


def configure(data_dir: str | None = None, daily_limit: int | None = None) -> None:
    """Set the storage path and/or daily limit for token tracking.

    Call once at startup.  If never called, defaults apply.
    """
    global _usage_path, _DEFAULT_DAILY_LIMIT
    if data_dir is not None:
        _usage_path = os.path.join(os.path.expanduser(data_dir), "token_usage.json")
    if daily_limit is not None:
        _DEFAULT_DAILY_LIMIT = daily_limit


def _get_path() -> str:
    global _usage_path
    if _usage_path is None:
        _usage_path = os.path.join(os.path.expanduser("~/.roshni-data"), "token_usage.json")
    return _usage_path


def _load() -> dict:
    """Load usage data, resetting if the date has changed."""
    try:
        with open(_get_path()) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        data = {}

    today = date.today().isoformat()
    if data.get("date") != today:
        data = {"date": today, "input_tokens": 0, "output_tokens": 0, "calls": 0}
    return data


def _save(data: dict) -> None:
    """Atomic write: temp file + rename so a kill can't corrupt."""
    path = _get_path()
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)  # atomic on POSIX
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def record_usage(input_tokens: int, output_tokens: int, provider: str = "", model: str = "") -> None:
    """Add tokens to today's tally.  Silent on errors."""
    try:
        if not _lock.acquire(timeout=_LOCK_TIMEOUT):
            return
        try:
            data = _load()
            data["input_tokens"] = data.get("input_tokens", 0) + input_tokens
            data["output_tokens"] = data.get("output_tokens", 0) + output_tokens
            data["calls"] = data.get("calls", 0) + 1
            _save(data)
        finally:
            _lock.release()
    except Exception:
        pass  # budget tracking never blocks


def check_budget(daily_limit: int | None = None) -> tuple[bool, int]:
    """Return (within_budget, remaining_tokens)."""
    limit = daily_limit or _DEFAULT_DAILY_LIMIT
    try:
        if not _lock.acquire(timeout=_LOCK_TIMEOUT):
            return True, limit  # fail-open on contention
        try:
            data = _load()
        finally:
            _lock.release()
        total = data.get("input_tokens", 0) + data.get("output_tokens", 0)
        return total < limit, limit - total
    except Exception:
        return True, limit  # fail-open


def get_usage_summary(daily_limit: int | None = None) -> dict:
    """Return today's usage stats."""
    limit = daily_limit or _DEFAULT_DAILY_LIMIT
    try:
        if not _lock.acquire(timeout=_LOCK_TIMEOUT):
            return {}
        try:
            data = _load()
        finally:
            _lock.release()
        total = data.get("input_tokens", 0) + data.get("output_tokens", 0)
        return {
            "date": data.get("date"),
            "input_tokens": data.get("input_tokens", 0),
            "output_tokens": data.get("output_tokens", 0),
            "total_tokens": total,
            "calls": data.get("calls", 0),
            "daily_limit": limit,
            "remaining": limit - total,
            "pct_used": round(total / limit * 100, 1) if limit else 0,
        }
    except Exception:
        return {}
