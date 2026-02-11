"""
Daily token budget tracker.

Stores a running total in a JSON file (default: ~/.roshni-data/token_usage.json).
Auto-resets when the date changes.  Thread-safe for executor threads.
"""

import json
import os
import tempfile
import threading
from datetime import datetime, timedelta

_lock = threading.Lock()
_LOCK_TIMEOUT = 5  # seconds — don't block LLM calls forever on hung fs
_DEFAULT_DAILY_LIMIT = 500_000
_DEFAULT_DAILY_COST_LIMIT = 7.0  # USD — primary budget control
_BUDGET_RESET_HOUR = 6  # Reset budget at 6am, not midnight

# Module-level path — set via configure() or defaults to ~/.roshni-data/token_usage.json
_usage_path: str | None = None


def configure(
    data_dir: str | None = None,
    daily_limit: int | None = None,
    daily_cost_limit: float | None = None,
) -> None:
    """Set the storage path and/or daily limit for token tracking.

    Call once at startup.  If never called, defaults apply.
    """
    global _usage_path, _DEFAULT_DAILY_LIMIT, _DEFAULT_DAILY_COST_LIMIT
    if data_dir is not None:
        _usage_path = os.path.join(os.path.expanduser(data_dir), "token_usage.json")
    if daily_limit is not None:
        _DEFAULT_DAILY_LIMIT = daily_limit
    if daily_cost_limit is not None:
        _DEFAULT_DAILY_COST_LIMIT = daily_cost_limit


def _get_path() -> str:
    global _usage_path
    if _usage_path is None:
        _usage_path = os.path.join(os.path.expanduser("~/.roshni-data"), "token_usage.json")
    return _usage_path


def _budget_date() -> str:
    """Return the current budget day as ISO date string.

    The budget day resets at ``_BUDGET_RESET_HOUR`` (default 6am), not midnight.
    Before 6am counts as the previous day's budget.
    """
    now = datetime.now()
    if now.hour < _BUDGET_RESET_HOUR:
        return (now - timedelta(days=1)).date().isoformat()
    return now.date().isoformat()


def _load() -> dict:
    """Load usage data, resetting if the budget day has changed."""
    try:
        with open(_get_path()) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        data = {}

    today = _budget_date()
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


def record_usage(
    input_tokens: int,
    output_tokens: int,
    provider: str = "",
    model: str = "",
    cache_creation_tokens: int = 0,
    cache_read_tokens: int = 0,
    cost_usd: float = 0.0,
) -> None:
    """Add tokens to today's tally.  Silent on errors."""
    try:
        if not _lock.acquire(timeout=_LOCK_TIMEOUT):
            return
        try:
            data = _load()
            data["input_tokens"] = data.get("input_tokens", 0) + input_tokens
            data["output_tokens"] = data.get("output_tokens", 0) + output_tokens
            data["calls"] = data.get("calls", 0) + 1
            data["cache_creation_tokens"] = data.get("cache_creation_tokens", 0) + cache_creation_tokens
            data["cache_read_tokens"] = data.get("cache_read_tokens", 0) + cache_read_tokens
            data["cost_usd"] = round(data.get("cost_usd", 0.0) + cost_usd, 6)
            _save(data)
        finally:
            _lock.release()
    except Exception:
        pass  # budget tracking never blocks


def check_budget(daily_limit: int | None = None, daily_cost_limit: float | None = None) -> tuple[bool, int]:
    """Return (within_budget, remaining_tokens).

    If cost tracking is available, uses dollar limit. Falls back to token limit.
    """
    token_limit = daily_limit or _DEFAULT_DAILY_LIMIT
    cost_limit = daily_cost_limit or _DEFAULT_DAILY_COST_LIMIT
    try:
        if not _lock.acquire(timeout=_LOCK_TIMEOUT):
            return True, token_limit  # fail-open on contention
        try:
            data = _load()
        finally:
            _lock.release()

        # Cost-based check takes priority
        cost_used = data.get("cost_usd", 0.0)
        if cost_used > 0:
            within = cost_used < cost_limit
            remaining_frac = max(0, (cost_limit - cost_used) / cost_limit)
            remaining = int(token_limit * remaining_frac)
            return within, remaining

        # Fallback: token-based (legacy / cost not yet recorded)
        total = data.get("input_tokens", 0) + data.get("output_tokens", 0)
        return total < token_limit, token_limit - total
    except Exception:
        return True, token_limit  # fail-open


def get_usage_summary(daily_limit: int | None = None, daily_cost_limit: float | None = None) -> dict:
    """Return today's usage stats."""
    token_limit = daily_limit or _DEFAULT_DAILY_LIMIT
    cost_limit = daily_cost_limit or _DEFAULT_DAILY_COST_LIMIT
    try:
        if not _lock.acquire(timeout=_LOCK_TIMEOUT):
            return {}
        try:
            data = _load()
        finally:
            _lock.release()
        total = data.get("input_tokens", 0) + data.get("output_tokens", 0)
        cache_creation = data.get("cache_creation_tokens", 0)
        cache_read = data.get("cache_read_tokens", 0)
        cache_total = cache_creation + cache_read
        cache_hit_rate = round(cache_read / cache_total * 100, 1) if cache_total else 0.0
        cost_used = data.get("cost_usd", 0.0)
        # Use cost-based pct if cost tracking is active
        if cost_used > 0:
            pct_used = round(cost_used / cost_limit * 100, 1) if cost_limit else 0
        else:
            pct_used = round(total / token_limit * 100, 1) if token_limit else 0
        return {
            "date": data.get("date"),
            "input_tokens": data.get("input_tokens", 0),
            "output_tokens": data.get("output_tokens", 0),
            "total_tokens": total,
            "calls": data.get("calls", 0),
            "daily_limit": token_limit,
            "remaining": token_limit - total,
            "pct_used": pct_used,
            "cache_creation_tokens": cache_creation,
            "cache_read_tokens": cache_read,
            "cache_hit_rate": cache_hit_rate,
            "cost_usd": round(cost_used, 4),
            "cost_limit_usd": cost_limit,
        }
    except Exception:
        return {}


def get_budget_pressure() -> float:
    """Return 0.0-1.0 indicating budget pressure. 0=fine, 1=exhausted."""
    try:
        if not _lock.acquire(timeout=_LOCK_TIMEOUT):
            return 0.0  # fail-open
        try:
            data = _load()
        finally:
            _lock.release()
        cost_used = data.get("cost_usd", 0.0)
        if cost_used > 0:
            return min(1.0, cost_used / _DEFAULT_DAILY_COST_LIMIT)
        total = data.get("input_tokens", 0) + data.get("output_tokens", 0)
        return min(1.0, total / _DEFAULT_DAILY_LIMIT)
    except Exception:
        return 0.0  # fail-open
