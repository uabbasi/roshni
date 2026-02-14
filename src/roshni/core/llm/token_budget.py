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

from loguru import logger

_lock = threading.Lock()
_LOCK_TIMEOUT = 5  # seconds — don't block LLM calls forever on hung fs
_DEFAULT_DAILY_LIMIT = 500_000
_DEFAULT_DAILY_COST_LIMIT = 7.0  # USD — primary budget control
_BUDGET_RESET_HOUR = 6  # Reset budget at 6am, not midnight
_FAIL_OPEN_ON_ERROR = False  # safer default: fail closed when budget state is unavailable
_degraded_warned: set[str] = set()

# Module-level path — set via configure() or defaults to ~/.roshni-data/token_usage.json
_usage_path: str | None = None


def configure(
    data_dir: str | None = None,
    daily_limit: int | None = None,
    daily_cost_limit: float | None = None,
    fail_open_on_error: bool | None = None,
) -> None:
    """Set the storage path and/or daily limit for token tracking.

    Call once at startup.  If never called, defaults apply.
    """
    global _usage_path, _DEFAULT_DAILY_LIMIT, _DEFAULT_DAILY_COST_LIMIT, _FAIL_OPEN_ON_ERROR
    if data_dir is not None:
        _usage_path = os.path.join(os.path.expanduser(data_dir), "token_usage.json")
    if daily_limit is not None:
        _DEFAULT_DAILY_LIMIT = daily_limit
    if daily_cost_limit is not None:
        _DEFAULT_DAILY_COST_LIMIT = daily_cost_limit
    if fail_open_on_error is not None:
        _FAIL_OPEN_ON_ERROR = bool(fail_open_on_error)


def _log_degraded_once(reason: str) -> None:
    """Log one warning per degradation reason to avoid log spam."""
    if reason in _degraded_warned:
        return
    _degraded_warned.add(reason)
    logger.warning(f"Token budget tracker degraded: {reason}")


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
    """Add tokens to today's tally. Never raises."""
    try:
        if not _lock.acquire(timeout=_LOCK_TIMEOUT):
            _log_degraded_once("lock timeout in record_usage; usage increment skipped")
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
    except Exception as e:
        _log_degraded_once(f"record_usage write failure: {e}")


def check_budget(daily_limit: int | None = None, daily_cost_limit: float | None = None) -> tuple[bool, int]:
    """Return (within_budget, remaining_tokens).

    If cost tracking is available, uses dollar limit. Falls back to token limit.
    """
    token_limit = daily_limit or _DEFAULT_DAILY_LIMIT
    cost_limit = daily_cost_limit or _DEFAULT_DAILY_COST_LIMIT
    try:
        if not _lock.acquire(timeout=_LOCK_TIMEOUT):
            _log_degraded_once("lock timeout in check_budget")
            if _FAIL_OPEN_ON_ERROR:
                return True, token_limit
            return False, 0
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
    except Exception as e:
        _log_degraded_once(f"check_budget failure: {e}")
        if _FAIL_OPEN_ON_ERROR:
            return True, token_limit
        return False, 0


def get_usage_summary(daily_limit: int | None = None, daily_cost_limit: float | None = None) -> dict:
    """Return today's usage stats."""
    token_limit = daily_limit or _DEFAULT_DAILY_LIMIT
    cost_limit = daily_cost_limit or _DEFAULT_DAILY_COST_LIMIT
    try:
        if not _lock.acquire(timeout=_LOCK_TIMEOUT):
            _log_degraded_once("lock timeout in get_usage_summary")
            return {}
        try:
            data = _load()
        finally:
            _lock.release()
        total = data.get("input_tokens", 0) + data.get("output_tokens", 0)
        cache_creation = data.get("cache_creation_tokens", 0)
        cache_read = data.get("cache_read_tokens", 0)
        input_tokens = data.get("input_tokens", 0)
        # input_tokens (prompt_tokens) should include cache_read for OpenAI/Gemini;
        # for Anthropic it may exclude them, so use the larger denominator to be safe.
        cache_denominator = max(input_tokens, cache_creation + cache_read)
        if cache_denominator and cache_read:
            cache_hit_rate = min(round(cache_read / cache_denominator * 100, 1), 100.0)
        else:
            cache_hit_rate = 0.0
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
    except Exception as e:
        _log_degraded_once(f"get_usage_summary failure: {e}")
        return {}


def get_budget_pressure() -> float:
    """Return 0.0-1.0 indicating budget pressure. 0=fine, 1=exhausted."""
    try:
        if not _lock.acquire(timeout=_LOCK_TIMEOUT):
            _log_degraded_once("lock timeout in get_budget_pressure")
            return 0.0  # neutral pressure
        try:
            data = _load()
        finally:
            _lock.release()
        cost_used = data.get("cost_usd", 0.0)
        if cost_used > 0:
            return min(1.0, cost_used / _DEFAULT_DAILY_COST_LIMIT)
        total = data.get("input_tokens", 0) + data.get("output_tokens", 0)
        return min(1.0, total / _DEFAULT_DAILY_LIMIT)
    except Exception as e:
        _log_degraded_once(f"get_budget_pressure failure: {e}")
        return 0.0  # fail-open
