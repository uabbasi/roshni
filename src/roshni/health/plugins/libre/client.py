"""Libre auth cache helper primitives (generic skeleton)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


def get_cached_auth(
    *,
    cache: dict[str, Any],
    fetch_auth: Callable[[], tuple[str, str]],
    ttl_seconds: int = 3000,
) -> tuple[str, str]:
    """Return cached auth tuple (token, account_id) or refresh via callback."""
    if cache and time.time() < float(cache.get("expires", 0)):
        return str(cache["token"]), str(cache["account_id"])

    token, account_id = fetch_auth()
    cache.clear()
    cache.update(
        {
            "token": token,
            "account_id": account_id,
            "expires": time.time() + ttl_seconds,
        }
    )
    return token, account_id


def get_auth_with_retry(
    *,
    cache: dict[str, Any],
    fetch_auth: Callable[[], tuple[str, str]],
    ttl_seconds: int = 3000,
) -> tuple[str, str]:
    """Fetch auth, clearing cache and retrying once if the first attempt fails."""
    try:
        return get_cached_auth(cache=cache, fetch_auth=fetch_auth, ttl_seconds=ttl_seconds)
    except Exception:
        cache.clear()
        return get_cached_auth(cache=cache, fetch_auth=fetch_auth, ttl_seconds=ttl_seconds)


__all__ = ["get_auth_with_retry", "get_cached_auth"]

