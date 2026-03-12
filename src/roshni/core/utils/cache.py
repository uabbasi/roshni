"""
JSON-based caching with TTL expiry.

Provides both a class-based API and standalone convenience functions.
The cache directory is configurable — no hardcoded paths.

Uses JSON for serialization to avoid pickle deserialization risks.
Data must be JSON-serializable (dicts, lists, strings, numbers, bools, None).
"""

import hashlib
import json
import os
import re
from datetime import datetime
from typing import Any

from loguru import logger

_SAFE_KEY_RE = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


def _safe_cache_filename(key: str) -> str:
    """Convert a cache key to a safe filename, preventing path traversal.

    Keys matching ``[a-zA-Z0-9_\\-\\.]+`` are used directly (with ``.json``
    suffix).  All other keys are SHA-256 hashed to produce a flat filename.
    """
    if _SAFE_KEY_RE.match(key) and ".." not in key:
        return f"{key}.json"
    return f"{hashlib.sha256(key.encode()).hexdigest()}.json"


class Cache:
    """General-purpose JSON cache with TTL-based expiry."""

    def __init__(self, cache_dir: str | None = None):
        """
        Args:
            cache_dir: Directory to store cache files.
                       If None, defaults to ~/.roshni-data/cache.
        """
        if cache_dir is None:
            cache_dir = os.path.join(os.path.expanduser("~"), ".roshni-data", "cache")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def cache_data(self, key: str, data: Any, expiry_days: int = 30) -> None:
        """Store data with a TTL. Data must be JSON-serializable."""
        cache_file = os.path.join(self.cache_dir, _safe_cache_filename(key))
        payload = {"data": data, "expiry": datetime.now().timestamp() + (expiry_days * 86400)}
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def get_cached_data(self, key: str) -> Any | None:
        """Retrieve cached data if not expired.

        Automatically removes the cache file if the entry has expired.
        """
        cache_file = os.path.join(self.cache_dir, _safe_cache_filename(key))
        if not os.path.exists(cache_file):
            # Check for legacy pickle file and migrate it
            return self._try_migrate_pickle(key)
        try:
            with open(cache_file, encoding="utf-8") as f:
                cached = json.load(f)
        except (json.JSONDecodeError, OSError):
            os.remove(cache_file)
            return None
        if cached["expiry"] > datetime.now().timestamp():
            return cached["data"]
        # Entry expired — clean up the stale file
        os.remove(cache_file)
        return None

    def _try_migrate_pickle(self, key: str) -> Any | None:
        """Attempt to read a legacy .pkl cache file, migrate to JSON, and return data."""
        pkl_file = os.path.join(self.cache_dir, f"{key}.pkl")
        if not os.path.exists(pkl_file):
            return None
        try:
            import pickle

            with open(pkl_file, "rb") as f:
                cached = pickle.load(f)  # one-time migration of trusted local files
            # Re-save as JSON
            if cached.get("expiry", 0) > datetime.now().timestamp():
                remaining_days = max(1, int((cached["expiry"] - datetime.now().timestamp()) / 86400))
                self.cache_data(key, cached["data"], expiry_days=remaining_days)
                os.remove(pkl_file)
                logger.info(f"Migrated cache key '{key}' from pickle to JSON")
                return cached["data"]
            # Expired — just remove
            os.remove(pkl_file)
        except Exception:
            # Corrupted pickle — remove silently
            try:
                os.remove(pkl_file)
            except OSError:
                pass
        return None

    def cleanup_expired(self) -> int:
        """Remove all expired cache entries from disk.

        Returns:
            Number of expired entries removed.
        """
        removed = 0
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            if filename.endswith(".json"):
                try:
                    with open(filepath, encoding="utf-8") as f:
                        cached = json.load(f)
                    if cached["expiry"] <= datetime.now().timestamp():
                        os.remove(filepath)
                        removed += 1
                except (json.JSONDecodeError, KeyError, OSError):
                    os.remove(filepath)
                    removed += 1
            elif filename.endswith(".pkl"):
                # Remove legacy pickle files
                os.remove(filepath)
                removed += 1
        return removed

    def clear_cache(self, key: str | None = None) -> None:
        """Clear a specific key or all cached data."""
        if key:
            cache_file = os.path.join(self.cache_dir, _safe_cache_filename(key))
            if os.path.exists(cache_file):
                os.remove(cache_file)
            # Also remove legacy pickle file
            pkl_file = os.path.join(self.cache_dir, f"{key}.pkl")
            if os.path.exists(pkl_file):
                os.remove(pkl_file)
        else:
            for file in os.listdir(self.cache_dir):
                if file.endswith((".json", ".pkl")):
                    os.remove(os.path.join(self.cache_dir, file))


# Standalone convenience functions


def cache_data(key: str, data: Any, expiry_days: int = 30, cache_dir: str | None = None) -> None:
    """Store data in cache with key and expiry."""
    Cache(cache_dir=cache_dir).cache_data(key, data, expiry_days)


def get_cached_data(key: str, cache_dir: str | None = None) -> Any | None:
    """Retrieve cached data if not expired."""
    return Cache(cache_dir=cache_dir).get_cached_data(key)


def clear_cache(key: str | None = None, cache_dir: str | None = None) -> None:
    """Clear cache for a specific key or all keys."""
    Cache(cache_dir=cache_dir).clear_cache(key)


def cleanup_expired(cache_dir: str | None = None) -> int:
    """Remove all expired cache entries from disk."""
    return Cache(cache_dir=cache_dir).cleanup_expired()
