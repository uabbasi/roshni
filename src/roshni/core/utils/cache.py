"""
Pickle-based caching with TTL expiry.

Provides both a class-based API and standalone convenience functions.
The cache directory is configurable — no hardcoded paths.

Note: Uses pickle for serialization (matching the existing weeklies caching
pattern). Only caches data that the consumer explicitly stores — this is a
local-only cache for trusted data, not for untrusted external input.
"""

import os
import pickle
from datetime import datetime
from typing import Any


class Cache:
    """General-purpose pickle cache with TTL-based expiry."""

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
        """Store data with a TTL."""
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        with open(cache_file, "wb") as f:
            pickle.dump(
                {"data": data, "expiry": datetime.now().timestamp() + (expiry_days * 86400)},
                f,
            )

    def get_cached_data(self, key: str) -> Any | None:
        """Retrieve cached data if not expired."""
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        if not os.path.exists(cache_file):
            return None
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
            if cached["expiry"] > datetime.now().timestamp():
                return cached["data"]
        return None

    def clear_cache(self, key: str | None = None) -> None:
        """Clear a specific key or all cached data."""
        if key:
            cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
            if os.path.exists(cache_file):
                os.remove(cache_file)
        else:
            for file in os.listdir(self.cache_dir):
                if file.endswith(".pkl"):
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
