"""Tests for roshni.core.utils.cache."""

import os
import time

from roshni.core.utils.cache import Cache, cache_data, cleanup_expired, clear_cache, get_cached_data


class TestCache:
    def test_store_and_retrieve(self, tmp_dir):
        cache = Cache(cache_dir=tmp_dir)
        cache.cache_data("test_key", {"hello": "world"})
        result = cache.get_cached_data("test_key")
        assert result == {"hello": "world"}

    def test_expired_data_is_cleaned_up(self, tmp_dir):
        cache = Cache(cache_dir=tmp_dir)
        # Store with 0 days expiry (already expired)
        cache.cache_data("expired", "data", expiry_days=0)
        # Tiny sleep to ensure timestamp passes
        time.sleep(0.01)
        assert cache.get_cached_data("expired") is None
        # Stale file should have been removed from disk
        assert not os.path.exists(os.path.join(tmp_dir, "expired.pkl"))

    def test_missing_key(self, tmp_dir):
        cache = Cache(cache_dir=tmp_dir)
        assert cache.get_cached_data("nonexistent") is None

    def test_clear_specific_key(self, tmp_dir):
        cache = Cache(cache_dir=tmp_dir)
        cache.cache_data("keep", "data1")
        cache.cache_data("remove", "data2")
        cache.clear_cache("remove")
        assert cache.get_cached_data("keep") == "data1"
        assert cache.get_cached_data("remove") is None

    def test_clear_all(self, tmp_dir):
        cache = Cache(cache_dir=tmp_dir)
        cache.cache_data("a", 1)
        cache.cache_data("b", 2)
        cache.clear_cache()
        assert cache.get_cached_data("a") is None
        assert cache.get_cached_data("b") is None

    def test_creates_directory(self, tmp_dir):
        cache_dir = os.path.join(tmp_dir, "nested", "cache")
        cache = Cache(cache_dir=cache_dir)
        assert os.path.isdir(cache_dir)
        cache.cache_data("test", "value")
        assert cache.get_cached_data("test") == "value"

    def test_cleanup_expired_removes_stale_entries(self, tmp_dir):
        cache = Cache(cache_dir=tmp_dir)
        cache.cache_data("fresh", "keep_me", expiry_days=30)
        cache.cache_data("stale1", "remove_me", expiry_days=0)
        cache.cache_data("stale2", "remove_me_too", expiry_days=0)
        time.sleep(0.01)

        removed = cache.cleanup_expired()

        assert removed == 2
        assert cache.get_cached_data("fresh") == "keep_me"
        assert not os.path.exists(os.path.join(tmp_dir, "stale1.pkl"))
        assert not os.path.exists(os.path.join(tmp_dir, "stale2.pkl"))

    def test_cleanup_expired_returns_zero_when_nothing_expired(self, tmp_dir):
        cache = Cache(cache_dir=tmp_dir)
        cache.cache_data("a", 1, expiry_days=30)
        cache.cache_data("b", 2, expiry_days=30)

        assert cache.cleanup_expired() == 0

    def test_cleanup_expired_handles_corrupted_files(self, tmp_dir):
        cache = Cache(cache_dir=tmp_dir)
        # Write a corrupted pickle file
        corrupted_path = os.path.join(tmp_dir, "corrupted.pkl")
        with open(corrupted_path, "wb") as f:
            f.write(b"not a valid pickle")

        removed = cache.cleanup_expired()

        assert removed == 1
        assert not os.path.exists(corrupted_path)


class TestStandaloneFunctions:
    def test_roundtrip(self, tmp_dir):
        cache_data("key", [1, 2, 3], cache_dir=tmp_dir)
        assert get_cached_data("key", cache_dir=tmp_dir) == [1, 2, 3]

    def test_clear(self, tmp_dir):
        cache_data("key", "value", cache_dir=tmp_dir)
        clear_cache("key", cache_dir=tmp_dir)
        assert get_cached_data("key", cache_dir=tmp_dir) is None

    def test_cleanup_expired(self, tmp_dir):
        cache_data("stale", "old", expiry_days=0, cache_dir=tmp_dir)
        cache_data("fresh", "new", expiry_days=30, cache_dir=tmp_dir)
        time.sleep(0.01)
        removed = cleanup_expired(cache_dir=tmp_dir)
        assert removed == 1
        assert get_cached_data("fresh", cache_dir=tmp_dir) == "new"
