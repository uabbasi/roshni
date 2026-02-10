"""Tests for auth profile rotation / multi-key failover."""

import time

import pytest

from roshni.core.llm.auth_profiles import AuthProfile, AuthProfileManager


@pytest.fixture
def profiles():
    return [
        AuthProfile(name="primary", provider="anthropic", api_key="key-1"),
        AuthProfile(name="secondary", provider="anthropic", api_key="key-2"),
        AuthProfile(name="tertiary", provider="anthropic", api_key="key-3"),
    ]


class TestAuthProfileManager:
    def test_get_active_returns_first(self, profiles):
        mgr = AuthProfileManager(profiles)
        active = mgr.get_active()
        assert active is not None
        assert active.name == "primary"

    def test_mark_failed_puts_on_cooldown(self, profiles):
        mgr = AuthProfileManager(profiles)
        mgr.mark_failed("primary", cooldown_seconds=60)
        active = mgr.get_active()
        assert active is not None
        assert active.name == "secondary"

    def test_all_failed_returns_none(self, profiles):
        mgr = AuthProfileManager(profiles)
        for p in profiles:
            mgr.mark_failed(p.name, cooldown_seconds=60)
        assert mgr.get_active() is None

    def test_mark_success_clears_cooldown(self, profiles):
        mgr = AuthProfileManager(profiles)
        mgr.mark_failed("primary", cooldown_seconds=60)
        assert mgr.get_active().name == "secondary"
        mgr.mark_success("primary")
        assert mgr.get_active().name == "primary"

    def test_rotate_skips_failed(self, profiles):
        mgr = AuthProfileManager(profiles)
        mgr.mark_failed("secondary", cooldown_seconds=60)
        rotated = mgr.rotate()
        assert rotated is not None
        # Should skip secondary and go to tertiary
        assert rotated.name in ("secondary", "tertiary")
        # Actually the rotation starts from current_idx + 1
        # current_idx=0, so tries idx 1 (secondary, failed), then idx 2 (tertiary)
        assert rotated.name == "tertiary"

    def test_rotate_wraps_around(self, profiles):
        mgr = AuthProfileManager(profiles)
        mgr._current_idx = 2  # at tertiary
        rotated = mgr.rotate()
        assert rotated is not None
        assert rotated.name == "primary"

    def test_rotate_all_failed_returns_none(self, profiles):
        mgr = AuthProfileManager(profiles)
        for p in profiles:
            mgr.mark_failed(p.name, cooldown_seconds=60)
        assert mgr.rotate() is None

    def test_cooldown_expires(self, profiles):
        mgr = AuthProfileManager(profiles)
        mgr.mark_failed("primary", cooldown_seconds=0.01)
        time.sleep(0.02)
        active = mgr.get_active()
        assert active.name == "primary"

    def test_empty_profiles(self):
        mgr = AuthProfileManager([])
        assert mgr.get_active() is None
        assert mgr.rotate() is None

    def test_profiles_property_returns_copy(self, profiles):
        mgr = AuthProfileManager(profiles)
        copy = mgr.profiles
        assert len(copy) == 3
        copy.pop()
        assert len(mgr.profiles) == 3  # original unchanged
