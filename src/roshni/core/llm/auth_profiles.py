"""Auth profile rotation — multi-key failover for LLM providers.

When multiple API keys are available for the same provider, this module
manages rotation and cooldown so that rate-limited or errored keys are
temporarily skipped.

Config format::

    llm:
      auth_profiles:
        - name: primary
          provider: anthropic
          api_key_secret: ANTHROPIC_API_KEY
        - name: secondary
          provider: anthropic
          api_key_secret: ANTHROPIC_API_KEY_2
"""

from __future__ import annotations

from dataclasses import dataclass
from time import time

from loguru import logger


@dataclass
class AuthProfile:
    """A single set of LLM API credentials."""

    name: str
    provider: str
    api_key: str
    model: str | None = None
    cooldown_until: float = 0.0


class AuthProfileManager:
    """Manages rotation across multiple :class:`AuthProfile` instances.

    Profiles are tried in order.  When a profile fails, it is put on cooldown
    and the next available profile is returned.
    """

    def __init__(self, profiles: list[AuthProfile]) -> None:
        self._profiles = list(profiles)
        self._current_idx = 0

    @property
    def profiles(self) -> list[AuthProfile]:
        return list(self._profiles)

    def get_active(self) -> AuthProfile | None:
        """Return the first non-cooled-down profile, or ``None``."""
        now = time()
        for profile in self._profiles:
            if profile.cooldown_until <= now:
                return profile
        return None

    def mark_failed(self, profile_name: str, cooldown_seconds: float = 60) -> None:
        """Put *profile_name* on cooldown for *cooldown_seconds*."""
        now = time()
        for profile in self._profiles:
            if profile.name == profile_name:
                profile.cooldown_until = now + cooldown_seconds
                logger.info(f"Auth profile '{profile_name}' on cooldown for {cooldown_seconds}s")
                return

    def mark_success(self, profile_name: str) -> None:
        """Clear cooldown for *profile_name*."""
        for profile in self._profiles:
            if profile.name == profile_name:
                profile.cooldown_until = 0.0
                return

    def rotate(self) -> AuthProfile | None:
        """Get the next available (non-cooled-down) profile after the current one.

        Cycles through all profiles once looking for one that is not on
        cooldown.  Returns ``None`` if every profile is cooled down.
        """
        now = time()
        n = len(self._profiles)
        for offset in range(1, n + 1):
            idx = (self._current_idx + offset) % n
            profile = self._profiles[idx]
            if profile.cooldown_until <= now:
                self._current_idx = idx
                logger.info(f"Rotated to auth profile '{profile.name}'")
                return profile
        return None
