"""Circuit breaker for specialist/service health tracking.

Monitors success/failure of delegated calls and opens the circuit
(blocks further calls) after consecutive failures. The circuit
auto-resets after a configurable cooldown period.
"""

import time
from collections import deque
from dataclasses import dataclass


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 3
    """Consecutive failures required to open the circuit."""

    open_duration: float = 300.0
    """Seconds to keep the circuit open before allowing retries."""

    history_size: int = 20
    """Rolling window of call outcomes to retain."""


class CircuitBreaker:
    """Track service success/failure and implement circuit-breaker pattern.

    Usage::

        cb = CircuitBreaker()
        if cb.is_available("search"):
            try:
                result = do_search(query)
                cb.record("search", success=True, duration=0.5)
            except Exception:
                cb.record("search", success=False, duration=0.0)
        else:
            # Circuit is open — skip or use fallback
            ...
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self.config = config or CircuitBreakerConfig()
        self._history: dict[str, deque] = {}
        self._open_until: dict[str, float] = {}

    def record(self, service: str, *, success: bool, duration: float = 0.0) -> None:
        """Record a call outcome for *service*."""
        if service not in self._history:
            self._history[service] = deque(maxlen=self.config.history_size)
        self._history[service].append((time.monotonic(), success, duration))

        recent = list(self._history[service])[-self.config.failure_threshold :]
        if len(recent) >= self.config.failure_threshold and all(not r[1] for r in recent):
            self._open_until[service] = time.monotonic() + self.config.open_duration

    def is_available(self, service: str) -> bool:
        """Return True if *service* circuit is closed (available)."""
        deadline = self._open_until.get(service, 0.0)
        return time.monotonic() >= deadline

    def reset(self, service: str) -> None:
        """Manually close the circuit for *service*."""
        self._open_until.pop(service, None)

    def get_status(self) -> dict[str, dict]:
        """Return debug info about all tracked services."""
        status: dict[str, dict] = {}
        for service, hist in self._history.items():
            recent = list(hist)
            successes = sum(1 for _, s, _ in recent if s)
            status[service] = {
                "total_calls": len(recent),
                "successes": successes,
                "failures": len(recent) - successes,
                "circuit_open": not self.is_available(service),
            }
        return status
