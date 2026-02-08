"""Tests for roshni.agent.circuit_breaker."""

import time

from roshni.agent.circuit_breaker import CircuitBreaker, CircuitBreakerConfig


class TestCircuitBreaker:
    def test_initially_available(self):
        cb = CircuitBreaker()
        assert cb.is_available("search")

    def test_stays_available_after_success(self):
        cb = CircuitBreaker()
        cb.record("search", success=True, duration=0.1)
        cb.record("search", success=True, duration=0.2)
        assert cb.is_available("search")

    def test_opens_after_consecutive_failures(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3, open_duration=60))
        cb.record("search", success=False, duration=0.0)
        cb.record("search", success=False, duration=0.0)
        assert cb.is_available("search")  # Only 2, need 3
        cb.record("search", success=False, duration=0.0)
        assert not cb.is_available("search")  # Now open

    def test_success_resets_consecutive_count(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
        cb.record("search", success=False, duration=0.0)
        cb.record("search", success=False, duration=0.0)
        cb.record("search", success=True, duration=0.1)  # Resets
        cb.record("search", success=False, duration=0.0)
        assert cb.is_available("search")  # Not 3 consecutive

    def test_circuit_reopens_after_duration(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, open_duration=0.1))
        cb.record("search", success=False, duration=0.0)
        cb.record("search", success=False, duration=0.0)
        assert not cb.is_available("search")
        time.sleep(0.15)
        assert cb.is_available("search")

    def test_manual_reset(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2, open_duration=999))
        cb.record("search", success=False, duration=0.0)
        cb.record("search", success=False, duration=0.0)
        assert not cb.is_available("search")
        cb.reset("search")
        assert cb.is_available("search")

    def test_independent_services(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2))
        cb.record("a", success=False, duration=0.0)
        cb.record("a", success=False, duration=0.0)
        cb.record("b", success=True, duration=0.1)
        assert not cb.is_available("a")
        assert cb.is_available("b")

    def test_get_status(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=2))
        cb.record("a", success=True, duration=0.1)
        cb.record("a", success=False, duration=0.0)
        status = cb.get_status()
        assert status["a"]["total_calls"] == 2
        assert status["a"]["successes"] == 1
        assert status["a"]["failures"] == 1
        assert status["a"]["circuit_open"] is False

    def test_custom_history_size(self):
        cb = CircuitBreaker(CircuitBreakerConfig(history_size=5))
        for _ in range(10):
            cb.record("s", success=True, duration=0.0)
        assert cb.get_status()["s"]["total_calls"] == 5
