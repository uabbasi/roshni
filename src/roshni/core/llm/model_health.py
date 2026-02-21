"""Model-level health tracking using the existing CircuitBreaker.

Tracks success/failure of individual models and providers so the LLM
client and model selector can route around unhealthy targets proactively.

Uses a module-level singleton (same pattern as ``model_selector.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roshni.agent.circuit_breaker import CircuitBreaker

# Module-level singleton — shared across all LLM client instances
_model_health: CircuitBreaker | None = None


def _get_health_tracker() -> CircuitBreaker:
    """Get or create the module-level model health tracker."""
    # Lazy import to avoid circular: model_health → agent → model_selector
    from roshni.agent.circuit_breaker import CircuitBreaker, CircuitBreakerConfig

    global _model_health
    if _model_health is None:
        _model_health = CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=3,
                open_duration=120.0,  # 2 minutes before retrying an unhealthy model
                history_size=10,
            )
        )
    return _model_health


def is_model_healthy(model: str) -> bool:
    """Return True if the model circuit is closed (available)."""
    return _get_health_tracker().is_available(f"model:{model}")


def is_provider_healthy(provider: str) -> bool:
    """Return True if the provider circuit is closed (available)."""
    return _get_health_tracker().is_available(f"provider:{provider}")


def record_model_outcome(model: str, *, success: bool, duration: float = 0.0) -> None:
    """Record a model call outcome for circuit breaker tracking."""
    tracker = _get_health_tracker()
    tracker.record(f"model:{model}", success=success, duration=duration)
    # Also track at provider level
    provider = model.split("/")[0] if "/" in model else "unknown"
    tracker.record(f"provider:{provider}", success=success, duration=duration)


def get_model_health_status() -> dict[str, dict]:
    """Return debug info about all tracked models/providers."""
    return _get_health_tracker().get_status()


def reset_model_health() -> None:
    """Reset the model health tracker (useful for testing)."""
    global _model_health
    _model_health = None
