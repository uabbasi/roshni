"""Aggregated health check — single call to assess system health.

Combines budget pressure, circuit breaker status, and model health
into one cohesive status object for monitoring and dashboards.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HealthStatus:
    """Snapshot of system health across all subsystems."""

    healthy: bool = True
    """Overall health — False if any critical subsystem is degraded."""

    budget_pressure: float = 0.0
    """0.0 = fine, 1.0 = exhausted."""

    budget_within_limit: bool = True
    """Whether the daily budget is still available."""

    model_health: dict[str, dict] = field(default_factory=dict)
    """Per-model/provider circuit breaker status."""

    tool_health: dict[str, dict] = field(default_factory=dict)
    """Per-service circuit breaker status."""

    warnings: list[str] = field(default_factory=list)
    """Human-readable warnings for degraded subsystems."""


def check_health(circuit_breaker=None) -> HealthStatus:
    """Assess system health across budget, models, and tools.

    Args:
        circuit_breaker: Optional ``CircuitBreaker`` instance for tool health.
            If None, tool health is not checked.

    Returns:
        A :class:`HealthStatus` snapshot.
    """
    status = HealthStatus()

    # Budget check
    from roshni.core.llm.token_budget import check_budget, get_budget_pressure

    status.budget_pressure = get_budget_pressure()
    within_budget, _remaining = check_budget()
    status.budget_within_limit = within_budget

    if not within_budget:
        status.healthy = False
        status.warnings.append("Daily token budget exhausted")
    elif status.budget_pressure >= 0.80:
        status.warnings.append(f"Budget pressure high ({status.budget_pressure:.0%})")

    # Model health
    from roshni.core.llm.model_health import get_model_health_status

    status.model_health = get_model_health_status()
    for name, info in status.model_health.items():
        if info.get("circuit_open"):
            status.warnings.append(f"Model/provider circuit open: {name}")
            if name.startswith("provider:"):
                status.healthy = False  # Provider-level outage is critical

    # Tool / service health
    if circuit_breaker is not None:
        status.tool_health = circuit_breaker.get_status()
        for service, info in status.tool_health.items():
            if info.get("circuit_open"):
                status.warnings.append(f"Service circuit open: {service}")

    return status
