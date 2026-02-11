"""Built-in advisors for the agent framework."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roshni.agent.circuit_breaker import CircuitBreaker
    from roshni.agent.memory import MemoryManager


class MemoryAdvisor:
    """Injects memory context + daily notes into system prompt."""

    name = "memory"

    def __init__(self, memory_manager: MemoryManager) -> None:
        self._mm = memory_manager

    def advise(self, *, message: str, channel: str | None = None) -> str:
        parts: list[str] = []
        mem = self._mm.get_context()
        if mem:
            parts.append(mem)
        daily = self._mm.get_daily_context()
        if daily:
            parts.append(daily)
        return "\n\n".join(parts)


class SystemHealthAdvisor:
    """Injects system health signals so the agent can adapt behavior.

    Zero-cost when healthy — returns empty string, nothing injected.
    Only adds tokens when there's something actionable to communicate.
    """

    name = "system_health"

    def __init__(
        self,
        circuit_breaker: CircuitBreaker | None = None,
        budget_threshold: float = 0.6,
    ) -> None:
        self._cb = circuit_breaker
        self._budget_threshold = budget_threshold

    def advise(self, *, message: str, channel: str | None = None) -> str:
        parts: list[str] = []

        # Budget pressure
        from roshni.core.llm.token_budget import get_budget_pressure

        pressure = get_budget_pressure()
        if pressure > self._budget_threshold:
            pct = int(pressure * 100)
            if pressure > 0.9:
                parts.append(f"[BUDGET] {pct}% of daily budget used. Be very concise. Avoid unnecessary tool calls.")
            else:
                parts.append(f"[BUDGET] {pct}% of daily budget used. Prefer efficiency.")

        # Circuit breaker status
        if self._cb:
            status = self._cb.get_status()
            broken = [svc for svc, info in status.items() if info["circuit_open"]]
            if broken:
                parts.append(f"[SERVICES DOWN] {', '.join(broken)} — circuit open, do not attempt.")

            degraded = []
            for svc, info in status.items():
                if not info["circuit_open"] and info["total_calls"] >= 3:
                    fail_rate = info["failures"] / info["total_calls"]
                    if fail_rate > 0.5:
                        degraded.append(f"{svc} ({int(fail_rate * 100)}% failures)")
            if degraded:
                parts.append(f"[DEGRADED] {', '.join(degraded)} — use with caution.")

        if not parts:
            return ""
        return "\n".join(parts)


__all__ = ["MemoryAdvisor", "SystemHealthAdvisor"]
