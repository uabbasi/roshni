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


class SystemStateAdvisor:
    """Injects current system state for audit and temporal awareness.

    Always emits — gives the LLM accurate clock, process, and resource info.
    """

    name = "system_state"

    def advise(self, *, message: str, channel: str | None = None) -> str:
        import os
        import platform
        import time
        from datetime import datetime

        now = datetime.now()
        lines = [
            "[SYSTEM STATE — authoritative clock, always use this for current time]",
            f"Current time: {now.strftime('%I:%M %p')} ({now.strftime('%Y-%m-%d %H:%M:%S')} unix {int(time.time())})",
            f"Host: {platform.node()} | PID {os.getpid()}",
        ]

        try:
            import psutil

            proc = psutil.Process()
            uptime_s = time.time() - proc.create_time()
            h, rem = divmod(int(uptime_s), 3600)
            m, _s = divmod(rem, 60)
            mem_mb = proc.memory_info().rss / 1024 / 1024
            cpu = proc.cpu_percent(interval=0.1)
            lines.append(f"Uptime: {h}h{m:02d}m | Mem: {mem_mb:.0f} MB | CPU: {cpu:.1f}%")
        except ImportError:
            pass

        return "\n".join(lines)


__all__ = ["MemoryAdvisor", "SystemHealthAdvisor", "SystemStateAdvisor"]
