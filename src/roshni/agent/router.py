"""Message router — dispatches messages to agents via configurable commands.

Parses slash commands and prefix-style routing (``cfo:``, ``ask:``), then
dispatches to the appropriate agent.  Command maps and agent factories
are injected at construction time.

Usage::

    router = Router(
        command_modes={"/analyze": "analyze", "/coach": "coach"},
        agent_factory=lambda: MyAgent(),
    )
    response, agent_name = await router.route("Hello!")
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger


@dataclass
class RouteResult:
    """Result of routing a message."""

    text: str
    """Response text."""

    agent_name: str = "agent"
    """Name of the agent that handled the message."""


@dataclass
class CommandParseResult:
    """Parsed slash command components."""

    agent_hint: str | None = None
    """Routing hint for a specialist agent (e.g. 'cfo')."""

    mode: str | None = None
    """Mode override (e.g. 'analyze', 'coach')."""

    query: str = ""
    """The cleaned query text with command prefix stripped."""


class Router:
    """Routes messages to agents via slash commands and keyword hints.

    Args:
        command_modes: Mapping of ``/command`` → mode string.
        prefix_routes: Mapping of ``prefix:`` → agent hint.
        keyword_patterns: List of ``(compiled_regex, agent_hint)`` for
            keyword-based routing.
        agent_factory: Callable that creates the primary agent instance.
        mode_labels: Optional mapping of mode/hint → display label.
        default_mode: Mode to use when no command is matched.
    """

    def __init__(
        self,
        *,
        command_modes: dict[str, str] | None = None,
        prefix_routes: dict[str, str] | None = None,
        keyword_patterns: list[tuple[re.Pattern, str]] | None = None,
        agent_factory: Callable[[], Any] | None = None,
        mode_labels: dict[str, str] | None = None,
        default_mode: str = "smart",
    ):
        self.command_modes = command_modes or {}
        self.prefix_routes = prefix_routes or {}
        self.keyword_patterns = keyword_patterns or []
        self._agent_factory = agent_factory
        self._agent: Any = None
        self.mode_labels = mode_labels or {}
        self.default_mode = default_mode

    def _get_agent(self) -> Any:
        """Lazy-init the primary agent."""
        if self._agent is None:
            if self._agent_factory is None:
                raise RuntimeError("No agent_factory configured on Router")
            self._agent = self._agent_factory()
            logger.info("Router: agent initialized")
        return self._agent

    @property
    def is_busy(self) -> bool:
        """True while the underlying agent is processing."""
        return self._agent is not None and hasattr(self._agent, "is_busy") and self._agent.is_busy

    def steer(self, message: str) -> None:
        """Pass-through steering to the active agent."""
        agent = self._get_agent()
        if hasattr(agent, "steer"):
            agent.steer(message)

    def enqueue_followup(self, message: str) -> None:
        """Pass-through follow-up to the active agent."""
        agent = self._get_agent()
        if hasattr(agent, "enqueue_followup"):
            agent.enqueue_followup(message)

    def parse_command(self, message: str) -> CommandParseResult:
        """Parse a message into routing components.

        Returns:
            :class:`CommandParseResult` with agent_hint, mode, and clean query.
        """
        lowered = message.lower()

        # Check prefix routes (e.g. "cfo:", "ask:")
        for prefix, hint in self.prefix_routes.items():
            if lowered.startswith(prefix):
                return CommandParseResult(
                    agent_hint=hint,
                    query=message[len(prefix) :].lstrip(),
                )

        # Non-slash messages — check keyword patterns
        if not message.startswith("/"):
            for pattern, hint in self.keyword_patterns:
                if pattern.search(message):
                    return CommandParseResult(agent_hint=hint, query=message)
            return CommandParseResult(query=message)

        # Slash commands
        parts = message.split(None, 1)
        cmd = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""

        if cmd in self.command_modes:
            return CommandParseResult(
                mode=self.command_modes[cmd],
                query=rest,
            )

        # Unknown slash command — pass through as-is
        return CommandParseResult(query=message)

    async def route(
        self,
        message: str,
        channel: str = "default",
        on_tool_start: Callable[[str, int, dict | None], None] | None = None,
    ) -> RouteResult:
        """Route a message through the configured agent.

        Args:
            message: User message text.
            channel: Channel identifier.
            on_tool_start: Optional progress callback.

        Returns:
            :class:`RouteResult` with response text and agent name.
        """
        parsed = self.parse_command(message)

        # Bare command with no query → acknowledge the mode
        if (parsed.agent_hint or parsed.mode) and not parsed.query.strip():
            label = self.mode_labels.get(
                parsed.mode or parsed.agent_hint or "",
                self.default_mode,
            )
            return RouteResult(text=f"{label} — send your next message.", agent_name="router")

        mode = parsed.mode or self.default_mode

        try:
            agent = self._get_agent()
            response = await agent.invoke(
                parsed.query,
                mode=mode,
                channel=channel,
                on_tool_start=on_tool_start,
            )
            return RouteResult(text=response, agent_name=agent.name if hasattr(agent, "name") else "agent")
        except Exception as e:
            logger.error(f"Agent failed: {e}")
            return RouteResult(text=f"Error: {e}", agent_name="agent")
