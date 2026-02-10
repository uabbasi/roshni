"""Tool policy layers — allowlist/blocklist filtering per channel and agent.

Provides fine-grained control over which tools are available in different
contexts (channels, agents) beyond the existing permission tier system.

Config format::

    tool_policy:
      global:
        blocklist: [admin_tool]
      channels:
        telegram:
          allowlist: [weather, search, vault_read]
        scheduled:
          blocklist: [send_email]
      agents:
        cfo:
          allowlist: [market_data, calculator, vault_read]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from roshni.agent.tools import ToolDefinition
    from roshni.core.config import Config


@dataclass
class ToolPolicy:
    """A single policy layer with optional allowlist and blocklist."""

    allowlist: set[str] | None = None
    blocklist: set[str] | None = None


@dataclass
class LayeredToolPolicy:
    """Composes global, channel-specific, and agent-specific policies.

    Filtering order:
    1. Global policy
    2. Channel-specific policy (if channel matches)
    3. Agent-specific policy (if agent matches)

    Blocklist always wins over allowlist at each layer.
    """

    global_policy: ToolPolicy | None = None
    channel_policies: dict[str, ToolPolicy] = field(default_factory=dict)
    agent_policies: dict[str, ToolPolicy] = field(default_factory=dict)

    def _apply_policy(self, tool_names: set[str], policy: ToolPolicy) -> set[str]:
        """Apply a single policy layer to a set of tool names."""
        if policy.allowlist is not None:
            tool_names = tool_names & policy.allowlist
        if policy.blocklist:
            tool_names -= policy.blocklist
        return tool_names

    def filter_tools(
        self,
        tools: list[ToolDefinition],
        *,
        channel: str | None = None,
        agent_name: str | None = None,
    ) -> list[ToolDefinition]:
        """Return only tools permitted by all applicable policy layers."""
        allowed_names = {t.name for t in tools}

        if self.global_policy:
            allowed_names = self._apply_policy(allowed_names, self.global_policy)

        if channel and channel in self.channel_policies:
            allowed_names = self._apply_policy(allowed_names, self.channel_policies[channel])

        if agent_name and agent_name in self.agent_policies:
            allowed_names = self._apply_policy(allowed_names, self.agent_policies[agent_name])

        return [t for t in tools if t.name in allowed_names]

    def is_tool_allowed(
        self,
        tool_name: str,
        *,
        channel: str | None = None,
        agent_name: str | None = None,
    ) -> bool:
        """Check whether a single tool name passes all policy layers."""
        names = {tool_name}

        if self.global_policy:
            names = self._apply_policy(names, self.global_policy)

        if channel and channel in self.channel_policies:
            names = self._apply_policy(names, self.channel_policies[channel])

        if agent_name and agent_name in self.agent_policies:
            names = self._apply_policy(names, self.agent_policies[agent_name])

        return bool(names)


def load_tool_policy(config: Config) -> LayeredToolPolicy | None:
    """Build a :class:`LayeredToolPolicy` from the ``tool_policy`` config section.

    Returns ``None`` if no policy is configured.
    """
    raw: dict[str, Any] = config.get("tool_policy", {}) or {}
    if not raw:
        return None

    def _parse_policy(data: dict[str, Any] | None) -> ToolPolicy | None:
        if not data:
            return None
        allowlist = set(data["allowlist"]) if data.get("allowlist") else None
        blocklist = set(data["blocklist"]) if data.get("blocklist") else None
        return ToolPolicy(allowlist=allowlist, blocklist=blocklist)

    global_policy = _parse_policy(raw.get("global"))

    channel_policies: dict[str, ToolPolicy] = {}
    for ch_name, ch_data in (raw.get("channels") or {}).items():
        policy = _parse_policy(ch_data)
        if policy:
            channel_policies[ch_name] = policy

    agent_policies: dict[str, ToolPolicy] = {}
    for ag_name, ag_data in (raw.get("agents") or {}).items():
        policy = _parse_policy(ag_data)
        if policy:
            agent_policies[ag_name] = policy

    return LayeredToolPolicy(
        global_policy=global_policy,
        channel_policies=channel_policies,
        agent_policies=agent_policies,
    )
