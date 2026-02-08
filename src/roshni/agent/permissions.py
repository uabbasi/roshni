"""Permission tier system for tool access control.

Tiers act as an upstream filter — they decide which tools get created at all,
before the existing per-tool approval gate kicks in.
"""

from __future__ import annotations

from enum import IntEnum

from roshni.agent.tools import ToolDefinition

# Permission string → minimum tier required
_PERMISSION_MIN_TIER: dict[str, int] = {
    "read": 1,  # OBSERVE
    "write": 2,  # INTERACT
    "send": 3,  # FULL
    "admin": 3,  # FULL
}


class PermissionTier(IntEnum):
    """Escalating levels of tool access."""

    NONE = 0
    OBSERVE = 1
    INTERACT = 2
    FULL = 3


def get_domain_tier(
    permissions_cfg: dict,
    domain: str,
    default: PermissionTier = PermissionTier.INTERACT,
) -> PermissionTier:
    """Read a domain's tier from a permissions config dict.

    Expected shape: ``{"gmail": "full", "trello": "observe", ...}``
    Values are case-insensitive tier names or int levels.
    """
    raw = permissions_cfg.get(domain)
    if raw is None:
        return default
    if isinstance(raw, int):
        return PermissionTier(raw)
    name = str(raw).strip().upper()
    try:
        return PermissionTier[name]
    except KeyError:
        return default


def filter_tools_by_tier(
    tools: list[ToolDefinition],
    tier: PermissionTier,
) -> list[ToolDefinition]:
    """Return only the tools whose permission level fits within *tier*."""
    if tier == PermissionTier.NONE:
        return []
    return [t for t in tools if _PERMISSION_MIN_TIER.get(t.permission, 3) <= tier]
