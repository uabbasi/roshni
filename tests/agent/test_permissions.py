"""Tests for roshni.agent.permissions."""

from roshni.agent.permissions import PermissionTier, filter_tools_by_tier, get_domain_tier
from roshni.agent.tools import ToolDefinition


def _tool(name: str, permission: str = "read") -> ToolDefinition:
    return ToolDefinition(name=name, description="", parameters={}, function=lambda: "", permission=permission)


# -- PermissionTier enum --


class TestPermissionTierValues:
    def test_enum_values(self):
        assert PermissionTier.NONE == 0
        assert PermissionTier.OBSERVE == 1
        assert PermissionTier.INTERACT == 2
        assert PermissionTier.FULL == 3

    def test_ordering(self):
        assert PermissionTier.NONE < PermissionTier.OBSERVE
        assert PermissionTier.OBSERVE < PermissionTier.INTERACT
        assert PermissionTier.INTERACT < PermissionTier.FULL


# -- get_domain_tier --


class TestGetDomainTier:
    def test_reads_from_config(self):
        cfg = {"gmail": "full", "trello": "observe"}
        assert get_domain_tier(cfg, "gmail") == PermissionTier.FULL
        assert get_domain_tier(cfg, "trello") == PermissionTier.OBSERVE

    def test_case_insensitive(self):
        assert get_domain_tier({"x": "INTERACT"}, "x") == PermissionTier.INTERACT
        assert get_domain_tier({"x": "interact"}, "x") == PermissionTier.INTERACT

    def test_integer_value(self):
        assert get_domain_tier({"x": 3}, "x") == PermissionTier.FULL

    def test_missing_key_returns_default(self):
        assert get_domain_tier({}, "gmail") == PermissionTier.INTERACT

    def test_custom_default(self):
        assert get_domain_tier({}, "gmail", default=PermissionTier.FULL) == PermissionTier.FULL

    def test_invalid_value_returns_default(self):
        assert get_domain_tier({"x": "bogus"}, "x") == PermissionTier.INTERACT


# -- filter_tools_by_tier --


class TestFilterToolsByTier:
    def test_none_returns_empty(self):
        tools = [_tool("a", "read"), _tool("b", "write")]
        assert filter_tools_by_tier(tools, PermissionTier.NONE) == []

    def test_observe_returns_read_only(self):
        tools = [_tool("r", "read"), _tool("w", "write"), _tool("s", "send")]
        result = filter_tools_by_tier(tools, PermissionTier.OBSERVE)
        assert [t.name for t in result] == ["r"]

    def test_interact_returns_read_and_write(self):
        tools = [_tool("r", "read"), _tool("w", "write"), _tool("s", "send"), _tool("a", "admin")]
        result = filter_tools_by_tier(tools, PermissionTier.INTERACT)
        assert [t.name for t in result] == ["r", "w"]

    def test_full_returns_all(self):
        tools = [_tool("r", "read"), _tool("w", "write"), _tool("s", "send"), _tool("a", "admin")]
        result = filter_tools_by_tier(tools, PermissionTier.FULL)
        assert [t.name for t in result] == ["r", "w", "s", "a"]

    def test_empty_list_returns_empty(self):
        assert filter_tools_by_tier([], PermissionTier.FULL) == []
