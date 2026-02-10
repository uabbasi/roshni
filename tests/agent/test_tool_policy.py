"""Tests for tool policy layers."""

import pytest

from roshni.agent.tool_policy import LayeredToolPolicy, ToolPolicy, load_tool_policy
from roshni.agent.tools import ToolDefinition


def _make_tool(name: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description=f"Tool {name}",
        parameters={"type": "object", "properties": {}, "required": []},
        function=lambda: "ok",
    )


@pytest.fixture
def all_tools():
    return [_make_tool(n) for n in ["weather", "search", "admin_tool", "send_email", "vault_read", "calculator"]]


class TestToolPolicy:
    def test_allowlist_only(self, all_tools):
        policy = LayeredToolPolicy(global_policy=ToolPolicy(allowlist={"weather", "search"}))
        filtered = policy.filter_tools(all_tools)
        names = {t.name for t in filtered}
        assert names == {"weather", "search"}

    def test_blocklist_only(self, all_tools):
        policy = LayeredToolPolicy(global_policy=ToolPolicy(blocklist={"admin_tool"}))
        filtered = policy.filter_tools(all_tools)
        names = {t.name for t in filtered}
        assert "admin_tool" not in names
        assert len(filtered) == 5

    def test_blocklist_wins_over_allowlist(self, all_tools):
        policy = LayeredToolPolicy(
            global_policy=ToolPolicy(allowlist={"weather", "admin_tool"}, blocklist={"admin_tool"})
        )
        filtered = policy.filter_tools(all_tools)
        names = {t.name for t in filtered}
        assert names == {"weather"}

    def test_no_policy_returns_all(self, all_tools):
        policy = LayeredToolPolicy()
        filtered = policy.filter_tools(all_tools)
        assert len(filtered) == len(all_tools)


class TestLayeredFiltering:
    def test_channel_policy_applied(self, all_tools):
        policy = LayeredToolPolicy(
            channel_policies={"telegram": ToolPolicy(allowlist={"weather", "search", "vault_read"})}
        )
        filtered = policy.filter_tools(all_tools, channel="telegram")
        names = {t.name for t in filtered}
        assert names == {"weather", "search", "vault_read"}

    def test_agent_policy_applied(self, all_tools):
        policy = LayeredToolPolicy(agent_policies={"cfo": ToolPolicy(allowlist={"calculator", "vault_read"})})
        filtered = policy.filter_tools(all_tools, agent_name="cfo")
        names = {t.name for t in filtered}
        assert names == {"calculator", "vault_read"}

    def test_global_then_channel(self, all_tools):
        policy = LayeredToolPolicy(
            global_policy=ToolPolicy(blocklist={"admin_tool"}),
            channel_policies={"scheduled": ToolPolicy(blocklist={"send_email"})},
        )
        filtered = policy.filter_tools(all_tools, channel="scheduled")
        names = {t.name for t in filtered}
        assert "admin_tool" not in names
        assert "send_email" not in names

    def test_unmatched_channel_uses_global_only(self, all_tools):
        policy = LayeredToolPolicy(
            global_policy=ToolPolicy(blocklist={"admin_tool"}),
            channel_policies={"telegram": ToolPolicy(allowlist={"weather"})},
        )
        filtered = policy.filter_tools(all_tools, channel="web")
        names = {t.name for t in filtered}
        assert "admin_tool" not in names
        assert len(filtered) == 5  # all except admin_tool


class TestIsToolAllowed:
    def test_allowed(self):
        policy = LayeredToolPolicy(global_policy=ToolPolicy(allowlist={"weather"}))
        assert policy.is_tool_allowed("weather") is True

    def test_blocked(self):
        policy = LayeredToolPolicy(global_policy=ToolPolicy(blocklist={"admin_tool"}))
        assert policy.is_tool_allowed("admin_tool") is False

    def test_channel_blocked(self):
        policy = LayeredToolPolicy(channel_policies={"telegram": ToolPolicy(blocklist={"send_email"})})
        assert policy.is_tool_allowed("send_email", channel="telegram") is False
        assert policy.is_tool_allowed("send_email", channel="web") is True


class TestLoadToolPolicy:
    def test_returns_none_when_no_config(self, tmp_dir):
        from roshni.core.config import Config

        cfg = Config(data_dir=tmp_dir, defaults={})
        result = load_tool_policy(cfg)
        assert result is None

    def test_parses_full_config(self, tmp_dir):
        from roshni.core.config import Config

        cfg = Config(
            data_dir=tmp_dir,
            defaults={
                "tool_policy": {
                    "global": {"blocklist": ["admin_tool"]},
                    "channels": {"telegram": {"allowlist": ["weather", "search"]}},
                    "agents": {"cfo": {"allowlist": ["calculator"]}},
                }
            },
        )
        policy = load_tool_policy(cfg)
        assert policy is not None
        assert policy.global_policy is not None
        assert "admin_tool" in policy.global_policy.blocklist
        assert "telegram" in policy.channel_policies
        assert "cfo" in policy.agent_policies
