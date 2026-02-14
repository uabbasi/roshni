"""Tests for top-level tool loader resiliency."""

from unittest.mock import patch

from roshni.agent.tools import create_tools
from roshni.core.config import Config
from roshni.core.secrets import SecretsManager


def test_create_tools_logs_degraded_summary_on_integration_failure(tmp_path):
    config = Config(data_dir=str(tmp_path), defaults={"integrations": {"builtins": {"enabled": True}}})
    secrets = SecretsManager(providers=[])

    with (
        patch("roshni.agent.tools.builtin_tool.create_builtin_tools", side_effect=RuntimeError("boom")),
        patch("roshni.agent.tools.logger.warning") as mock_warning,
    ):
        _ = create_tools(config, secrets)

    warned = [str(call.args[0]) for call in mock_warning.call_args_list if call.args]
    assert any("Tool loader degraded; disabled integrations" in msg for msg in warned)
