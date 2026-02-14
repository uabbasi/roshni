"""Tests for roshni.core.config_schema and Config.validated()."""

import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from roshni.core.config import Config, reset_config
from roshni.core.config_schema import RoshniConfig


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_config()
    yield
    reset_config()


@pytest.mark.smoke
class TestConfigSchema:
    def test_valid_config_roundtrip(self):
        data = {
            "paths": {"data_dir": "/tmp/test-data", "cache_dir": "/tmp/test-cache"},
            "llm": {
                "default": "anthropic",
                "providers": {"anthropic": {"model": "claude-sonnet-4-5-20250929", "temperature": 0.5}},
                "mode_overrides": {"smart": "openai/gpt-5.2-chat-latest"},
                "selector": {
                    "tool_result_chars_threshold": 800,
                    "complex_query_chars_threshold": 220,
                },
            },
            "bot": {"name": "mybot", "token": "tok-123"},
            "security": {"require_write_approval": False},
            "integrations": {"gmail": {"enabled": False}},
            "vault": {"path": "/some/vault"},
        }
        cfg = RoshniConfig.model_validate(data)
        assert cfg.paths.data_dir == Path("/tmp/test-data")
        assert cfg.llm.providers["anthropic"].model == "claude-sonnet-4-5-20250929"
        assert cfg.llm.mode_overrides["smart"] == "openai/gpt-5.2-chat-latest"
        assert cfg.llm.selector.tool_result_chars_threshold == 800
        assert cfg.bot.name == "mybot"
        assert cfg.security.require_write_approval is False
        assert cfg.integrations.gmail.enabled is False
        assert cfg.vault.path == "/some/vault"

    def test_defaults_populate(self):
        cfg = RoshniConfig()
        assert cfg.paths.data_dir is not None
        assert cfg.llm.default == "anthropic"
        assert cfg.bot.name == "roshni"
        assert cfg.security.require_write_approval is True
        assert cfg.integrations.trello.enabled is True
        assert cfg.vault.path is None

    def test_path_expansion(self):
        cfg = RoshniConfig.model_validate({"paths": {"data_dir": "~/.roshni-data"}})
        assert cfg.paths.data_dir.is_absolute()
        assert "~" not in str(cfg.paths.data_dir)

    def test_llm_provider_cross_validation_fails(self):
        with pytest.raises(ValidationError, match="default provider"):
            RoshniConfig.model_validate(
                {
                    "paths": {"data_dir": "/tmp/d"},
                    "llm": {
                        "default": "foo",
                        "providers": {"bar": {"model": "m"}},
                    },
                }
            )

    def test_llm_provider_cross_validation_passes(self):
        cfg = RoshniConfig.model_validate(
            {
                "paths": {"data_dir": "/tmp/d"},
                "llm": {
                    "default": "bar",
                    "providers": {"bar": {"model": "m"}},
                },
            }
        )
        assert cfg.llm.default == "bar"

    def test_extra_keys_allowed_at_root(self):
        cfg = RoshniConfig.model_validate(
            {
                "paths": {"data_dir": "/tmp/d"},
                "custom_section": {"key": "value"},
            }
        )
        assert cfg.model_extra["custom_section"] == {"key": "value"}

    def test_config_validated_integration(self, tmp_dir):
        config_data = {
            "paths": {
                "data_dir": os.path.join(tmp_dir, "data"),
                "cache_dir": os.path.join(tmp_dir, "cache"),
            },
            "llm": {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"},
        }
        config_path = os.path.join(tmp_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = Config(config_file=config_path, data_dir=tmp_dir)
        validated = config.validated()
        assert isinstance(validated, RoshniConfig)
        assert validated.paths.data_dir == Path(tmp_dir) / "data"
