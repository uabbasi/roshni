"""Tests for roshni.core.config."""

import os

import pytest
import yaml

from roshni.core.config import Config, get_config, reset_config


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset config singleton between tests."""
    reset_config()
    yield
    reset_config()


class TestConfig:
    def test_defaults(self):
        config = Config()
        data_dir = config.get("paths.data_dir")
        assert data_dir.endswith(".roshni-data")

    def test_custom_data_dir(self, tmp_dir):
        config = Config(data_dir=tmp_dir)
        assert config.get("paths.data_dir") == tmp_dir
        assert config.get("paths.cache_dir") == os.path.join(tmp_dir, "cache")

    def test_custom_env_prefix(self, monkeypatch, tmp_dir):
        monkeypatch.setenv("MYAPP_LLM__PROVIDER", "openai")
        config = Config(env_prefix="MYAPP_", data_dir=tmp_dir)
        assert config.get("llm.provider") == "openai"

    def test_yaml_config_file(self, tmp_dir):
        config_data = {"llm": {"provider": "gemini", "model": "gemini-pro"}}
        config_path = os.path.join(tmp_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = Config(config_file=config_path, data_dir=tmp_dir)
        assert config.get("llm.provider") == "gemini"
        assert config.get("llm.model") == "gemini-pro"

    def test_env_overrides_file(self, tmp_dir, monkeypatch):
        config_path = os.path.join(tmp_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump({"llm": {"provider": "gemini"}}, f)

        monkeypatch.setenv("ROSHNI_LLM__PROVIDER", "anthropic")
        config = Config(config_file=config_path, data_dir=tmp_dir)
        assert config.get("llm.provider") == "anthropic"

    def test_get_missing_key(self, tmp_dir):
        config = Config(data_dir=tmp_dir)
        assert config.get("nonexistent.key") is None
        assert config.get("nonexistent.key", "fallback") == "fallback"

    def test_set(self, tmp_dir):
        config = Config(data_dir=tmp_dir)
        config.set("custom.nested.value", 42)
        assert config.get("custom.nested.value") == 42

    def test_get_data_dir(self, tmp_dir):
        config = Config(data_dir=tmp_dir)
        assert config.get_data_dir() == tmp_dir

    def test_ensure_directories(self, tmp_dir):
        config = Config(data_dir=tmp_dir)
        config.ensure_directories()
        assert os.path.isdir(os.path.join(tmp_dir, "cache"))
        assert os.path.isdir(os.path.join(tmp_dir, "logs"))

    def test_extra_defaults(self, tmp_dir):
        config = Config(data_dir=tmp_dir, defaults={"custom": {"key": "value"}})
        assert config.get("custom.key") == "value"


class TestGetConfig:
    def test_singleton(self, tmp_dir):
        c1 = get_config(data_dir=tmp_dir)
        c2 = get_config()
        assert c1 is c2

    def test_reset_clears_singleton(self, tmp_dir):
        c1 = get_config(data_dir=tmp_dir)
        reset_config()
        c2 = get_config(data_dir=tmp_dir)
        assert c1 is not c2
