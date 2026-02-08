"""Tests for roshni.core.secrets."""

import os

import pytest
import yaml

from roshni.core.exceptions import SecretNotFoundError
from roshni.core.secrets import (
    DotenvProvider,
    EnvProvider,
    JsonFileProvider,
    SecretsManager,
    YamlFileProvider,
)


class TestEnvProvider:
    def test_get(self, monkeypatch):
        monkeypatch.setenv("MYAPP_TRELLO__API_KEY", "abc123")
        provider = EnvProvider(prefix="MYAPP_")
        assert provider.get("trello.api_key") == "abc123"

    def test_get_missing(self):
        provider = EnvProvider(prefix="NONEXISTENT_")
        assert provider.get("foo.bar") is None

    def test_get_namespace(self, monkeypatch):
        monkeypatch.setenv("TEST_TRELLO__API_KEY", "key1")
        monkeypatch.setenv("TEST_TRELLO__TOKEN", "tok1")
        monkeypatch.setenv("TEST_OTHER__X", "y")
        provider = EnvProvider(prefix="TEST_")
        ns = provider.get_namespace("trello")
        assert ns == {"api_key": "key1", "token": "tok1"}


class TestYamlFileProvider:
    def test_get(self, tmp_dir):
        path = os.path.join(tmp_dir, "secrets.yaml")
        with open(path, "w") as f:
            yaml.dump({"gmail": {"app_password": "secret123"}}, f)

        provider = YamlFileProvider(path)
        assert provider.get("gmail.app_password") == "secret123"

    def test_get_missing_key(self, tmp_dir):
        path = os.path.join(tmp_dir, "secrets.yaml")
        with open(path, "w") as f:
            yaml.dump({"gmail": {"app_password": "x"}}, f)

        provider = YamlFileProvider(path)
        assert provider.get("gmail.nonexistent") is None

    def test_missing_file(self, tmp_dir):
        provider = YamlFileProvider(os.path.join(tmp_dir, "nope.yaml"))
        assert provider.get("any.key") is None

    def test_get_namespace(self, tmp_dir):
        path = os.path.join(tmp_dir, "secrets.yaml")
        with open(path, "w") as f:
            yaml.dump({"fitbit": {"access_token": "at", "refresh_token": "rt"}}, f)

        provider = YamlFileProvider(path)
        ns = provider.get_namespace("fitbit")
        assert ns == {"access_token": "at", "refresh_token": "rt"}

    def test_reload(self, tmp_dir):
        path = os.path.join(tmp_dir, "secrets.yaml")
        with open(path, "w") as f:
            yaml.dump({"key": {"val": "v1"}}, f)

        provider = YamlFileProvider(path)
        assert provider.get("key.val") == "v1"

        with open(path, "w") as f:
            yaml.dump({"key": {"val": "v2"}}, f)

        # Still cached
        assert provider.get("key.val") == "v1"

        provider.reload()
        assert provider.get("key.val") == "v2"


class TestJsonFileProvider:
    def test_get_namespaced(self, tmp_dir):
        import json

        path = os.path.join(tmp_dir, "fitbit.json")
        with open(path, "w") as f:
            json.dump({"access_token": "at123", "refresh_token": "rt456"}, f)

        provider = JsonFileProvider(path, namespace="fitbit")
        assert provider.get("fitbit.access_token") == "at123"
        assert provider.get("other.key") is None

    def test_get_without_namespace(self, tmp_dir):
        import json

        path = os.path.join(tmp_dir, "data.json")
        with open(path, "w") as f:
            json.dump({"key": "value"}, f)

        provider = JsonFileProvider(path)
        assert provider.get("key") == "value"


class TestDotenvProvider:
    def test_get(self, tmp_dir):
        path = os.path.join(tmp_dir, ".env")
        with open(path, "w") as f:
            f.write("API_KEY=abc\nTOKEN=xyz\n")

        provider = DotenvProvider(path, namespace="trello")
        assert provider.get("trello.api_key") == "abc"
        assert provider.get("trello.token") == "xyz"
        assert provider.get("other.key") is None


class TestSecretsManager:
    def test_chain_priority(self, tmp_dir, monkeypatch):
        """Env provider (first in chain) should take priority."""
        monkeypatch.setenv("TEST_GMAIL__PASSWORD", "from_env")

        yaml_path = os.path.join(tmp_dir, "secrets.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump({"gmail": {"password": "from_yaml"}}, f)

        manager = SecretsManager(
            providers=[
                EnvProvider(prefix="TEST_"),
                YamlFileProvider(yaml_path),
            ]
        )

        assert manager.get("gmail.password") == "from_env"

    def test_fallback(self, tmp_dir):
        yaml_path = os.path.join(tmp_dir, "secrets.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump({"gmail": {"password": "from_yaml"}}, f)

        manager = SecretsManager(
            providers=[
                EnvProvider(prefix="NONEXISTENT_"),
                YamlFileProvider(yaml_path),
            ]
        )

        assert manager.get("gmail.password") == "from_yaml"

    def test_default_value(self):
        manager = SecretsManager(providers=[EnvProvider(prefix="NOPE_")])
        assert manager.get("any.key") is None
        assert manager.get("any.key", "fallback") == "fallback"

    def test_require_raises(self):
        manager = SecretsManager(providers=[EnvProvider(prefix="NOPE_")])
        with pytest.raises(SecretNotFoundError, match=r"missing\.key"):
            manager.require("missing.key")

    def test_require_returns(self, monkeypatch):
        monkeypatch.setenv("T_API__KEY", "found")
        manager = SecretsManager(providers=[EnvProvider(prefix="T_")])
        assert manager.require("api.key") == "found"

    def test_get_namespace_merges(self, tmp_dir, monkeypatch):
        monkeypatch.setenv("T_FITBIT__CLIENT_ID", "env_id")

        yaml_path = os.path.join(tmp_dir, "secrets.yaml")
        with open(yaml_path, "w") as f:
            yaml.dump({"fitbit": {"client_id": "yaml_id", "client_secret": "yaml_secret"}}, f)

        manager = SecretsManager(
            providers=[
                EnvProvider(prefix="T_"),
                YamlFileProvider(yaml_path),
            ]
        )

        ns = manager.get_namespace("fitbit")
        # Env provider (higher priority) wins for client_id
        assert ns["client_id"] == "env_id"
        # YAML fills in client_secret
        assert ns["client_secret"] == "yaml_secret"

    def test_add_provider(self, monkeypatch):
        monkeypatch.setenv("A_X__Y", "found")
        manager = SecretsManager(providers=[])
        assert manager.get("x.y") is None

        manager.add_provider(EnvProvider(prefix="A_"))
        assert manager.get("x.y") == "found"
