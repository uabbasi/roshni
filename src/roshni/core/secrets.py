"""
Dependency-injected secrets management.

Secrets are resolved through a chain of providers. Each provider implements
the SecretProvider protocol. The SecretsManager checks providers in order,
returning the first non-None result.

Usage:
    from roshni.core.secrets import SecretsManager, EnvProvider, YamlFileProvider

    manager = SecretsManager(providers=[
        EnvProvider("WEEKLIES_"),
        YamlFileProvider("~/.weeklies-data/config/secrets.yaml"),
    ])

    api_key = manager.get("trello.api_key")
    all_trello = manager.get_namespace("trello")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml
from loguru import logger

# ---------------------------------------------------------------------------
# Provider protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SecretProvider(Protocol):
    """Interface for secret providers. Implement this to add new secret sources."""

    def get(self, key_path: str) -> str | None:
        """Return a secret value for the given dot-notation key, or None."""
        ...

    def get_namespace(self, namespace: str) -> dict[str, Any]:
        """Return all secrets under a namespace as a flat dict."""
        ...


# ---------------------------------------------------------------------------
# Built-in providers
# ---------------------------------------------------------------------------


class EnvProvider:
    """
    Read secrets from environment variables.

    Maps dot-notation keys to env vars:
        "trello.api_key" -> PREFIX_TRELLO__API_KEY
    """

    def __init__(self, prefix: str = "ROSHNI_"):
        self.prefix = prefix

    def get(self, key_path: str) -> str | None:
        env_key = self.prefix + key_path.replace(".", "__").upper()
        return os.environ.get(env_key)

    def get_namespace(self, namespace: str) -> dict[str, Any]:
        result = {}
        prefix = self.prefix + namespace.upper() + "__"
        for env_key, env_value in os.environ.items():
            if env_key.startswith(prefix):
                sub_key = env_key[len(prefix) :].lower()
                result[sub_key] = env_value
        return result


class YamlFileProvider:
    """
    Read secrets from a YAML file.

    Expected format:
        trello:
          api_key: "abc123"
          token: "xyz789"
        gmail:
          app_password: "secret"
    """

    def __init__(self, path: str | Path):
        self._path = Path(path).expanduser()
        self._data: dict | None = None

    def _load(self) -> dict:
        if self._data is None:
            if self._path.exists():
                try:
                    with open(self._path) as f:
                        self._data = yaml.safe_load(f) or {}
                except Exception as e:
                    logger.warning(f"Could not load secrets from {self._path}: {e}")
                    self._data = {}
            else:
                self._data = {}
        return self._data

    def get(self, key_path: str) -> str | None:
        data = self._load()
        parts = key_path.split(".")
        current: Any = data
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return str(current) if current is not None else None

    def get_namespace(self, namespace: str) -> dict[str, Any]:
        data = self._load()
        section = data.get(namespace, {})
        return dict(section) if isinstance(section, dict) else {}

    def reload(self) -> None:
        """Force reload from disk on next access."""
        self._data = None


class JsonFileProvider:
    """Read secrets from a JSON file (e.g. legacy fitbit.json token files)."""

    def __init__(self, path: str | Path, namespace: str | None = None):
        """
        Args:
            path: Path to the JSON file.
            namespace: If set, all keys are scoped under this namespace.
                       e.g. namespace="fitbit" means get("fitbit.access_token") works.
        """
        self._path = Path(path).expanduser()
        self._namespace = namespace
        self._data: dict | None = None

    def _load(self) -> dict:
        if self._data is None:
            if self._path.exists():
                try:
                    with open(self._path) as f:
                        self._data = json.load(f)
                except Exception as e:
                    logger.warning(f"Could not load secrets from {self._path}: {e}")
                    self._data = {}
            else:
                self._data = {}
        return self._data

    def get(self, key_path: str) -> str | None:
        data = self._load()
        parts = key_path.split(".")

        # If namespaced, strip the namespace prefix
        if self._namespace:
            if not parts or parts[0] != self._namespace:
                return None
            parts = parts[1:]

        current: Any = data
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return str(current) if current is not None else None

    def get_namespace(self, namespace: str) -> dict[str, Any]:
        if self._namespace and namespace != self._namespace:
            return {}
        data = self._load()
        return dict(data) if isinstance(data, dict) else {}

    def reload(self) -> None:
        """Force reload from disk on next access."""
        self._data = None


class DotenvProvider:
    """Read secrets from a .env file."""

    def __init__(self, path: str | Path, namespace: str | None = None):
        """
        Args:
            path: Path to the .env file.
            namespace: If set, keys are scoped under this namespace.
        """
        self._path = Path(path).expanduser()
        self._namespace = namespace
        self._data: dict | None = None

    def _load(self) -> dict:
        if self._data is None:
            self._data = {}
            if self._path.exists():
                try:
                    from dotenv import dotenv_values

                    self._data = {k.lower(): v for k, v in dotenv_values(self._path).items() if v is not None}
                except Exception as e:
                    logger.warning(f"Could not load .env from {self._path}: {e}")
        return self._data

    def get(self, key_path: str) -> str | None:
        data = self._load()
        parts = key_path.split(".")

        if self._namespace:
            if not parts or parts[0] != self._namespace:
                return None
            parts = parts[1:]

        if len(parts) == 1:
            return data.get(parts[0])
        return None

    def get_namespace(self, namespace: str) -> dict[str, Any]:
        if self._namespace and namespace != self._namespace:
            return {}
        return dict(self._load())


class GCPSecretManagerProvider:
    """
    Read secrets from GCP Secret Manager.

    Maps dot-notation keys to GCP secret names:
        "trello.api_key" -> trello-api-key
    """

    def __init__(self, project_id: str):
        self._project_id = project_id
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google.cloud import secretmanager

                self._client = secretmanager.SecretManagerServiceClient()
            except ImportError:
                raise ImportError("Install with: pip install roshni[google]")
        return self._client

    def get(self, key_path: str) -> str | None:
        try:
            client = self._get_client()
            gcp_name = key_path.replace(".", "-").replace("_", "-")
            name = f"projects/{self._project_id}/secrets/{gcp_name}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            logger.debug(f"GCP Secret Manager lookup failed for {key_path}: {e}")
            return None

    def get_namespace(self, namespace: str) -> dict[str, Any]:
        # GCP Secret Manager doesn't support listing by prefix efficiently
        return {}


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class SecretsManager:
    """
    Chain-of-responsibility secrets manager.

    Queries providers in order, returning the first non-None result.
    """

    def __init__(self, providers: list[SecretProvider] | None = None):
        self._providers: list[SecretProvider] = providers or [EnvProvider()]

    def add_provider(self, provider: SecretProvider) -> None:
        """Append a provider to the chain."""
        self._providers.append(provider)

    def get(self, key_path: str, default: str | None = None) -> str | None:
        """
        Get a secret by dot-notation path.

        Queries each provider in order; returns the first non-None result.
        """
        for provider in self._providers:
            value = provider.get(key_path)
            if value is not None:
                return value
        return default

    def get_namespace(self, namespace: str) -> dict[str, Any]:
        """
        Get all secrets under a namespace, merging from all providers.

        Later providers override earlier ones (env vars typically first = highest priority).
        """
        result: dict[str, Any] = {}
        # Iterate in reverse so higher-priority providers (earlier in list) win
        for provider in reversed(self._providers):
            result.update(provider.get_namespace(namespace))
        return result

    def require(self, key_path: str) -> str:
        """Get a secret, raising SecretNotFoundError if not found."""
        from roshni.core.exceptions import SecretNotFoundError

        value = self.get(key_path)
        if value is None:
            providers_desc = ", ".join(type(p).__name__ for p in self._providers)
            raise SecretNotFoundError(f"Secret '{key_path}' not found in providers: [{providers_desc}]")
        return value
