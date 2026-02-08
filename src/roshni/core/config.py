"""
Hierarchical configuration management.

Loads configuration from multiple sources with this precedence (highest wins):
    1. Environment variables (PREFIX_SECTION__KEY)
    2. Config file (YAML or JSON)
    3. Built-in defaults

Consumers customize behavior via constructor args — no hardcoded app-specific paths.

Usage:
    # Roshni defaults
    config = Config()

    # Consumer override
    config = Config(
        config_file="config/config.yaml",
        env_prefix="WEEKLIES_",
        data_dir="~/.weeklies-data",
    )

    config.get("llm.provider")       # dot-notation access
    config.get("paths.data_dir")     # returns resolved path
"""

import json
import os
from typing import Any

import yaml

_DEFAULT_ENV_PREFIX = "ROSHNI_"
_DEFAULT_DATA_DIR_NAME = ".roshni-data"


class Config:
    """
    Central configuration manager.

    Loads and merges configuration from defaults, a config file, and
    environment variables. Env vars use double-underscore to denote nesting:
    ROSHNI_LLM__PROVIDER=anthropic -> config["llm"]["provider"] = "anthropic"
    """

    def __init__(
        self,
        config_file: str | None = None,
        env_prefix: str = _DEFAULT_ENV_PREFIX,
        data_dir: str | None = None,
        defaults: dict[str, Any] | None = None,
    ):
        """
        Args:
            config_file: Path to YAML or JSON configuration file.
            env_prefix: Prefix for environment variable overrides.
            data_dir: Base directory for data storage. Defaults to ~/.roshni-data.
            defaults: Additional default values to merge (consumer-specific).
        """
        self.config_file = config_file
        self.env_prefix = env_prefix or ""
        self._data_dir = data_dir or os.path.join("~", _DEFAULT_DATA_DIR_NAME)
        self._extra_defaults = defaults or {}
        self.config_data: dict[str, Any] = {}

        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from all sources."""
        self.config_data = self._get_default_config()

        # Merge consumer-provided defaults
        if self._extra_defaults:
            self._update_dict(self.config_data, self._extra_defaults)

        # Load from config file
        if self.config_file and os.path.exists(self.config_file):
            file_config = self._load_file(self.config_file)
            self._update_dict(self.config_data, file_config)

        # Env vars override everything
        self._load_from_env()

    def _get_default_config(self) -> dict[str, Any]:
        """Build default configuration. No app-specific paths — just structure."""
        data_dir = os.path.expanduser(self._data_dir)
        return {
            "paths": {
                "data_dir": data_dir,
                "cache_dir": os.path.join(data_dir, "cache"),
                "log_dir": os.path.join(data_dir, "logs"),
            },
            "llm": {
                "provider": "anthropic",
                "model": "",
                "api_key": "",
            },
        }

    @staticmethod
    def _load_file(path: str) -> dict[str, Any]:
        """Load a YAML or JSON config file."""
        ext = os.path.splitext(path)[1].lower()
        with open(path) as f:
            if ext in (".yaml", ".yml"):
                return yaml.safe_load(f) or {}
            elif ext == ".json":
                return json.load(f)
        return {}

    def _update_dict(self, target: dict, source: dict) -> None:
        """Recursively merge source into target."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_dict(target[key], value)
            else:
                target[key] = value

    def _load_from_env(self) -> None:
        """Override config values from environment variables."""
        if not self.env_prefix:
            return
        for env_key, env_value in os.environ.items():
            if not env_key.startswith(self.env_prefix):
                continue
            config_key = env_key[len(self.env_prefix) :].lower()
            key_parts = config_key.split("__")

            current = self.config_data
            for part in key_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[key_parts[-1]] = env_value

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a config value by dot-notation path.

        Args:
            key_path: e.g. "paths.data_dir", "llm.provider"
            default: Returned when key is not found.
        """
        parts = key_path.split(".")
        current = self.config_data
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def set(self, key_path: str, value: Any) -> None:
        """Set a config value by dot-notation path, creating intermediate dicts."""
        parts = key_path.split(".")
        current = self.config_data
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def get_data_dir(self) -> str:
        """Return the resolved data directory path."""
        return os.path.expanduser(self.get("paths.data_dir", self._data_dir))

    def ensure_directories(self) -> None:
        """Create all configured directories if they don't exist."""
        for path_value in self.config_data.get("paths", {}).values():
            if isinstance(path_value, str):
                expanded = os.path.expanduser(path_value)
                os.makedirs(expanded, exist_ok=True)


# Module-level singleton
_config_instance: Config | None = None


def get_config(
    config_file: str | None = None,
    env_prefix: str = _DEFAULT_ENV_PREFIX,
    data_dir: str | None = None,
) -> Config:
    """Get or create the global Config singleton."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_file=config_file, env_prefix=env_prefix, data_dir=data_dir)
    return _config_instance


def reset_config() -> None:
    """Reset the global Config singleton (useful for testing)."""
    global _config_instance
    _config_instance = None
