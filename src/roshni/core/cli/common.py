"""Shared setup logic for CLI commands."""

from __future__ import annotations

import sys
from pathlib import Path

import click

ROSHNI_DIR = Path.home() / ".roshni"
CONFIG_PATH = ROSHNI_DIR / "config.yaml"
SECRETS_PATH = ROSHNI_DIR / "secrets.yaml"


def ensure_init() -> None:
    """Check that roshni init has been run."""
    if not CONFIG_PATH.exists():
        click.echo("No configuration found. Run 'roshni init' first.")
        sys.exit(1)


def load_config():
    """Load config from ~/.roshni/config.yaml."""
    from roshni.core.config import Config

    return Config(config_file=str(CONFIG_PATH), data_dir=str(ROSHNI_DIR))


def load_secrets():
    """Load secrets from ~/.roshni/secrets.yaml."""
    from roshni.core.secrets import SecretsManager, YamlFileProvider

    return SecretsManager(providers=[YamlFileProvider(SECRETS_PATH)])


def set_api_key_env(config, secrets) -> None:
    """Set API keys as env vars so litellm can find them.

    Supports both new multi-provider format (llm.api_keys.<provider>)
    and legacy single-provider format (llm.api_key).
    """
    import os

    from roshni.core.llm.config import PROVIDER_ENV_MAP

    # New format: llm.api_keys.<provider> — set all configured providers
    api_keys: dict = secrets.get("llm.api_keys", {}) or {}
    if api_keys:
        for provider, key in api_keys.items():
            env_var = PROVIDER_ENV_MAP.get(provider)
            if env_var and key and env_var not in os.environ:
                os.environ[env_var] = key
        return

    # Legacy format: single llm.api_key → infer provider from config
    api_key = secrets.get("llm.api_key", "")
    if not api_key:
        return

    provider = config.get("llm.default", "") or config.get("llm.provider", "openai")
    env_var = PROVIDER_ENV_MAP.get(provider)
    if env_var and env_var not in os.environ:
        os.environ[env_var] = api_key


def create_agent(config, secrets):
    """Create a DefaultAgent with tools based on config."""
    from roshni.agent.default import DefaultAgent
    from roshni.agent.tools import create_tools

    set_api_key_env(config, secrets)

    # Resolve paths from vault when configured, else fall back to legacy paths
    vault_cfg = config.get("vault", {}) or {}
    vault_path = vault_cfg.get("path", "")
    if vault_path:
        from roshni.agent.vault import VaultManager

        vault = VaultManager(vault_path, vault_cfg.get("agent_dir", "jarvis"))
        persona_dir = str(vault.persona_dir)
        memory_path = str(vault.memory_dir / "MEMORY.md")
    else:
        persona_dir = config.get("paths.persona_dir", str(ROSHNI_DIR / "persona"))
        memory_path = None

    tools = create_tools(config, secrets)

    kwargs: dict = dict(
        config=config,
        secrets=secrets,
        tools=tools,
        persona_dir=persona_dir,
        name=config.get("bot.name", "Roshni"),
    )
    if memory_path:
        kwargs["memory_path"] = memory_path

    return DefaultAgent(**kwargs)
