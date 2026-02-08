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
    """Set the provider's API key as an env var so litellm can find it."""
    import os

    provider = config.get("llm.provider", "openai")
    api_key = secrets.get("llm.api_key", "")
    if not api_key:
        return

    env_var_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    env_var = env_var_map.get(provider)
    if env_var and env_var not in os.environ:
        os.environ[env_var] = api_key


def create_agent(config, secrets):
    """Create a DefaultAgent with tools based on config."""
    from roshni.agent.default import DefaultAgent
    from roshni.agent.tools import create_tools

    set_api_key_env(config, secrets)

    persona_dir = config.get("paths.persona_dir", str(ROSHNI_DIR / "persona"))
    tools = create_tools(config, secrets)

    return DefaultAgent(
        config=config,
        secrets=secrets,
        tools=tools,
        persona_dir=persona_dir,
        name=config.get("bot.name", "Roshni"),
    )
