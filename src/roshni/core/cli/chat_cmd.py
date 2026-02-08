"""roshni chat — terminal chat for testing."""

from __future__ import annotations

import asyncio

import click


@click.command()
def chat() -> None:
    """Chat with your bot in the terminal."""
    from roshni.core.cli.common import create_agent, ensure_init, load_config, load_secrets

    ensure_init()

    config = load_config()
    secrets = load_secrets()
    agent = create_agent(config, secrets)

    from roshni.gateway.cli_gateway import CliGateway

    gateway = CliGateway(agent)

    bot_name = config.get("bot.name", "Roshni")
    click.echo(f"Starting chat with {bot_name}...\n")

    asyncio.run(gateway.start())
