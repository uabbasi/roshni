"""roshni run — start the Telegram bot."""

from __future__ import annotations

import asyncio

import click


@click.command()
def run() -> None:
    """Start your bot on Telegram."""
    from roshni.core.cli.common import create_agent, ensure_init, load_config, load_secrets

    ensure_init()

    config = load_config()
    secrets = load_secrets()

    # Check platform
    platform = config.get("platform", "telegram")
    if platform != "telegram":
        click.echo("Platform is set to 'terminal-only'. Use 'roshni chat' instead.")
        return

    # Get Telegram credentials
    bot_token = secrets.get("telegram.bot_token", "")
    if not bot_token:
        click.echo("No Telegram bot token found. Run 'roshni init' to set it up.")
        return

    allowed_ids = config.get("telegram.allowed_user_ids", []) or []
    if not allowed_ids:
        click.echo("No Telegram allowed user IDs configured. Run 'roshni init' and add your Telegram user ID.")
        return

    # Create agent and gateway
    agent = create_agent(config, secrets)

    try:
        from roshni.gateway.plugins.telegram.bot import TelegramGateway
    except ImportError:
        click.echo("Telegram support not installed. Run: pip install 'roshni[bot]'")
        return

    gateway = TelegramGateway(
        agent=agent,
        bot_token=bot_token,
        allowed_user_ids=allowed_ids,
    )

    bot_name = config.get("bot.name", "Roshni")
    click.echo(f"Starting {bot_name} on Telegram...")
    click.echo("Press Ctrl+C to stop.\n")

    asyncio.run(gateway.start())
