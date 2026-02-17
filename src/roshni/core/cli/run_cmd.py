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

    scheduler_enabled = config.get("scheduler.enabled", False)

    if scheduler_enabled:
        asyncio.run(_run_with_scheduler(agent, bot_token, allowed_ids, config))
    else:
        gateway = TelegramGateway(
            agent=agent,
            bot_token=bot_token,
            allowed_user_ids=allowed_ids,
        )
        bot_name = config.get("bot.name", "Roshni")
        click.echo(f"Starting {bot_name} on Telegram...")
        click.echo("Press Ctrl+C to stop.\n")
        asyncio.run(gateway.start())


async def _run_with_scheduler(agent, bot_token, allowed_ids, config) -> None:  # type: ignore[no-untyped-def]
    """Start EventGateway + Scheduler + TelegramGateway wired together."""
    from roshni.gateway.event_gateway import EventGateway
    from roshni.gateway.plugins.telegram.bot import TelegramGateway
    from roshni.gateway.scheduler import GatewayScheduler

    # Build the event gateway
    event_gw = EventGateway(agent=agent)
    event_gw.start()

    # Build the Telegram gateway with event routing
    telegram_gw = TelegramGateway(
        agent=agent,
        bot_token=bot_token,
        allowed_user_ids=allowed_ids,
        event_gateway=event_gw,
    )

    # Wire response handler: heartbeat/scheduled responses go to Telegram
    async def _send_response(event, response: str) -> None:  # type: ignore[no-untyped-def]
        if response and response.strip():
            await telegram_gw.send_proactive(response)

    event_gw.set_response_handler(_send_response)

    # Build and start scheduler
    scheduler = GatewayScheduler(submit_fn=event_gw.submit)
    scheduler.add_jobs_from_config(config)
    scheduler.start()

    bot_name = config.get("bot.name", "Roshni")
    click.echo(f"Starting {bot_name} on Telegram with scheduler...")
    click.echo("Press Ctrl+C to stop.\n")

    try:
        await telegram_gw.start()
    finally:
        scheduler.shutdown()
        await event_gw.stop()
