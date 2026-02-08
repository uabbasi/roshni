"""Roshni CLI — entry point for init, run, and chat commands."""

import click

from roshni import __version__


@click.group()
@click.version_option(version=__version__, package_name="roshni")
def main() -> None:
    """Roshni — your personal AI assistant."""


# Register subcommands (lazy imports keep startup fast)
from .chat_cmd import chat
from .init_cmd import init
from .run_cmd import run

main.add_command(init)
main.add_command(run)
main.add_command(chat)
