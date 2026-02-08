"""Gateway framework — abstract messaging platform integration."""

from .base import BotGateway
from .cli_gateway import CliGateway

__all__ = ["BotGateway", "CliGateway"]
