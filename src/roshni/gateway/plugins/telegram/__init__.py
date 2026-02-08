"""Telegram gateway plugin."""

try:
    from .bot import TelegramGateway

    __all__ = ["TelegramGateway"]
except ImportError:
    # python-telegram-bot not installed
    __all__ = []
