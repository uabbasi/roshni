"""Base gateway — abstract interface for messaging platform integrations.

Defines the contract for bots that receive messages from external platforms
(Telegram, Slack, Discord, CLI, etc.) and route them to an agent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BotGateway(ABC):
    """Abstract base for messaging platform integrations.

    Subclasses implement platform-specific message handling.
    The gateway owns the connection lifecycle and delegates
    message processing to an agent (via a router or directly).

    Usage::

        class TelegramGateway(BotGateway):
            async def start(self):
                # Connect to Telegram API
                ...

            async def handle_message(self, message, user_id):
                # Route through agent and return response
                return await self.router.route(message)

            async def stop(self):
                # Disconnect
                ...

        gateway = TelegramGateway()
        await gateway.start()
    """

    @abstractmethod
    async def start(self) -> None:
        """Start the gateway (connect to platform, begin listening)."""

    @abstractmethod
    async def handle_message(self, message: str, user_id: str) -> str:
        """Handle an incoming message and return a response.

        Args:
            message: User message text.
            user_id: Platform-specific user identifier.

        Returns:
            Response text to send back to the user.
        """

    async def stop(self) -> None:  # noqa: B027
        """Stop the gateway (disconnect, cleanup). Optional override."""
