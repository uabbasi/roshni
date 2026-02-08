"""CLI gateway — terminal chat interface with rich formatting."""

from __future__ import annotations

from roshni.agent.base import BaseAgent
from roshni.gateway.base import BotGateway


class CliGateway(BotGateway):
    """Terminal-based chat gateway using rich for formatting.

    Runs an input loop, sends messages to the agent, and renders
    responses with markdown formatting.
    """

    def __init__(self, agent: BaseAgent, *, user_id: str = "cli_user"):
        self.agent = agent
        self.user_id = user_id
        self._running = False

    async def start(self) -> None:
        """Start the interactive terminal chat loop."""
        try:
            from rich.console import Console
            from rich.markdown import Markdown
            from rich.panel import Panel
        except ImportError:
            raise ImportError("Install rich: pip install rich")

        console = Console()
        self._running = True

        console.print(Panel("Type a message to chat. Commands: /help, /clear, /exit", title="Roshni Chat"))
        console.print()

        while self._running:
            try:
                user_input = console.input("[bold cyan]You:[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "/exit":
                console.print("Goodbye!")
                break
            elif user_input.lower() == "/clear":
                self.agent.clear_history()
                console.print("[dim]Conversation cleared.[/dim]\n")
                continue
            elif user_input.lower() == "/help":
                console.print(
                    Panel(
                        "Just type naturally to chat.\n\n"
                        "Commands:\n"
                        "  /clear — Start a fresh conversation\n"
                        "  /exit  — Quit\n"
                        "  /help  — Show this help",
                        title="Help",
                    )
                )
                console.print()
                continue

            # Get response from agent
            with console.status("[bold green]Thinking..."):
                response = await self.handle_message(user_input, self.user_id)

            console.print()
            console.print(Markdown(response))
            console.print()

    async def handle_message(self, message: str, user_id: str) -> str:
        """Send message to agent and return response."""
        return await self.agent.invoke(message)

    async def stop(self) -> None:
        """Stop the chat loop."""
        self._running = False
