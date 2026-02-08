"""Telegram gateway — bot implementation using python-telegram-bot.

Implements BotGateway ABC with whitelist auth, typing indicator,
HTML formatting, and message splitting for long responses.
"""

from __future__ import annotations

import asyncio
import re
from html import escape as html_escape

from loguru import logger

from roshni.agent.base import BaseAgent
from roshni.gateway.base import BotGateway

# Telegram message length limit
MAX_MESSAGE_LENGTH = 4096


def _md_to_html(text: str) -> str:
    """Convert basic markdown to Telegram-safe HTML.

    Handles bold, italic, code blocks, inline code, headers, and lists.
    Uses HTML parse mode which is more forgiving than MarkdownV2.
    """
    lines = text.split("\n")
    result_lines: list[str] = []
    in_code_block = False
    code_block_lines: list[str] = []

    for line in lines:
        if line.strip().startswith("```"):
            if in_code_block:
                code_content = html_escape("\n".join(code_block_lines))
                result_lines.append(f"<pre>{code_content}</pre>")
                code_block_lines = []
                in_code_block = False
            else:
                in_code_block = True
            continue

        if in_code_block:
            code_block_lines.append(line)
            continue

        # Escape HTML entities
        line = html_escape(line)

        # Headers -> bold
        line = re.sub(r"^#{1,3}\s+(.+)$", r"<b>\1</b>", line)

        # Bold: **text**
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)

        # Italic: *text* (not inside words)
        line = re.sub(r"(?<!\w)\*([^*]+?)\*(?!\w)", r"<i>\1</i>", line)

        # Inline code: `text`
        line = re.sub(r"`([^`]+?)`", r"<code>\1</code>", line)

        # Bullet lists
        line = re.sub(r"^\s*[\-\*]\s+", "• ", line)

        result_lines.append(line)

    # Close any unclosed code block
    if in_code_block and code_block_lines:
        code_content = html_escape("\n".join(code_block_lines))
        result_lines.append(f"<pre>{code_content}</pre>")

    return "\n".join(result_lines).strip()


def _split_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split a message into chunks that fit Telegram's limit."""
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break

        # Try to split at a newline
        split_at = text.rfind("\n", 0, max_length)
        if split_at == -1:
            # No newline — split at space
            split_at = text.rfind(" ", 0, max_length)
        if split_at == -1:
            # No space — hard cut
            split_at = max_length

        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")

    return chunks


class TelegramGateway(BotGateway):
    """Telegram bot gateway with whitelist authentication.

    Uses python-telegram-bot to receive messages and route them
    through a BaseAgent.
    """

    def __init__(
        self,
        agent: BaseAgent,
        bot_token: str,
        allowed_user_ids: list[str | int] | None = None,
    ):
        self.agent = agent
        self.bot_token = bot_token
        self.allowed_user_ids: set[int] = {int(uid) for uid in (allowed_user_ids or [])}
        self._app = None

    def _is_authorized(self, user_id: int) -> bool:
        """Check if a user is in the allowlist."""
        if not self.allowed_user_ids:
            logger.warning("No Telegram allowlist configured — allowing all users")
            return True
        return user_id in self.allowed_user_ids

    async def start(self) -> None:
        """Start the Telegram bot polling loop."""
        try:
            from telegram import BotCommand, Update
            from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
        except ImportError:
            raise ImportError("Install Telegram support: pip install 'roshni[bot]'")

        agent = self.agent
        gateway = self

        async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            if not update.effective_user or not update.message:
                return
            user_id = update.effective_user.id
            name = agent.name or "Roshni"
            await update.message.reply_text(
                f"Hi! I'm {name}, your personal assistant.\n\n"
                f"Just send me a message to chat.\n"
                f"Use /help to see what I can do.\n\n"
                f"Your user ID: <code>{user_id}</code>",
                parse_mode="HTML",
            )

        async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            if not update.message:
                return
            await update.message.reply_text(
                "<b>Commands</b>\n"
                "• /help — Show this help\n"
                "• /clear — Start a fresh conversation\n\n"
                "Just type normally to chat!",
                parse_mode="HTML",
            )

        async def handle_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            if not update.message or not update.effective_user:
                return
            if not gateway._is_authorized(update.effective_user.id):
                return
            agent.clear_history()
            await update.message.reply_text("Conversation cleared. Fresh start!")

        async def keep_typing(chat_id: int, bot, stop_event: asyncio.Event) -> None:
            """Send typing indicator until stopped."""
            while not stop_event.is_set():
                try:
                    await bot.send_chat_action(chat_id=chat_id, action="typing")
                except Exception:
                    break
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=4.0)
                    break
                except TimeoutError:
                    continue

        async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            if not update.effective_user or not update.message or not update.message.text:
                return

            user_id = update.effective_user.id
            if not gateway._is_authorized(user_id):
                logger.warning(f"Unauthorized Telegram user: {user_id}")
                await update.message.reply_text("Not authorized.")
                return

            message_text = update.message.text.strip()
            if not message_text:
                return

            logger.info(f"Telegram [{user_id}]: {message_text[:80]}")

            # Typing indicator
            stop_typing = asyncio.Event()
            typing_task = asyncio.create_task(keep_typing(update.effective_chat.id, context.bot, stop_typing))

            try:
                response = await gateway.handle_message(message_text, str(user_id))

                stop_typing.set()
                await typing_task

                # Convert to HTML and split if needed
                html_response = _md_to_html(response)
                chunks = _split_message(html_response)

                for chunk in chunks:
                    try:
                        await update.message.reply_text(chunk, parse_mode="HTML")
                    except Exception:
                        # HTML parsing failed — send as plain text
                        await update.message.reply_text(chunk)

            except Exception as e:
                stop_typing.set()
                await typing_task
                logger.error(f"Telegram message handling failed: {e}")
                await update.message.reply_text(f"Something went wrong: {e}")

        # Build application
        app = Application.builder().token(self.bot_token).build()
        app.add_handler(CommandHandler("start", handle_start))
        app.add_handler(CommandHandler("help", handle_help))
        app.add_handler(CommandHandler("clear", handle_clear))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        # Set bot commands
        async def post_init(app: Application) -> None:
            commands = [
                BotCommand("help", "Show available commands"),
                BotCommand("clear", "Start a fresh conversation"),
            ]
            try:
                await app.bot.set_my_commands(commands)
            except Exception as e:
                logger.debug(f"Could not set bot commands: {e}")

        app.post_init = post_init
        self._app = app

        logger.info("Starting Telegram bot polling...")
        await app.initialize()
        await app.start()
        await app.updater.start_polling()

        # Keep running until stopped
        stop_event = asyncio.Event()

        def _signal_handler():
            stop_event.set()

        import signal

        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _signal_handler)
            except NotImplementedError:
                pass  # Windows

        await stop_event.wait()
        await self.stop()

    async def handle_message(self, message: str, user_id: str) -> str:
        """Route a message through the agent."""
        return await self.agent.invoke(message)

    async def stop(self) -> None:
        """Stop the bot gracefully."""
        if self._app:
            logger.info("Stopping Telegram bot...")
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
