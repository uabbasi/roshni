"""Tool framework — defines tools that agents can call."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger

from roshni.core.config import Config
from roshni.core.secrets import SecretsManager


@dataclass
class ToolDefinition:
    """A tool that an agent can invoke via function calling."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    function: Callable[..., str]
    permission: str = "read"  # read, write, send, admin
    requires_approval: bool | None = None

    def to_litellm_schema(self) -> dict[str, Any]:
        """Convert to litellm/OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def needs_approval(self) -> bool:
        """Whether this tool must be user-approved before execution."""
        if self.requires_approval is not None:
            return self.requires_approval
        return self.permission in {"write", "send", "admin"}

    def execute(self, arguments: dict[str, Any] | str) -> str:
        """Execute the tool with the given arguments."""
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return f"Error: could not parse arguments: {arguments}"
        try:
            return self.function(**arguments)
        except Exception as e:
            logger.error(f"Tool {self.name} failed: {e}")
            return f"Error executing {self.name}: {e}"


def create_tools(config: Config, secrets: SecretsManager) -> list[ToolDefinition]:
    """Build the tool list based on enabled integrations in config."""
    tools: list[ToolDefinition] = []
    integrations = config.get("integrations", {}) or {}

    # Notes tool — always available
    from .notes_tool import create_notes_tools

    notes_dir = config.get("paths.notes_dir") or str(config.get("paths.data_dir", "~/.roshni") + "/notes")
    tools.extend(create_notes_tools(notes_dir))

    # Built-in tools (weather + web search/fetch) — enabled by default
    builtins_cfg = integrations.get("builtins", {}) or {}
    if builtins_cfg.get("enabled", True):
        try:
            from .builtin_tool import create_builtin_tools

            tools.extend(create_builtin_tools())
        except Exception as e:
            logger.warning(f"Could not load builtin tools: {e}")

    # Delighter tools — enabled by default
    delighters_cfg = integrations.get("delighters", {}) or {}
    if delighters_cfg.get("enabled", True):
        try:
            from .delighter_tool import create_delighter_tools

            reminders_path = config.get(
                "paths.reminders_path",
                str(config.get("paths.data_dir", "~/.roshni") + "/reminders.json"),
            )
            tools.extend(create_delighter_tools(reminders_path))
        except Exception as e:
            logger.warning(f"Could not load delighter tools: {e}")

    # Gmail tool — when gmail is enabled
    gmail_cfg = integrations.get("gmail", {}) or {}
    if gmail_cfg.get("enabled"):
        try:
            from .gmail_tool import create_gmail_tools

            tools.extend(create_gmail_tools(config, secrets))
        except Exception as e:
            logger.warning(f"Could not load Gmail tools: {e}")

    # Obsidian tool — when obsidian is enabled
    obsidian_cfg = integrations.get("obsidian", {}) or {}
    if obsidian_cfg.get("enabled"):
        vault_path = obsidian_cfg.get("vault_path", "")
        if vault_path:
            try:
                from .obsidian_tool import create_obsidian_tools

                tools.extend(create_obsidian_tools(vault_path))
            except Exception as e:
                logger.warning(f"Could not load Obsidian tools: {e}")

    # Trello tool — when trello is enabled
    trello_cfg = integrations.get("trello", {}) or {}
    if trello_cfg.get("enabled"):
        try:
            from .trello_tool import create_trello_tools

            tools.extend(create_trello_tools(config, secrets))
        except Exception as e:
            logger.warning(f"Could not load Trello tools: {e}")

    # Notion tool — when notion is enabled
    notion_cfg = integrations.get("notion", {}) or {}
    if notion_cfg.get("enabled"):
        try:
            from .notion_tool import create_notion_tools

            tools.extend(create_notion_tools(config, secrets))
        except Exception as e:
            logger.warning(f"Could not load Notion tools: {e}")

    # HealthKit (Apple Health export) — when healthkit is enabled
    healthkit_cfg = integrations.get("healthkit", {}) or {}
    if healthkit_cfg.get("enabled"):
        export_path = healthkit_cfg.get("export_path", "")
        if export_path:
            try:
                from .health_tool import create_health_tools

                tools.extend(create_health_tools(export_path))
            except Exception as e:
                logger.warning(f"Could not load HealthKit tools: {e}")

    return tools


__all__ = ["ToolDefinition", "create_tools"]
