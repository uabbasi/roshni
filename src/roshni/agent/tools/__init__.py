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

    # Notes tool — always available
    from .notes_tool import create_notes_tools

    notes_dir = config.get("paths.notes_dir") or str(config.get("paths.data_dir", "~/.roshni") + "/notes")
    tools.extend(create_notes_tools(notes_dir))

    # Gmail tool — when gmail is enabled
    integrations = config.get("integrations", {}) or {}
    gmail_cfg = integrations.get("gmail", {}) or {}
    if gmail_cfg.get("enabled"):
        try:
            from .gmail_tool import create_gmail_tools

            tools.extend(create_gmail_tools(secrets))
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

    return tools


__all__ = ["ToolDefinition", "create_tools"]
