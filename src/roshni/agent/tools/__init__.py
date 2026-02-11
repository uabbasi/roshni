"""Tool framework — defines tools that agents can call."""

from __future__ import annotations

import json
import time as _time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from loguru import logger

from roshni.core.config import Config
from roshni.core.secrets import SecretsManager

_TRANSIENT_ERRORS = (ConnectionError, TimeoutError, OSError)


@dataclass
class ToolDefinition:
    """A tool that an agent can invoke via function calling."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    function: Callable[..., str]
    permission: str = "read"  # read, write, send, admin
    requires_approval: bool | None = None

    @classmethod
    def from_function(
        cls,
        func: Callable[..., str],
        name: str,
        description: str,
        args_schema: Any,
        permission: str = "read",
        requires_approval: bool | None = None,
    ) -> ToolDefinition:
        """Create a ToolDefinition from a function and a schema class.

        The *args_schema* must have a ``model_json_schema()`` method (e.g. a
        Pydantic BaseModel subclass). This avoids importing Pydantic directly.
        """
        schema = args_schema.model_json_schema()
        schema.pop("title", None)
        schema.pop("$defs", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return cls(
            name=name,
            description=description,
            parameters=schema,
            function=func,
            permission=permission,
            requires_approval=requires_approval,
        )

    def to_litellm_schema(self) -> dict[str, Any]:
        """Convert to litellm/OpenAI function calling format."""
        func: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
        }
        params = self.parameters
        # Omit parameters entirely for zero-arg tools — Gemini and OpenAI
        # strict mode reject {"type": "object", "properties": {}}.
        if params.get("type") == "object" and not params.get("properties"):
            pass  # no parameters key
        else:
            if params.get("type") == "object" and "required" not in params:
                params = {**params, "required": []}
            func["parameters"] = params
        return {"type": "function", "function": func}

    def needs_approval(self) -> bool:
        """Whether this tool must be user-approved before execution."""
        if self.requires_approval is not None:
            return self.requires_approval
        return self.permission in {"write", "send", "admin"}

    def execute(self, arguments: dict[str, Any] | str, *, max_attempts: int = 3) -> str:
        """Execute the tool, retrying transient errors with exponential backoff."""
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return f"Error: could not parse arguments: {arguments}"
        assert isinstance(arguments, dict)
        for attempt in range(1, max_attempts + 1):
            try:
                return self.function(**arguments)
            except _TRANSIENT_ERRORS as e:
                if attempt < max_attempts:
                    _time.sleep(2 ** (attempt - 1))
                    logger.warning(f"Tool {self.name} transient error (attempt {attempt}/{max_attempts}): {e}")
                    continue
                logger.error(f"Tool {self.name} failed after {max_attempts} attempts: {e}")
                return f"Error: {self.name} failed after {max_attempts} attempts: {e}"
            except Exception as e:
                logger.error(f"Tool {self.name} failed: {e}")
                return f"Error executing {self.name}: {e}"
        return f"Error: {self.name} failed after {max_attempts} attempts"


def create_tools(config: Config, secrets: SecretsManager) -> list[ToolDefinition]:
    """Build the tool list based on enabled integrations in config."""
    from roshni.agent.permissions import get_domain_tier

    tools: list[ToolDefinition] = []
    integrations = config.get("integrations", {}) or {}
    permissions_cfg = config.get("permissions", {}) or {}

    # Vault tools (task + vault sections) — always-on when vault is configured
    vault_cfg = config.get("vault", {}) or {}
    vault_path = vault_cfg.get("path", "")
    tasks_dir = ""
    if vault_path:
        try:
            from roshni.agent.vault import VaultManager

            from .task_tool import create_task_tools
            from .vault_tools import create_vault_tools

            vault = VaultManager(vault_path, vault_cfg.get("agent_dir", "jarvis"))
            tasks_dir = str(vault.tasks_dir)

            vault_tier = get_domain_tier(permissions_cfg, "vault")
            tools.extend(create_vault_tools(vault, vault_tier))

            task_tier = get_domain_tier(permissions_cfg, "tasks")
            tools.extend(create_task_tools(tasks_dir, task_tier, projects_dir=str(vault.projects_dir)))
        except Exception as e:
            logger.warning(f"Could not load vault/task tools: {e}")

    # Notes tool — only when vault is not configured (superseded by vault tools)
    if not vault_path:
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
            tools.extend(create_delighter_tools(reminders_path, tasks_dir=tasks_dir))
        except Exception as e:
            logger.warning(f"Could not load delighter tools: {e}")

    # Gmail tool — when gmail is enabled
    gmail_cfg = integrations.get("gmail", {}) or {}
    if gmail_cfg.get("enabled"):
        try:
            from .gmail_tool import create_gmail_tools

            gmail_tier = get_domain_tier(permissions_cfg, "gmail")
            tools.extend(create_gmail_tools(config, secrets, tier=gmail_tier))
        except Exception as e:
            logger.warning(f"Could not load Gmail tools: {e}")

    # Obsidian tool — when obsidian is enabled
    obsidian_cfg = integrations.get("obsidian", {}) or {}
    if obsidian_cfg.get("enabled"):
        obs_vault_path = obsidian_cfg.get("vault_path", "")
        if obs_vault_path:
            try:
                from .obsidian_tool import create_obsidian_tools

                obsidian_tier = get_domain_tier(permissions_cfg, "obsidian")
                tools.extend(create_obsidian_tools(obs_vault_path, tier=obsidian_tier))
            except Exception as e:
                logger.warning(f"Could not load Obsidian tools: {e}")

    # Trello tool — when trello is enabled
    trello_cfg = integrations.get("trello", {}) or {}
    if trello_cfg.get("enabled"):
        try:
            from .trello_tool import create_trello_tools

            trello_tier = get_domain_tier(permissions_cfg, "trello")
            tools.extend(create_trello_tools(config, secrets, tier=trello_tier))
        except Exception as e:
            logger.warning(f"Could not load Trello tools: {e}")

    # Notion tool — when notion is enabled
    notion_cfg = integrations.get("notion", {}) or {}
    if notion_cfg.get("enabled"):
        try:
            from .notion_tool import create_notion_tools

            notion_tier = get_domain_tier(permissions_cfg, "notion")
            tools.extend(create_notion_tools(config, secrets, tier=notion_tier))
        except Exception as e:
            logger.warning(f"Could not load Notion tools: {e}")

    # HealthKit (Apple Health export) — when healthkit is enabled
    healthkit_cfg = integrations.get("healthkit", {}) or {}
    if healthkit_cfg.get("enabled"):
        export_path = healthkit_cfg.get("export_path", "")
        if export_path:
            try:
                from .health_tool import create_health_tools

                health_tier = get_domain_tier(permissions_cfg, "health")
                tools.extend(create_health_tools(export_path, tier=health_tier))
            except Exception as e:
                logger.warning(f"Could not load HealthKit tools: {e}")

    return tools


__all__ = ["ToolDefinition", "create_tools"]
