"""Pydantic models for config validation.

Opt-in schema validation for ``Config.config_data``.  Call
``Config.validated()`` to obtain a typed, validated ``RoshniConfig``
instance.  Existing dict-based access continues to work unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class PathsConfig(BaseModel):
    """File-system paths used by the application."""

    data_dir: Path
    cache_dir: Path | None = None
    log_dir: Path | None = None
    persona_dir: Path | None = None

    @field_validator("data_dir", "cache_dir", "log_dir", "persona_dir", mode="before")
    @classmethod
    def _expand_user(cls, v: Any) -> Any:
        if isinstance(v, str):
            return Path(v).expanduser()
        if isinstance(v, Path):
            return v.expanduser()
        return v


class LLMProviderConfig(BaseModel):
    """Settings for a single LLM provider."""

    model: str = ""
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout: int = 60


class LLMSelectorConfig(BaseModel):
    """Selector tuning knobs for model routing heuristics."""

    tool_result_chars_threshold: int = 500
    complex_query_chars_threshold: int = 150
    mode_overrides: dict[str, str] = {}


class LLMConfig(BaseModel):
    """Top-level LLM configuration."""

    model_config = ConfigDict(extra="allow")

    default: str = "anthropic"
    fallback: str | None = None
    providers: dict[str, LLMProviderConfig] = {}
    mode_overrides: dict[str, str] = {}
    selector: LLMSelectorConfig = LLMSelectorConfig()
    daily_budget: float | None = None

    @model_validator(mode="after")
    def _default_in_providers(self) -> LLMConfig:
        if self.default and self.providers and self.default not in self.providers:
            raise ValueError(f"default provider {self.default!r} not found in providers: {sorted(self.providers)}")
        return self


class BotConfig(BaseModel):
    """Telegram / messaging bot settings."""

    name: str = "roshni"
    token: str = ""


class SecurityConfig(BaseModel):
    """Security-related toggles."""

    require_write_approval: bool = True
    persist_approval_grants: bool = True
    auto_approve_channels: list[str] = []
    approval_grants_path: str | None = None


class IntegrationToggle(BaseModel):
    """On/off switch for a single integration, with extra fields allowed."""

    model_config = ConfigDict(extra="allow")

    enabled: bool = True


class IntegrationsConfig(BaseModel):
    """Known integration toggles."""

    gmail: IntegrationToggle = IntegrationToggle()
    obsidian: IntegrationToggle = IntegrationToggle()
    trello: IntegrationToggle = IntegrationToggle()
    notion: IntegrationToggle = IntegrationToggle()
    healthkit: IntegrationToggle = IntegrationToggle()
    builtins: IntegrationToggle = IntegrationToggle()
    delighters: IntegrationToggle = IntegrationToggle()


class VaultConfig(BaseModel):
    """Vault / persona directory config."""

    path: str | None = None
    agent_dir: str | None = None


class RoshniConfig(BaseModel):
    """Root configuration model.

    Uses ``extra="allow"`` so consumers can bolt on custom sections
    without touching this schema.
    """

    model_config = ConfigDict(extra="allow")

    paths: PathsConfig = PathsConfig(data_dir=Path("~/.roshni-data"))
    llm: LLMConfig = LLMConfig()
    bot: BotConfig = BotConfig()
    security: SecurityConfig = SecurityConfig()
    integrations: IntegrationsConfig = IntegrationsConfig()
    vault: VaultConfig = VaultConfig()
