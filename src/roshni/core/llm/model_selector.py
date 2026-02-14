"""
Model selection for light, heavy, and thinking tasks.

Provides a programmatic API for choosing between fast/cheap models,
slower/smarter ones, and reasoning models based on task complexity.
Settings are persisted to disk so they survive restarts.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime

from loguru import logger

from .config import (
    MODEL_CATALOG,
    THINKING_BUDGET_MAP,
    ModelConfig,
    ThinkingLevel,
    get_available_families,
    get_family_models,
)
from .token_budget import get_budget_pressure


@dataclass
class TaskSignals:
    """Runtime signals for dynamic model selection.

    Lightweight data bag passed into ``ModelSelector.select()`` to let the
    selector upgrade/downgrade the tier based on what has happened *so far*
    in the current tool loop — without coupling to specific tool names.
    """

    iteration: int = 0
    tool_result_chars: int = 0  # total chars of tool results so far
    needs_synthesis: bool = False
    needs_escalation: bool = False  # light model refused/hedged → retry on heavy
    channel: str | None = None


# Keywords that suggest a query needs a heavier model.
_COMPLEX_KEYWORDS: set[str] = {
    "analyze",
    "compare",
    "explain",
    "plan",
    "design",
    "refactor",
    "review",
    "debug",
    "evaluate",
    "research",
    "strategy",
    "architect",
    "optimize",
    "trade-off",
    "tradeoff",
    "pros and cons",
}

# Keywords that suggest a light model is sufficient.
_LIGHT_KEYWORDS: list[str] = ["summary", "summarize", "list", "quick", "simple", "brief"]

# Query modes that map to light models.
_LIGHT_MODES: set[str] = {"summary", "answer", "timeline"}


class ModelSelector:
    """Manages model selection for light, heavy, and thinking tasks."""

    def __init__(
        self,
        light_model: ModelConfig | None = None,
        heavy_model: ModelConfig | None = None,
        thinking_model: ModelConfig | None = None,
        settings_path: str = "~/.roshni-data/model_settings.json",
        quiet_hours: tuple[int, int] | None = None,
        quiet_model: ModelConfig | None = None,
        mode_overrides: dict[str, ModelConfig] | None = None,
        tool_result_chars_threshold: int = 500,
        complex_query_chars_threshold: int = 150,
    ):
        self._settings_path = os.path.expanduser(settings_path)
        self._quiet_hours = quiet_hours
        self._quiet_model = quiet_model
        self._mode_overrides: dict[str, ModelConfig] = dict(mode_overrides) if mode_overrides else {}
        self._tool_result_chars_threshold = max(0, int(tool_result_chars_threshold))
        self._complex_query_chars_threshold = max(0, int(complex_query_chars_threshold))

        saved_light, saved_heavy, saved_thinking, saved_family = self._load_saved_settings()
        self.light_model = light_model or saved_light or self._default_light()
        self.heavy_model = heavy_model or saved_heavy or self._default_heavy()
        self.thinking_model = thinking_model or saved_thinking or self._default_thinking()
        self._active_family: str | None = saved_family or self._infer_family()

    @staticmethod
    def _default_light() -> ModelConfig:
        return MODEL_CATALOG["gemini"][0]  # Gemini 3 Flash

    @staticmethod
    def _default_heavy() -> ModelConfig:
        return MODEL_CATALOG["gemini"][1]  # Gemini 3 Pro

    @staticmethod
    def _default_thinking() -> ModelConfig:
        return MODEL_CATALOG["gemini"][2]  # Gemini 2.5 Pro (Thinking)

    def select(
        self,
        query: str,
        *,
        mode: str | None = None,
        heavy_modes: set[str] | None = None,
        think: bool = False,
        thinking_level: ThinkingLevel = ThinkingLevel.OFF,
        signals: TaskSignals | None = None,
    ) -> ModelConfig:
        """Single entry point for model selection.

        Priority:
        0. Quiet hours -> quiet model (cheapest, overnight)
        0b. Budget pressure >= 95% -> light model (near exhaustion)
        0c. Budget pressure >= 80% -> light model (moderate pressure)
        0d. Consumer-registered mode override -> exact model
        1. think=True or thinking_level > OFF or mode=="think" -> thinking model
        2. Signal: lightweight channels (boot, heartbeat) -> light model
        3. Signal: substantial tool results or synthesis -> heavy model
        4. mode in heavy_modes -> heavy model
        5. Query length > 150 or complex keywords -> heavy model
        6. mode in light modes -> light model
        7. Light keywords -> light model
        8. Default -> light model

        When *thinking_level* is provided and a thinking model is selected,
        ``selected_config.thinking_budget_tokens`` is set based on the level.
        """
        # Quiet hours: use cheapest model overnight
        if self._quiet_hours and self._quiet_model:
            start, end = self._quiet_hours
            hour = datetime.now().hour
            in_quiet = (hour >= start or hour < end) if start > end else (start <= hour < end)
            if in_quiet:
                logger.debug(f"Quiet hours ({start}:00-{end}:00) -> {self._quiet_model.display_name}")
                return self._quiet_model

        # Budget pressure: graceful degradation
        pressure = get_budget_pressure()
        if pressure >= 0.95:
            logger.info(f"Budget pressure {pressure:.0%} -> forcing light model")
            return self.light_model
        if pressure >= 0.80:
            if think or thinking_level > ThinkingLevel.OFF:
                logger.info(f"Budget pressure {pressure:.0%} -> downgrading thinking to light")
            return self.light_model

        # Consumer-registered per-mode override (after budget guards)
        if mode and mode in self._mode_overrides:
            override = self._mode_overrides[mode]
            logger.debug(f"Mode override '{mode}' -> {override.display_name}")
            return override

        if think or thinking_level > ThinkingLevel.OFF or mode == "think":
            logger.debug(f"thinking requested -> thinking model: {self.thinking_model.display_name}")
            # Return a copy with thinking_budget_tokens set
            budget = THINKING_BUDGET_MAP.get(
                thinking_level if thinking_level > ThinkingLevel.OFF else ThinkingLevel.MEDIUM,
                4096,
            )
            # Cap thinking budget under moderate pressure
            if pressure >= 0.60:
                budget = min(budget, THINKING_BUDGET_MAP[ThinkingLevel.LOW])
                logger.debug(f"Budget pressure {pressure:.0%} -> capping thinking to LOW")
            return ModelConfig(
                name=self.thinking_model.name,
                display_name=self.thinking_model.display_name,
                provider=self.thinking_model.provider,
                is_heavy=self.thinking_model.is_heavy,
                is_thinking=self.thinking_model.is_thinking,
                max_tokens=self.thinking_model.max_tokens,
                cost_tier=self.thinking_model.cost_tier,
                thinking_budget_tokens=budget,
            )

        # Signal-based: lightweight channels stay light
        if signals and signals.channel in ("boot", "heartbeat"):
            logger.debug(f"Channel '{signals.channel}' -> light model: {self.light_model.display_name}")
            return self.light_model

        # Signal-based: substantial tool results, synthesis, or escalation → heavy
        if signals and (
            signals.tool_result_chars > self._tool_result_chars_threshold
            or signals.needs_synthesis
            or signals.needs_escalation
        ):
            logger.debug(
                f"Signal upgrade (chars={signals.tool_result_chars}, "
                f"synthesis={signals.needs_synthesis}, escalation={signals.needs_escalation})"
                f" -> heavy model: {self.heavy_model.display_name}"
            )
            return self.heavy_model

        if mode:
            if heavy_modes and mode in heavy_modes:
                logger.debug(f"Mode '{mode}' in heavy_modes -> heavy model: {self.heavy_model.display_name}")
                return self.heavy_model
            if mode.lower() in _LIGHT_MODES:
                logger.debug(f"Mode '{mode}' -> light model: {self.light_model.display_name}")
                return self.light_model
            # Unknown modes (smart, explore, data, etc.) fall through to query heuristics
            logger.debug(f"Mode '{mode}' not in heavy/light sets, falling through to query heuristics")

        query_lower = query.lower()
        if len(query) > self._complex_query_chars_threshold or any(kw in query_lower for kw in _COMPLEX_KEYWORDS):
            logger.debug(f"Complex query -> heavy model: {self.heavy_model.display_name}")
            return self.heavy_model

        if any(kw in query_lower for kw in _LIGHT_KEYWORDS):
            logger.debug(f"Light query -> light model: {self.light_model.display_name}")
            return self.light_model

        return self.light_model

    def get_model_for_task(self, task_type: str, query_mode: str | None = None) -> ModelConfig:
        """Pick light or heavy model based on task type or query mode.

        Delegates to select() for unified logic.
        """
        return self.select(task_type, mode=query_mode)

    @staticmethod
    def search_catalog(query: str) -> list[ModelConfig]:
        """Search MODEL_CATALOG for models matching a partial query string."""
        query_lower = query.strip().lower()
        matches = []
        for provider_models in MODEL_CATALOG.values():
            for model in provider_models:
                if query_lower in model.name.lower() or query_lower in model.display_name.lower():
                    matches.append(model)
        return matches

    def get_current_models(self) -> dict[str, ModelConfig]:
        return {"light": self.light_model, "heavy": self.heavy_model, "thinking": self.thinking_model}

    @property
    def active_family(self) -> str | None:
        """Current family name, or None if models are from mixed providers."""
        return self._active_family

    def _infer_family(self) -> str | None:
        """Check if all 3 models share a provider — return it, or None if mixed."""
        providers = {self.light_model.provider, self.heavy_model.provider, self.thinking_model.provider}
        if len(providers) == 1:
            provider = providers.pop()
            if provider in get_available_families():
                return provider
        return None

    def switch_family(self, provider_key: str) -> tuple[ModelConfig, ModelConfig, ModelConfig] | None:
        """Switch all 3 tiers to a provider family. Returns the new models or None if invalid."""
        family = get_family_models(provider_key)
        if family is None:
            return None
        light, heavy, thinking = family
        self.light_model = light
        self.heavy_model = heavy
        self.thinking_model = thinking
        self._active_family = provider_key
        self._save_settings()
        logger.info(
            f"Switched model family to {provider_key}: "
            f"{light.display_name} / {heavy.display_name} / {thinking.display_name}"
        )
        return family

    def set_models(
        self,
        light: ModelConfig | None = None,
        heavy: ModelConfig | None = None,
        thinking: ModelConfig | None = None,
    ) -> None:
        if light:
            self.light_model = light
        if heavy:
            self.heavy_model = heavy
        if thinking:
            self.thinking_model = thinking
        self._active_family = self._infer_family()
        self._save_settings()

    def set_mode_overrides(self, mode_overrides: dict[str, ModelConfig]) -> None:
        """Replace mode overrides at runtime."""
        self._mode_overrides = dict(mode_overrides)

    def set_thresholds(
        self,
        *,
        tool_result_chars_threshold: int | None = None,
        complex_query_chars_threshold: int | None = None,
    ) -> None:
        """Update routing thresholds at runtime."""
        if tool_result_chars_threshold is not None:
            self._tool_result_chars_threshold = max(0, int(tool_result_chars_threshold))
        if complex_query_chars_threshold is not None:
            self._complex_query_chars_threshold = max(0, int(complex_query_chars_threshold))

    # --- persistence --------------------------------------------------------

    def _save_settings(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._settings_path), exist_ok=True)
            settings = {
                "light_model": {"name": self.light_model.name, "provider": self.light_model.provider},
                "heavy_model": {"name": self.heavy_model.name, "provider": self.heavy_model.provider},
                "thinking_model": {"name": self.thinking_model.name, "provider": self.thinking_model.provider},
                "active_family": self._active_family,
            }
            with open(self._settings_path, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model settings: {e}")

    def _load_saved_settings(self) -> tuple[ModelConfig | None, ModelConfig | None, ModelConfig | None, str | None]:
        try:
            if not os.path.exists(self._settings_path):
                return None, None, None, None
            with open(self._settings_path) as f:
                settings = json.load(f)

            light = self._find_in_catalog(settings.get("light_model", {}))
            heavy = self._find_in_catalog(settings.get("heavy_model", {}))
            thinking = self._find_in_catalog(settings.get("thinking_model", {}))
            family = settings.get("active_family")
            return light, heavy, thinking, family
        except Exception as e:
            logger.error(f"Failed to load model settings: {e}")
            return None, None, None, None

    @staticmethod
    def _find_in_catalog(data: dict) -> ModelConfig | None:
        name = data.get("name")
        provider = data.get("provider")
        if not provider or provider not in MODEL_CATALOG:
            return None
        for model in MODEL_CATALOG[provider]:
            if model.name == name:
                return model
        return None


# --- module-level singleton ------------------------------------------------

_model_selector: ModelSelector | None = None


def get_model_selector(**kwargs) -> ModelSelector:
    """Get or create the global ModelSelector instance."""
    global _model_selector
    if _model_selector is None:
        _model_selector = ModelSelector(**kwargs)
    return _model_selector


def reset_model_selector() -> None:
    global _model_selector
    _model_selector = None
