"""
Model selection for light, heavy, and thinking tasks.

Provides a programmatic API for choosing between fast/cheap models,
slower/smarter ones, and reasoning models based on task complexity.
Settings are persisted to disk so they survive restarts.
"""

import json
import os

from loguru import logger

from .config import MODEL_CATALOG, ModelConfig

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
    ):
        self._settings_path = os.path.expanduser(settings_path)

        saved_light, saved_heavy, saved_thinking = self._load_saved_settings()
        self.light_model = light_model or saved_light or self._default_light()
        self.heavy_model = heavy_model or saved_heavy or self._default_heavy()
        self.thinking_model = thinking_model or saved_thinking or self._default_thinking()

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
    ) -> ModelConfig:
        """Single entry point for model selection.

        Priority:
        1. think=True -> thinking model
        2. mode in heavy_modes -> heavy model
        3. Query length > 150 or complex keywords -> heavy model
        4. mode in light modes -> light model
        5. Light keywords -> light model
        6. Default -> light model
        """
        if think:
            logger.debug(f"think=True -> thinking model: {self.thinking_model.display_name}")
            return self.thinking_model

        if mode:
            if heavy_modes and mode in heavy_modes:
                logger.debug(f"Mode '{mode}' in heavy_modes -> heavy model: {self.heavy_model.display_name}")
                return self.heavy_model
            if mode.lower() in _LIGHT_MODES:
                logger.debug(f"Mode '{mode}' -> light model: {self.light_model.display_name}")
                return self.light_model
            # Any other explicit mode defaults to heavy
            logger.debug(f"Mode '{mode}' -> heavy model: {self.heavy_model.display_name}")
            return self.heavy_model

        query_lower = query.lower()
        if len(query) > 150 or any(kw in query_lower for kw in _COMPLEX_KEYWORDS):
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

    def get_current_models(self) -> dict[str, ModelConfig]:
        return {"light": self.light_model, "heavy": self.heavy_model, "thinking": self.thinking_model}

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
        self._save_settings()

    # --- persistence --------------------------------------------------------

    def _save_settings(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._settings_path), exist_ok=True)
            settings = {
                "light_model": {"name": self.light_model.name, "provider": self.light_model.provider},
                "heavy_model": {"name": self.heavy_model.name, "provider": self.heavy_model.provider},
                "thinking_model": {"name": self.thinking_model.name, "provider": self.thinking_model.provider},
            }
            with open(self._settings_path, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model settings: {e}")

    def _load_saved_settings(self) -> tuple[ModelConfig | None, ModelConfig | None, ModelConfig | None]:
        try:
            if not os.path.exists(self._settings_path):
                return None, None, None
            with open(self._settings_path) as f:
                settings = json.load(f)

            light = self._find_in_catalog(settings.get("light_model", {}))
            heavy = self._find_in_catalog(settings.get("heavy_model", {}))
            thinking = self._find_in_catalog(settings.get("thinking_model", {}))
            return light, heavy, thinking
        except Exception as e:
            logger.error(f"Failed to load model settings: {e}")
            return None, None, None

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
