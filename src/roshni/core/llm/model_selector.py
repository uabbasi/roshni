"""
Model selection for light vs. heavy thinking tasks.

Provides a programmatic API for choosing between fast/cheap models and
slower/smarter ones based on task complexity.  Settings are persisted
to disk so they survive restarts.
"""

import json
import os

from loguru import logger

from .config import MODEL_CATALOG, ModelConfig


class ModelSelector:
    """Manages model selection for light and heavy thinking tasks."""

    def __init__(
        self,
        light_model: ModelConfig | None = None,
        heavy_model: ModelConfig | None = None,
        settings_path: str = "~/.roshni-data/model_settings.json",
    ):
        self._settings_path = os.path.expanduser(settings_path)

        saved_light, saved_heavy = self._load_saved_settings()
        self.light_model = light_model or saved_light or self._default_light()
        self.heavy_model = heavy_model or saved_heavy or self._default_heavy()

    @staticmethod
    def _default_light() -> ModelConfig:
        return MODEL_CATALOG["gemini"][1]  # Gemini 2.5 Flash

    @staticmethod
    def _default_heavy() -> ModelConfig:
        return MODEL_CATALOG["gemini"][2]  # Gemini 2.5 Pro

    def get_model_for_task(self, task_type: str, query_mode: str | None = None) -> ModelConfig:
        """Pick light or heavy model based on task type or query mode."""
        if query_mode:
            light_modes = {"summary", "answer", "timeline"}
            if query_mode.lower() in light_modes:
                logger.debug(f"Mode '{query_mode}' -> light model: {self.light_model.display_name}")
                return self.light_model
            logger.debug(f"Mode '{query_mode}' -> heavy model: {self.heavy_model.display_name}")
            return self.heavy_model

        task_lower = task_type.lower()
        light_keywords = ["summary", "summarize", "list", "quick", "simple", "brief"]
        if any(kw in task_lower for kw in light_keywords):
            return self.light_model
        return self.heavy_model

    def get_current_models(self) -> dict[str, ModelConfig]:
        return {"light": self.light_model, "heavy": self.heavy_model}

    def set_models(self, light: ModelConfig | None = None, heavy: ModelConfig | None = None) -> None:
        if light:
            self.light_model = light
        if heavy:
            self.heavy_model = heavy
        self._save_settings()

    # --- persistence --------------------------------------------------------

    def _save_settings(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._settings_path), exist_ok=True)
            settings = {
                "light_model": {"name": self.light_model.name, "provider": self.light_model.provider},
                "heavy_model": {"name": self.heavy_model.name, "provider": self.heavy_model.provider},
            }
            with open(self._settings_path, "w") as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model settings: {e}")

    def _load_saved_settings(self) -> tuple[ModelConfig | None, ModelConfig | None]:
        try:
            if not os.path.exists(self._settings_path):
                return None, None
            with open(self._settings_path) as f:
                settings = json.load(f)

            light = self._find_in_catalog(settings.get("light_model", {}))
            heavy = self._find_in_catalog(settings.get("heavy_model", {}))
            return light, heavy
        except Exception as e:
            logger.error(f"Failed to load model settings: {e}")
            return None, None

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
