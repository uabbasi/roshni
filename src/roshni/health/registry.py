"""
Health-collector plugin registry.

Discovers collectors at runtime via ``importlib.metadata`` entry points
(group: ``roshni.health_collectors``).  Third-party packages can register
collectors in their own ``pyproject.toml``:

    [project.entry-points."roshni.health_collectors"]
    my_device = "my_package.collector:MyCollector"
"""

from importlib.metadata import entry_points
from typing import Any

from loguru import logger

from .collector import HealthCollector


class HealthCollectorRegistry:
    """Discover and manage health-data collector plugins."""

    def __init__(self):
        self._collectors: dict[str, type] = {}

    def discover(self) -> dict[str, type]:
        """Scan entry points and return {name: collector_class}."""
        eps = entry_points(group="roshni.health_collectors")
        for ep in eps:
            try:
                cls = ep.load()
                if isinstance(cls, type) and issubclass(cls, HealthCollector):
                    self._collectors[ep.name] = cls
                    logger.debug(f"Discovered health collector: {ep.name}")
                else:
                    # Protocol check for non-class callables
                    self._collectors[ep.name] = cls
            except Exception as e:
                logger.warning(f"Failed to load health collector '{ep.name}': {e}")

        return dict(self._collectors)

    def register(self, name: str, collector_class: type) -> None:
        """Manually register a collector (useful for testing)."""
        self._collectors[name] = collector_class

    def get(self, name: str) -> type | None:
        """Get a registered collector class by name."""
        return self._collectors.get(name)

    def list_names(self) -> list[str]:
        return list(self._collectors.keys())

    def create(self, name: str, **config: Any) -> HealthCollector:
        """Instantiate a collector by name with the given config."""
        cls = self._collectors.get(name)
        if cls is None:
            raise KeyError(f"No collector registered as '{name}'. Available: {self.list_names()}")
        return cls(**config)
