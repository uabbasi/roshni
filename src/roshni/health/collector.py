"""
HealthCollector protocol and base class.

Any data source (Fitbit, Apple Health, Garmin, Oura, …) implements
this interface so the pipeline can treat them uniformly.
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import Any, Protocol, runtime_checkable

from .models import DailyHealth


@runtime_checkable
class HealthCollector(Protocol):
    """Protocol that every health data collector must satisfy."""

    name: str

    def collect(self, start_date: date, end_date: date) -> list[DailyHealth]:
        """Collect health data for a date range (inclusive)."""
        ...

    def validate(self) -> bool:
        """Check that credentials / connectivity are working."""
        ...

    def get_config_schema(self) -> dict[str, Any]:
        """Describe required config keys so consumers know what to provide."""
        ...


class BaseCollector(ABC):
    """Optional ABC providing shared plumbing for collectors.

    Subclass this if you want built-in caching, rate-limit tracking,
    and stats.  Or just implement the ``HealthCollector`` protocol directly.
    """

    name: str = "base"

    def __init__(self, **config: Any):
        self.config = config
        self.stats: dict[str, int] = {"api_calls": 0, "cache_hits": 0, "errors": 0}

    @abstractmethod
    def collect(self, start_date: date, end_date: date) -> list[DailyHealth]:
        """Collect health data for a date range."""

    def validate(self) -> bool:
        """Default validation — override for real checks."""
        return True

    def get_config_schema(self) -> dict[str, Any]:
        """Override to advertise required config keys."""
        return {}

    def _date_range(self, start: date, end: date) -> list[date]:
        """Generate a list of dates from start to end (inclusive)."""
        from datetime import timedelta

        days = (end - start).days + 1
        return [start + timedelta(days=i) for i in range(days)]
