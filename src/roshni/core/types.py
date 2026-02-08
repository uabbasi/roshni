"""Shared type aliases used across roshni."""

from pathlib import Path
from typing import Any

# Config value types
ConfigDict = dict[str, Any]
ConfigValue = str | int | float | bool | list | dict | None

# Path types
PathLike = str | Path
