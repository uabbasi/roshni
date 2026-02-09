"""Persistent approval grant store for agent tool execution."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger


class ApprovalStore:
    """Persistent store for tool approval grants.

    When a user approves a tool, the grant is remembered so the agent
    doesn't ask again.  Backed by a JSON file (default location:
    ``~/.weeklies-data/memory/approval-grants.json``).
    """

    def __init__(self, path: str | Path):
        self._path = Path(path).expanduser()
        self._grants: set[str] = set()
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._grants = set(data) if isinstance(data, list) else set()
            except Exception as e:
                logger.warning(f"Could not load approval grants from {self._path}: {e}")
                self._grants = set()

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(sorted(self._grants), indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Could not save approval grants to {self._path}: {e}")

    def is_approved(self, tool_name: str) -> bool:
        return tool_name in self._grants

    def grant(self, tool_name: str) -> None:
        if tool_name not in self._grants:
            self._grants.add(tool_name)
            self._save()
            logger.info(f"Approval granted and saved for tool: {tool_name}")

    def revoke(self, tool_name: str) -> None:
        if tool_name in self._grants:
            self._grants.discard(tool_name)
            self._save()
            logger.info(f"Approval revoked for tool: {tool_name}")
