"""Memory system — persistent, file-backed agent memory.

Provides a MemoryManager that reads/writes a MEMORY.md file organised into
sections, and a factory to expose it as a tool the LLM can call directly.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

from loguru import logger

# Patterns that suggest the user is expressing something the agent should remember.
# Tuned for conversational triggers — "always …", "never …", "remember …", etc.
_MEMORY_TRIGGER_PATTERNS: list[re.Pattern[str]] = [
    # TODO: Tune these patterns based on downstream usage.
    re.compile(r"\balways\b", re.IGNORECASE),
    re.compile(r"\bnever\b", re.IGNORECASE),
    re.compile(r"\bremember\b", re.IGNORECASE),
    re.compile(r"\bdon'?t forget\b", re.IGNORECASE),
    re.compile(r"\bfrom now on\b", re.IGNORECASE),
    re.compile(r"\bgoing forward\b", re.IGNORECASE),
    re.compile(r"\bprefer\b", re.IGNORECASE),
    re.compile(r"\bi like\b", re.IGNORECASE),
    re.compile(r"\bi hate\b", re.IGNORECASE),
    re.compile(r"\bkeep in mind\b", re.IGNORECASE),
]

VALID_SECTIONS = frozenset({"decisions", "open_loops", "preferences", "recurring_patterns"})

_SECTION_HEADER_RE = re.compile(r"^## (.+)$", re.MULTILINE)


class MemoryManager:
    """Read/write a MEMORY.md file with named sections.

    Thread-safe: concurrent ``save()`` calls are serialised via a lock.
    """

    def __init__(self, memory_path: str | Path) -> None:
        self._path = Path(memory_path).expanduser()
        self._lock = threading.Lock()
        self._ensure_file()

    def _ensure_file(self) -> None:
        """Create MEMORY.md with section headers if it doesn't exist."""
        if self._path.exists():
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Agent Memory\n"]
        for section in sorted(VALID_SECTIONS):
            lines.append(f"\n## {section}\n")
        self._path.write_text("\n".join(lines), encoding="utf-8")
        logger.debug(f"Created memory file: {self._path}")

    def save(self, section: str, content: str) -> str:
        """Append *content* under *section*. Returns confirmation string."""
        if section not in VALID_SECTIONS:
            return f"Error: unknown section '{section}'. Valid: {', '.join(sorted(VALID_SECTIONS))}"
        content = content.strip()
        if not content:
            return "Error: empty content"

        with self._lock:
            text = self._path.read_text(encoding="utf-8")
            header = f"## {section}"
            idx = text.find(header)
            if idx == -1:
                text += f"\n{header}\n\n- {content}\n"
            else:
                insert_at = idx + len(header)
                # Skip past the newline after the header
                while insert_at < len(text) and text[insert_at] in "\n\r":
                    insert_at += 1
                text = text[:insert_at] + f"- {content}\n" + text[insert_at:]
            self._path.write_text(text, encoding="utf-8")

        logger.debug(f"Memory saved to [{section}]: {content[:60]}...")
        return f"Saved to {section}: {content[:80]}"

    def get_context(self, query: str | None = None) -> str:
        """Return the full memory file contents for system prompt injection."""
        if not self._path.exists():
            return ""
        text = self._path.read_text(encoding="utf-8").strip()
        if not text or text == "# Agent Memory":
            return ""
        return f"[MEMORY]\n{text}\n[/MEMORY]"

    def detect_trigger(self, message: str) -> bool:
        """Return True if *message* contains a memory trigger pattern."""
        return any(p.search(message) for p in _MEMORY_TRIGGER_PATTERNS)


def create_save_memory_tool(memory_manager: MemoryManager) -> Any:
    """Factory: build a ToolDefinition for the save_memory action."""
    from roshni.agent.tools import ToolDefinition

    def _save_memory(section: str, content: str) -> str:
        return memory_manager.save(section, content)

    return ToolDefinition(
        name="save_memory",
        description=(
            "Save important information to persistent memory. "
            "Use when the user expresses a preference, makes a decision, "
            "or asks you to remember something. "
            f"Sections: {', '.join(sorted(VALID_SECTIONS))}."
        ),
        parameters={
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "enum": sorted(VALID_SECTIONS),
                    "description": "Memory section to save to",
                },
                "content": {
                    "type": "string",
                    "description": "The information to remember",
                },
            },
            "required": ["section", "content"],
        },
        function=_save_memory,
        permission="write",
        requires_approval=False,
    )
