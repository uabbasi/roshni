"""Memory system — persistent, file-backed agent memory.

Provides a MemoryManager that reads/writes a MEMORY.md file organised into
sections, plus daily session notes in a ``memory/`` subdirectory. The daily
notes provide a raw chronological log alongside the curated MEMORY.md.

A factory is included to expose memory as a tool the LLM can call directly.
"""

from __future__ import annotations

import re
import threading
from datetime import date
from pathlib import Path
from typing import Any

from loguru import logger

# Patterns that suggest the user is expressing something the agent should remember.
# Tuned for conversational triggers — "always …", "never …", "remember …", etc.
_MEMORY_TRIGGER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\balways\b", re.IGNORECASE),
    re.compile(r"\bnever\b", re.IGNORECASE),
    re.compile(r"\bremember\b.*\b(that|this|to)\b", re.IGNORECASE),
    re.compile(r"\bdon'?t forget\b", re.IGNORECASE),
    re.compile(r"\bfrom now on\b", re.IGNORECASE),
    re.compile(r"\bgoing forward\b", re.IGNORECASE),
    re.compile(r"\bprefer\b", re.IGNORECASE),
    re.compile(r"\bi like\b", re.IGNORECASE),
    re.compile(r"\bi hate\b", re.IGNORECASE),
    re.compile(r"\bkeep in mind\b", re.IGNORECASE),
    re.compile(r"\bmake sure you\b", re.IGNORECASE),
    re.compile(r"\bwhen I ask\b", re.IGNORECASE),
    re.compile(r"\bwhenever I\b", re.IGNORECASE),
    re.compile(r"\bby default\b", re.IGNORECASE),
]

# Patterns for implicit memorable events — no explicit "remember" but significant life events.
_MEMORY_OFFER_PATTERNS: list[re.Pattern[str]] = [
    # Life events
    re.compile(r"\b(got promoted|quit my job|moved to|broke up|got engaged|got married|had a baby)\b", re.IGNORECASE),
    re.compile(r"\b(started a new job|left my job|got fired|got laid off|retired)\b", re.IGNORECASE),
    # Decisions
    re.compile(r"\b(decided to|going to switch to|chose to|committed to|signed up for)\b", re.IGNORECASE),
    # Milestones
    re.compile(r"\b(hit my goal|personal record|first time I|finally did|reached my target)\b", re.IGNORECASE),
    # Reflective
    re.compile(r"\b(really struggling with|realized that|had an epiphany|it hit me that)\b", re.IGNORECASE),
]

VALID_SECTIONS = frozenset({"decisions", "open_loops", "preferences", "recurring_patterns"})


def detect_memory_trigger(message: str) -> bool:
    """Return True if *message* contains a memory trigger pattern."""
    return any(p.search(message) for p in _MEMORY_TRIGGER_PATTERNS)


def detect_memorable_event(message: str) -> bool:
    """Return True if *message* describes a significant event worth offering to save."""
    return any(p.search(message) for p in _MEMORY_OFFER_PATTERNS)


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

    # ------------------------------------------------------------------
    # Daily notes
    # ------------------------------------------------------------------

    @property
    def _daily_dir(self) -> Path:
        """Directory for daily session notes, sibling to MEMORY.md."""
        return self._path.parent / "memory"

    def _daily_path(self, day: date | None = None) -> Path:
        day = day or date.today()
        return self._daily_dir / f"{day.isoformat()}.md"

    def append_daily_note(self, note: str, *, day: date | None = None) -> str:
        """Append a timestamped line to today's daily note file.

        Creates the ``memory/`` directory and ``YYYY-MM-DD.md`` file on first use.

        Returns:
            Confirmation string.
        """
        note = note.strip()
        if not note:
            return "Error: empty note"

        path = self._daily_path(day)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_text(f"# {(day or date.today()).isoformat()}\n\n", encoding="utf-8")
            with path.open("a", encoding="utf-8") as f:
                f.write(f"- {note}\n")

        logger.debug(f"Daily note appended: {note[:60]}...")
        return f"Noted: {note[:80]}"

    def get_daily_context(self, *, day: date | None = None) -> str:
        """Return today's daily notes for system prompt injection."""
        path = self._daily_path(day)
        if not path.exists():
            return ""
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return ""
        return f"[DAILY NOTES]\n{text}\n[/DAILY NOTES]"

    # ------------------------------------------------------------------
    # Context & triggers
    # ------------------------------------------------------------------

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
