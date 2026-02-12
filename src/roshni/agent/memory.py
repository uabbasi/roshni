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

    # Threshold for auto-compaction: when a section exceeds this many entries
    AUTO_COMPACT_THRESHOLD = 50

    # Jaccard similarity threshold for near-dedup
    NEAR_DEDUP_THRESHOLD = 0.7

    def save(self, section: str, content: str, *, auto_compact: bool = False) -> str:
        """Append *content* under *section*. Returns confirmation string.

        Args:
            section: Section name to save under.
            content: The information to remember.
            auto_compact: If True, run compaction when the section exceeds
                ``AUTO_COMPACT_THRESHOLD`` entries.
        """
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

        if auto_compact:
            section_entries = self._count_section_entries(section)
            if section_entries > self.AUTO_COMPACT_THRESHOLD:
                logger.info(f"Section '{section}' has {section_entries} entries, auto-compacting...")
                self.compact_section(section)

        return f"Saved to {section}: {content[:80]}"

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    def _parse_sections(self, text: str) -> dict[str, list[str]]:
        """Parse MEMORY.md into {section_name: [entries]}.

        Entries are the raw lines (including the ``- `` or ``* `` prefix).
        Non-entry lines within a section are preserved as-is.
        """
        sections: dict[str, list[str]] = {}
        current_section: str | None = None

        for line in text.split("\n"):
            header_match = _SECTION_HEADER_RE.match(line)
            if header_match:
                current_section = header_match.group(1).strip()
                sections[current_section] = []
            elif current_section is not None:
                sections[current_section].append(line)

        return sections

    def _count_section_entries(self, section: str) -> int:
        """Count entries in a section without full parse overhead."""
        if not self._path.exists():
            return 0
        text = self._path.read_text(encoding="utf-8")
        parsed = self._parse_sections(text)
        entries = parsed.get(section, [])
        return sum(1 for line in entries if line.strip().startswith(("- ", "* ")))

    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        """Extract lowercase keyword tokens from text for similarity comparison."""
        # Strip bullet prefix and date/topic prefix for keyword extraction
        cleaned = re.sub(r"^[-*]\s*", "", text.strip())
        cleaned = re.sub(r"^\*\*\d{4}-\d{2}-\d{2}[^*]*\*\*\s*", "", cleaned)
        words = re.findall(r"\b[a-z]{3,}\b", cleaned.lower())
        return set(words)

    @staticmethod
    def _extract_date_topic(entry: str) -> str | None:
        """Extract date+topic prefix like '2026-02-10 — ADU:' from an entry."""
        m = re.match(r"^[-*]\s*\*\*(\d{4}-\d{2}-\d{2}\s*[—–-]\s*[^:*]+):", entry.strip())
        return m.group(1).strip() if m else None

    def _dedup_entries(self, entries: list[str]) -> list[str]:
        """Remove exact and near-duplicate entries from a list.

        Strategy:
        1. Exact dedup: remove entries with identical stripped text (keep first).
        2. Near dedup: for entries sharing the same date+topic prefix, keep only
           the longest version. For entries without a date+topic prefix, use
           Jaccard similarity on keywords — if similarity >= threshold, keep the
           longer entry.
        """
        # Filter to just entry lines vs non-entry lines (blank lines, etc.)
        entry_lines = []
        non_entry_lines = []
        for line in entries:
            if line.strip().startswith(("- ", "* ")):
                entry_lines.append(line)
            else:
                non_entry_lines.append(line)

        # Step 1: Exact dedup (keep first occurrence)
        seen_exact: set[str] = set()
        unique_entries: list[str] = []
        for line in entry_lines:
            normalized = line.strip()
            if normalized in seen_exact:
                continue
            seen_exact.add(normalized)
            unique_entries.append(line)

        # Step 2: Near dedup — date+topic prefix grouping
        date_topic_groups: dict[str, list[int]] = {}
        no_prefix_indices: list[int] = []

        for i, line in enumerate(unique_entries):
            dt = self._extract_date_topic(line)
            if dt:
                date_topic_groups.setdefault(dt, []).append(i)
            else:
                no_prefix_indices.append(i)

        # For each date+topic group, keep only the longest entry
        remove_indices: set[int] = set()
        for indices in date_topic_groups.values():
            if len(indices) <= 1:
                continue
            # Keep the longest entry (most information)
            longest_idx = max(indices, key=lambda i: len(unique_entries[i]))
            for idx in indices:
                if idx != longest_idx:
                    remove_indices.add(idx)

        # Step 3: Near dedup — Jaccard similarity for non-prefixed entries
        for i in range(len(no_prefix_indices)):
            idx_i = no_prefix_indices[i]
            if idx_i in remove_indices:
                continue
            kw_i = self._extract_keywords(unique_entries[idx_i])
            if not kw_i:
                continue
            for j in range(i + 1, len(no_prefix_indices)):
                idx_j = no_prefix_indices[j]
                if idx_j in remove_indices:
                    continue
                kw_j = self._extract_keywords(unique_entries[idx_j])
                if not kw_j:
                    continue
                intersection = kw_i & kw_j
                union = kw_i | kw_j
                similarity = len(intersection) / len(union) if union else 0.0
                if similarity >= self.NEAR_DEDUP_THRESHOLD:
                    # Keep the longer entry
                    shorter = idx_i if len(unique_entries[idx_i]) <= len(unique_entries[idx_j]) else idx_j
                    remove_indices.add(shorter)
                    if shorter == idx_i:
                        break  # idx_i removed, stop comparing it

        result = [line for i, line in enumerate(unique_entries) if i not in remove_indices]

        # Reconstruct: non-entry lines first (usually blank), then entries
        # Preserve a single blank line between header and entries
        cleaned_non_entry = [line for line in non_entry_lines if line.strip()]
        return cleaned_non_entry + result

    def compact(self) -> dict[str, int]:
        """Deduplicate and compact all memory sections.

        Returns:
            Dict with counts: {"removed": N, "sections_compacted": N}
        """
        with self._lock:
            text = self._path.read_text(encoding="utf-8")

            # Preserve the title line (# Agent Memory or similar)
            lines = text.split("\n")
            title = lines[0] if lines and lines[0].startswith("# ") else ""

            sections = self._parse_sections(text)
            total_before = 0
            total_after = 0
            sections_compacted = 0

            rebuilt_parts = [title, ""] if title else []

            for section_name in sections:
                entries = sections[section_name]
                before_count = sum(1 for e in entries if e.strip().startswith(("- ", "* ")))
                total_before += before_count

                deduped = self._dedup_entries(entries)
                after_count = sum(1 for e in deduped if e.strip().startswith(("- ", "* ")))
                total_after += after_count

                if after_count < before_count:
                    sections_compacted += 1

                rebuilt_parts.append(f"\n## {section_name}")
                rebuilt_parts.extend(deduped)

            self._path.write_text("\n".join(rebuilt_parts) + "\n", encoding="utf-8")

        removed = total_before - total_after
        logger.info(f"Memory compacted: {removed} entries removed from {sections_compacted} sections")
        return {"removed": removed, "sections_compacted": sections_compacted}

    def compact_section(self, section: str) -> dict[str, int]:
        """Deduplicate and compact a single memory section.

        Returns:
            Dict with counts: {"removed": N}
        """
        with self._lock:
            text = self._path.read_text(encoding="utf-8")
            sections = self._parse_sections(text)

            if section not in sections:
                return {"removed": 0}

            entries = sections[section]
            before_count = sum(1 for e in entries if e.strip().startswith(("- ", "* ")))
            deduped = self._dedup_entries(entries)
            after_count = sum(1 for e in deduped if e.strip().startswith(("- ", "* ")))

            # Rebuild just this section in the original text
            sections[section] = deduped

            # Preserve the title line
            lines = text.split("\n")
            title = lines[0] if lines and lines[0].startswith("# ") else ""
            rebuilt_parts = [title, ""] if title else []

            for section_name in sections:
                rebuilt_parts.append(f"\n## {section_name}")
                rebuilt_parts.extend(sections[section_name])

            self._path.write_text("\n".join(rebuilt_parts) + "\n", encoding="utf-8")

        removed = before_count - after_count
        logger.info(f"Section '{section}' compacted: {removed} entries removed")
        return {"removed": removed}

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


def create_compact_memory_tool(memory_manager: MemoryManager) -> Any:
    """Factory: build a ToolDefinition for the compact_memory action."""
    from roshni.agent.tools import ToolDefinition

    def _compact_memory(section: str | None = None) -> str:
        if section:
            result = memory_manager.compact_section(section)
            return f"Compacted section '{section}': {result['removed']} duplicates removed."
        result = memory_manager.compact()
        return (
            f"Compacted memory: {result['removed']} duplicates removed across {result['sections_compacted']} sections."
        )

    return ToolDefinition(
        name="compact_memory",
        description=(
            "Deduplicate and compact memory by removing exact and near-duplicate entries. "
            "Run this periodically or when memory feels bloated. "
            "Optionally target a specific section."
        ),
        parameters={
            "type": "object",
            "properties": {
                "section": {
                    "type": "string",
                    "enum": sorted(VALID_SECTIONS),
                    "description": "Optional: compact only this section. If omitted, compact all sections.",
                },
            },
        },
        function=_compact_memory,
        permission="write",
        requires_approval=False,
    )
