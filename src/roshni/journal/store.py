"""JournalStore protocol — the contract for journal backends.

Any system that stores dated text entries (Obsidian, Notion export,
plain directory of markdown files) can implement this protocol and
plug into roshni's search/processing pipeline.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class JournalStore(Protocol):
    """Protocol for accessing journal entries.

    Implementations provide read-only access to a collection of
    dated text entries. The simplest implementation is a directory
    of markdown files named ``YYYY-MM-DD.md``.
    """

    def list_entries(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Path]:
        """Return paths to journal entries in the date range.

        Args:
            start: Earliest date (inclusive). None = no lower bound.
            end: Latest date (inclusive). None = no upper bound.

        Returns:
            List of paths, sorted chronologically.
        """
        ...

    def read_entry(self, path: Path) -> str:
        """Read the text content of an entry.

        Args:
            path: Path returned by ``list_entries``.

        Returns:
            Full text content of the entry.
        """
        ...

    def get_metadata(self, path: Path) -> dict:
        """Extract metadata from an entry (frontmatter, filename date, etc.).

        Args:
            path: Path returned by ``list_entries``.

        Returns:
            Dict with at least ``"date"`` key if parseable from filename.
        """
        ...
