"""Vault manager — manages the agent's local markdown vault structure."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


class VaultManager:
    """Manages the file-based vault for an agent.

    The vault structure is:
        {vault_path}/{agent_dir}/
            persona/    — IDENTITY.md, SOUL.md, USER.md, AGENTS.md
            memory/     — MEMORY.md
            tasks/      — Task .md files with YAML frontmatter
                _archive/   — Archived completed tasks
                _index.md   — Auto-generated dashboard
            projects/   — Project overview .md files
            people/     — Person .md files with frontmatter
            ideas/      — Idea .md files with frontmatter
            admin/      — audit.md
    """

    _SUBDIRS = ("persona", "memory", "tasks", "projects", "people", "ideas", "admin")

    def __init__(self, vault_path: str | Path, agent_dir: str = "jarvis") -> None:
        self.vault_path = Path(vault_path).expanduser()
        self.agent_dir = agent_dir

    # -- directory properties --------------------------------------------------

    @property
    def base_dir(self) -> Path:
        return self.vault_path / self.agent_dir

    @property
    def persona_dir(self) -> Path:
        return self.base_dir / "persona"

    @property
    def memory_dir(self) -> Path:
        return self.base_dir / "memory"

    @property
    def tasks_dir(self) -> Path:
        return self.base_dir / "tasks"

    @property
    def projects_dir(self) -> Path:
        return self.base_dir / "projects"

    @property
    def people_dir(self) -> Path:
        return self.base_dir / "people"

    @property
    def ideas_dir(self) -> Path:
        return self.base_dir / "ideas"

    @property
    def admin_dir(self) -> Path:
        return self.base_dir / "admin"

    # -- scaffold --------------------------------------------------------------

    def scaffold(self) -> None:
        """Create the full directory structure and starter files."""
        for subdir in self._SUBDIRS:
            (self.base_dir / subdir).mkdir(parents=True, exist_ok=True)

        # tasks/_archive/
        (self.tasks_dir / "_archive").mkdir(exist_ok=True)

        # tasks/_index.md
        index_path = self.tasks_dir / "_index.md"
        if not index_path.exists():
            index_path.write_text("# Task Index\n\n_Auto-generated dashboard._\n", encoding="utf-8")

        # admin/audit.md
        audit_path = self.admin_dir / "audit.md"
        if not audit_path.exists():
            audit_path.write_text("# Audit Log\n\n", encoding="utf-8")

    # -- audit logging ---------------------------------------------------------

    def log_action(self, action: str, tool_name: str, details: str = "") -> None:
        """Append a timestamped entry to admin/audit.md."""
        self.admin_dir.mkdir(parents=True, exist_ok=True)
        audit_path = self.admin_dir / "audit.md"
        if not audit_path.exists():
            audit_path.write_text("# Audit Log\n\n", encoding="utf-8")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"- `{timestamp}` **{action}** via `{tool_name}`"
        if details:
            entry += f" — {details}"
        entry += "\n"

        with open(audit_path, "a", encoding="utf-8") as f:
            f.write(entry)

    # -- cross-section search --------------------------------------------------

    def search_all(self, query: str, limit: int = 10) -> list[dict]:
        """Keyword search across all vault markdown files.

        Returns a list of dicts with keys: section, path, snippets.
        """
        if not query:
            return []

        query_lower = query.lower()
        results: list[dict] = []

        for subdir in self._SUBDIRS:
            section_dir = self.base_dir / subdir
            if not section_dir.is_dir():
                continue
            for root, _dirs, files in os.walk(section_dir):
                for fname in files:
                    if not fname.endswith(".md"):
                        continue
                    fpath = Path(root) / fname
                    try:
                        content = fpath.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        continue
                    if query_lower not in content.lower():
                        continue

                    snippets = _extract_snippets(content, query_lower)
                    results.append(
                        {
                            "section": subdir,
                            "path": str(fpath.relative_to(self.base_dir)),
                            "snippets": snippets,
                        }
                    )
                    if len(results) >= limit:
                        return results

        return results


def _extract_snippets(content: str, query_lower: str, max_snippets: int = 3) -> list[str]:
    """Extract lines containing the query with one line of context each side."""
    lines = content.split("\n")
    snippets: list[str] = []
    for i, line in enumerate(lines):
        if query_lower in line.lower():
            start = max(0, i - 1)
            end = min(len(lines), i + 2)
            snippets.append("\n".join(lines[start:end]).strip())
            if len(snippets) >= max_snippets:
                break
    return snippets
