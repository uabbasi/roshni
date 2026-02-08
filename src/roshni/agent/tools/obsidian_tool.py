"""Obsidian tool — keyword search and write over a vault of markdown files."""

from __future__ import annotations

import os
import re
from pathlib import Path

from roshni.agent.permissions import PermissionTier, filter_tools_by_tier
from roshni.agent.tools import ToolDefinition


def _search_vault(query: str, vault_path: str, limit: int = 5) -> str:
    """Search .md files in the vault for a keyword query."""
    if not os.path.isdir(vault_path):
        return f"Vault not found: {vault_path}"

    if not query.strip():
        return "Please provide a search query."

    # Walk the vault, skip hidden dirs
    matches: list[tuple[str, list[str]]] = []
    query_lower = query.lower()
    pattern = re.compile(re.escape(query), re.IGNORECASE)

    for root, dirs, files in os.walk(vault_path):
        # Skip hidden directories and .obsidian
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if not f.endswith(".md"):
                continue
            filepath = os.path.join(root, f)
            try:
                with open(filepath, encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
            except OSError:
                continue

            if query_lower not in content.lower():
                continue

            # Extract matching lines with context
            lines = content.split("\n")
            snippets: list[str] = []
            for i, line in enumerate(lines):
                if pattern.search(line):
                    start = max(0, i - 1)
                    end = min(len(lines), i + 2)
                    snippet = "\n".join(lines[start:end]).strip()
                    if snippet:
                        snippets.append(snippet)
                    if len(snippets) >= 3:
                        break

            rel_path = os.path.relpath(filepath, vault_path)
            matches.append((rel_path, snippets))

            if len(matches) >= limit:
                break
        if len(matches) >= limit:
            break

    if not matches:
        return f"No notes matching '{query}' in your vault."

    results: list[str] = []
    for path, snippets in matches:
        snippet_text = "\n...\n".join(snippets)
        results.append(f"**{path}**\n{snippet_text}")

    return "\n\n---\n\n".join(results)


def _resolve_write_path(vault_path: str, sandbox_dir: str, note_path: str) -> Path:
    """Resolve a note path, scoping writes to sandbox_dir when set."""
    vault = Path(vault_path).expanduser().resolve()
    if sandbox_dir:
        base = (vault / sandbox_dir).resolve()
    else:
        base = vault
    target = (base / note_path).resolve()
    if not str(target).startswith(str(base)):
        raise ValueError(f"Path escapes allowed directory: {note_path}")
    return target


def _create_vault_note(vault_path: str, sandbox_dir: str, path: str, content: str) -> str:
    """Create a new markdown note in the vault."""
    if not path.endswith(".md"):
        path += ".md"
    target = _resolve_write_path(vault_path, sandbox_dir, path)
    if target.exists():
        return f"Note already exists: {target.relative_to(Path(vault_path).expanduser().resolve())}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Note created: {target.relative_to(Path(vault_path).expanduser().resolve())}"


def _update_vault_note(vault_path: str, sandbox_dir: str, path: str, content: str, mode: str = "append") -> str:
    """Update an existing markdown note in the vault."""
    if not path.endswith(".md"):
        path += ".md"
    target = _resolve_write_path(vault_path, sandbox_dir, path)
    if not target.exists():
        return f"Note not found: {path}"
    if mode == "append":
        with open(target, "a", encoding="utf-8") as f:
            f.write("\n" + content)
    else:
        target.write_text(content, encoding="utf-8")
    return f"Note updated ({mode}): {target.relative_to(Path(vault_path).expanduser().resolve())}"


def create_obsidian_tools(
    vault_path: str,
    sandbox_dir: str = "",
    tier: PermissionTier = PermissionTier.INTERACT,
) -> list[ToolDefinition]:
    """Create vault search and write tools."""
    tools = [
        ToolDefinition(
            name="search_vault",
            description=(
                "Search the user's Obsidian vault (notes/documents) by keyword. "
                "Use this when the user asks about something they've written or wants "
                "to find information in their notes."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword or phrase",
                    },
                },
                "required": ["query"],
            },
            function=lambda query: _search_vault(query, vault_path),
            permission="read",
        ),
        ToolDefinition(
            name="create_vault_note",
            description="Create a new markdown note in the Obsidian vault.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Note path relative to vault (e.g. 'ideas/my-note.md')"},
                    "content": {"type": "string", "description": "Markdown content for the note"},
                },
                "required": ["path", "content"],
            },
            function=lambda path, content: _create_vault_note(vault_path, sandbox_dir, path, content),
            permission="write",
        ),
        ToolDefinition(
            name="update_vault_note",
            description="Update an existing markdown note in the Obsidian vault.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Note path relative to vault"},
                    "content": {"type": "string", "description": "Content to write or append"},
                    "mode": {
                        "type": "string",
                        "description": "Write mode: 'append' (default) or 'overwrite'",
                        "enum": ["append", "overwrite"],
                    },
                },
                "required": ["path", "content"],
            },
            function=lambda path, content, mode="append": _update_vault_note(
                vault_path, sandbox_dir, path, content, mode
            ),
            permission="write",
        ),
    ]
    return filter_tools_by_tier(tools, tier)
