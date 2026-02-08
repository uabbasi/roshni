"""Obsidian tool — keyword search over a vault of markdown files."""

from __future__ import annotations

import os
import re

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


def create_obsidian_tools(vault_path: str) -> list[ToolDefinition]:
    """Create vault search tool."""
    return [
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
        ),
    ]
