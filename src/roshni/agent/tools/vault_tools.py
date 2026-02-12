"""Vault tools — people, projects, ideas, and cross-vault search."""

from __future__ import annotations

import os
import re
import threading
from datetime import datetime

from roshni.agent.permissions import PermissionTier, filter_tools_by_tier
from roshni.agent.tools import ToolDefinition
from roshni.agent.vault import VaultManager

_vault_write_lock = threading.Lock()


def _list_md_files(directory: str) -> str:
    """List .md files in a directory with updated timestamps from frontmatter."""
    if not os.path.isdir(directory):
        return "No entries found."
    files = sorted(f for f in os.listdir(directory) if f.endswith(".md"))
    if not files:
        return "No entries found."
    entries: list[str] = []
    for f in files:
        slug = f[:-3]
        path = os.path.join(directory, f)
        updated = _get_frontmatter_field(path, "updated")
        if updated:
            entries.append(f"- {slug} (updated {updated})")
        else:
            entries.append(f"- {slug}")
    return "\n".join(entries)


def _get_frontmatter_field(path: str, key: str) -> str | None:
    """Extract a single frontmatter field value from a .md file."""
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
    except (OSError, UnicodeDecodeError):
        return None
    m = re.search(rf'^{key}:\s*"?([^"\n]+)"?\s*$', content, re.MULTILINE)
    return m.group(1).strip() if m else None


def _update_frontmatter_field(content: str, key: str, value: str) -> str:
    """Update or insert a frontmatter field in markdown content."""
    pattern = rf'^({key}:\s*)"?[^"\n]+"?\s*$'
    replacement = rf'\g<1>"{value}"'
    updated, count = re.subn(pattern, replacement, content, count=1, flags=re.MULTILINE)
    if count > 0:
        return updated
    # Insert before closing --- (skip the opening --- on line 1)
    parts = content.split("---", 2)
    if len(parts) >= 3:
        return parts[0] + "---" + parts[1] + f'{key}: "{value}"\n---' + parts[2]
    return content


def _resolve_slug(directory: str, name: str) -> str | None:
    """Resolve a name to an existing file slug. Exact match first, then substring."""
    if not os.path.isdir(directory):
        return None
    slug = name.lower().replace(" ", "-")
    # Exact match
    if os.path.isfile(os.path.join(directory, f"{slug}.md")):
        return slug
    # Substring scan
    files = sorted(f for f in os.listdir(directory) if f.endswith(".md"))
    for f in files:
        if slug in f[:-3]:
            return f[:-3]
    return None


def _read_md_file(directory: str, name: str) -> str:
    """Read a .md file by exact name (without extension)."""
    path = os.path.join(directory, f"{name}.md")
    if not os.path.isfile(path):
        return f"Not found: {name}"
    with open(path, encoding="utf-8") as f:
        return f.read()


def _read_md_file_fuzzy(directory: str, name: str) -> str:
    """Read a .md file by name with partial matching."""
    slug = _resolve_slug(directory, name)
    if slug is None:
        return f"Not found: {name}"
    return _read_md_file(directory, slug)


def _save_md_file(directory: str, name: str, frontmatter: dict, body: str) -> str:
    """Save a .md file with YAML frontmatter."""
    os.makedirs(directory, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    if "created" not in frontmatter:
        frontmatter["created"] = now
    if "updated" not in frontmatter:
        frontmatter["updated"] = now

    lines = ["---"]
    for key, value in frontmatter.items():
        if isinstance(value, list):
            lines.append(f"{key}: {value}")
        else:
            lines.append(f'{key}: "{value}"')
    lines.append("---")
    lines.append(body)
    content = "\n".join(lines) + "\n"

    slug = name.lower().replace(" ", "-")
    path = os.path.join(directory, f"{slug}.md")
    with _vault_write_lock:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    return f"Saved: {slug}.md"


def _append_to_md_file(directory: str, slug: str, content: str) -> str:
    """Append a dated bullet to an existing .md file and update the 'updated' timestamp."""
    path = os.path.join(directory, f"{slug}.md")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    bullet = f"\n- {now}: {content}\n"

    with _vault_write_lock:
        with open(path, encoding="utf-8") as f:
            existing = f.read()
        existing = _update_frontmatter_field(existing, "updated", now)
        with open(path, "w", encoding="utf-8") as f:
            f.write(existing.rstrip("\n") + "\n" + bullet)
    return f"Appended to: {slug}.md"


def _search_md_files(directory: str, query: str, limit: int = 5) -> str:
    """Keyword search across .md files in a directory."""
    if not query:
        return "Please provide a search query."
    if not os.path.isdir(directory):
        return "No entries found."

    query_lower = query.lower()
    results: list[str] = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".md"):
            continue
        path = os.path.join(directory, fname)
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
        except (OSError, UnicodeDecodeError):
            continue
        if query_lower in content.lower():
            preview = content[:200] + "..." if len(content) > 200 else content
            results.append(f"**{fname[:-3]}**\n{preview}")
            if len(results) >= limit:
                break

    return "\n\n---\n\n".join(results) if results else f"No results for '{query}'."


def create_vault_tools(
    vault: VaultManager,
    tier: PermissionTier = PermissionTier.INTERACT,
) -> list[ToolDefinition]:
    """Create people/projects/ideas vault tools, filtered by permission tier."""
    people_dir = str(vault.people_dir)
    projects_dir = str(vault.projects_dir)
    ideas_dir = str(vault.ideas_dir)

    tools: list[ToolDefinition] = [
        # -- People --
        ToolDefinition(
            name="list_people",
            description="List all people in the vault.",
            parameters={"type": "object", "properties": {}, "required": []},
            function=lambda: _list_md_files(people_dir),
            permission="read",
        ),
        ToolDefinition(
            name="get_person",
            description="Read a person's profile from the vault (supports partial name matching).",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Person name or partial match"},
                },
                "required": ["name"],
            },
            function=lambda name: _read_md_file_fuzzy(people_dir, name),
            permission="read",
        ),
        ToolDefinition(
            name="save_person",
            description="Create or append to a person's profile with name, tags, and notes.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Person's full name"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags (e.g. colleague, friend)",
                    },
                    "last_contact": {"type": "string", "description": "Last contact date (YYYY-MM-DD)"},
                    "notes": {"type": "string", "description": "Notes about this person"},
                },
                "required": ["name", "notes"],
            },
            function=lambda name, notes, tags=None, last_contact="": _save_person(
                vault, name, notes, tags, last_contact
            ),
            permission="write",
        ),
        ToolDefinition(
            name="search_people",
            description="Search people profiles by keyword.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keyword"},
                },
                "required": ["query"],
            },
            function=lambda query: _search_md_files(people_dir, query),
            permission="read",
        ),
        # -- Projects --
        ToolDefinition(
            name="list_projects",
            description="List all projects in the vault.",
            parameters={"type": "object", "properties": {}, "required": []},
            function=lambda: _list_md_files(projects_dir),
            permission="read",
        ),
        ToolDefinition(
            name="get_project",
            description="Read a project's details from the vault (supports partial name matching).",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Project name or partial match"},
                },
                "required": ["name"],
            },
            function=lambda name: _read_md_file_fuzzy(projects_dir, name),
            permission="read",
        ),
        ToolDefinition(
            name="save_project",
            description="Create or append to a project with title, status, tags, and notes.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Project title"},
                    "status": {"type": "string", "description": "Status (active, paused, completed)"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for the project",
                    },
                    "notes": {"type": "string", "description": "Project notes and details"},
                },
                "required": ["title", "notes"],
            },
            function=lambda title, notes, status="active", tags=None: _save_project(vault, title, notes, status, tags),
            permission="write",
        ),
        # -- Ideas --
        ToolDefinition(
            name="list_ideas",
            description="List all ideas in the vault.",
            parameters={"type": "object", "properties": {}, "required": []},
            function=lambda: _list_md_files(ideas_dir),
            permission="read",
        ),
        ToolDefinition(
            name="get_idea",
            description="Read an idea's details from the vault (supports partial name matching).",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Idea name or partial match"},
                },
                "required": ["name"],
            },
            function=lambda name: _read_md_file_fuzzy(ideas_dir, name),
            permission="read",
        ),
        ToolDefinition(
            name="save_idea",
            description="Create or append to an idea with title, tags, and description.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Idea title"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for the idea",
                    },
                    "notes": {"type": "string", "description": "Idea description and details"},
                },
                "required": ["title", "notes"],
            },
            function=lambda title, notes, tags=None: _save_idea(vault, title, notes, tags),
            permission="write",
        ),
        ToolDefinition(
            name="search_ideas",
            description="Search ideas by keyword.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keyword"},
                },
                "required": ["query"],
            },
            function=lambda query: _search_md_files(ideas_dir, query),
            permission="read",
        ),
        # -- Cross-vault --
        ToolDefinition(
            name="search_vault_all",
            description="Search across all vault sections (people, projects, ideas, etc.) by keyword.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search keyword"},
                    "limit": {"type": "integer", "description": "Max results (default 10)"},
                },
                "required": ["query"],
            },
            function=lambda query, limit=10: _search_vault_all(vault, query, limit),
            permission="read",
        ),
    ]

    return filter_tools_by_tier(tools, tier)


# -- helper functions for write tools -----------------------------------------


def _save_person(vault: VaultManager, name: str, notes: str, tags: list[str] | None, last_contact: str) -> str:
    directory = str(vault.people_dir)
    slug = name.lower().replace(" ", "-")
    path = os.path.join(directory, f"{slug}.md")

    if os.path.isfile(path):
        result = _append_to_md_file(directory, slug, notes)
    else:
        frontmatter: dict = {"name": name}
        if tags:
            frontmatter["tags"] = tags
        if last_contact:
            frontmatter["last_contact"] = last_contact
        result = _save_md_file(directory, name, frontmatter, notes)
    vault.log_action("save", "save_person", f"name={name}")
    return result


def _save_project(vault: VaultManager, title: str, notes: str, status: str, tags: list[str] | None) -> str:
    directory = str(vault.projects_dir)
    slug = title.lower().replace(" ", "-")
    path = os.path.join(directory, f"{slug}.md")

    if os.path.isfile(path):
        result = _append_to_md_file(directory, slug, notes)
    else:
        frontmatter: dict = {"title": title, "status": status}
        if tags:
            frontmatter["tags"] = tags
        result = _save_md_file(directory, title, frontmatter, notes)
    vault.log_action("save", "save_project", f"title={title}")
    return result


def _save_idea(vault: VaultManager, title: str, notes: str, tags: list[str] | None) -> str:
    directory = str(vault.ideas_dir)
    slug = title.lower().replace(" ", "-")
    path = os.path.join(directory, f"{slug}.md")

    if os.path.isfile(path):
        result = _append_to_md_file(directory, slug, notes)
    else:
        frontmatter: dict = {
            "title": title,
            "status": "new",
        }
        if tags:
            frontmatter["tags"] = tags
        result = _save_md_file(directory, title, frontmatter, notes)
    vault.log_action("save", "save_idea", f"title={title}")
    return result


def _search_vault_all(vault: VaultManager, query: str, limit: int = 10) -> str:
    results = vault.search_all(query, limit=limit)
    if not results:
        return f"No results for '{query}'."
    parts: list[str] = []
    for r in results:
        snippets_text = "\n".join(r["snippets"][:2])
        parts.append(f"**[{r['section']}]** {r['path']}\n{snippets_text}")
    return "\n\n---\n\n".join(parts)
