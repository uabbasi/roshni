"""Notes tool — save and recall quick notes."""

from __future__ import annotations

import os
from datetime import datetime

from roshni.agent.tools import ToolDefinition


def _save_note(content: str, title: str = "", notes_dir: str = "") -> str:
    """Save a note to a markdown file."""
    os.makedirs(notes_dir, exist_ok=True)
    now = datetime.now()
    slug = title.lower().replace(" ", "-")[:40] if title else now.strftime("%H%M%S")
    filename = f"{now.strftime('%Y-%m-%d')}_{slug}.md"
    filepath = os.path.join(notes_dir, filename)

    header = f"# {title}\n\n" if title else ""
    timestamp = f"*Saved: {now.strftime('%Y-%m-%d %I:%M %p')}*\n\n"
    with open(filepath, "w") as f:
        f.write(header + timestamp + content + "\n")

    return f"Note saved: {filename}"


def _recall_notes(query: str = "", notes_dir: str = "", limit: int = 5) -> str:
    """Search notes by keyword. Returns matching snippets."""
    if not os.path.isdir(notes_dir):
        return "No notes found."

    files = sorted(
        [f for f in os.listdir(notes_dir) if f.endswith(".md")],
        reverse=True,
    )

    if not files:
        return "No notes found."

    if not query:
        # Return most recent notes
        results = []
        for f in files[:limit]:
            with open(os.path.join(notes_dir, f)) as fh:
                content = fh.read().strip()
                preview = content[:200] + "..." if len(content) > 200 else content
                results.append(f"**{f}**\n{preview}")
        return "\n\n---\n\n".join(results)

    # Keyword search
    query_lower = query.lower()
    results = []
    for f in files:
        filepath = os.path.join(notes_dir, f)
        with open(filepath) as fh:
            content = fh.read()
        if query_lower in content.lower():
            # Find matching lines with context
            lines = content.split("\n")
            snippets = []
            for i, line in enumerate(lines):
                if query_lower in line.lower():
                    start = max(0, i - 1)
                    end = min(len(lines), i + 2)
                    snippets.append("\n".join(lines[start:end]))
            preview = "\n...\n".join(snippets[:3])
            results.append(f"**{f}**\n{preview}")
            if len(results) >= limit:
                break

    return "\n\n---\n\n".join(results) if results else f"No notes matching '{query}'."


def create_notes_tools(notes_dir: str) -> list[ToolDefinition]:
    """Create save_note and recall_notes tools."""
    return [
        ToolDefinition(
            name="save_note",
            description="Save a quick note. Use this when the user wants to remember something.",
            parameters={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The note content to save",
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional short title for the note",
                    },
                },
                "required": ["content"],
            },
            function=lambda content, title="": _save_note(content, title, notes_dir),
        ),
        ToolDefinition(
            name="recall_notes",
            description="Search through saved notes by keyword, or list recent notes if no query given.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword (empty to list recent notes)",
                    },
                },
                "required": [],
            },
            function=lambda query="": _recall_notes(query, notes_dir),
        ),
    ]
