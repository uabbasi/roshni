"""
File I/O utilities: safe write, markdown processing, frontmatter handling.

All functions operate on explicit paths — no implicit directory lookups.
"""

from __future__ import annotations

import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import yaml
from loguru import logger


def safe_write(filepath: str, content: str, mode: str = "w", encoding: str = "utf-8") -> None:
    """Write content to a file, creating parent directories as needed."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, mode, encoding=encoding) as f:
        f.write(content)


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """
    Parse YAML frontmatter from markdown content.

    Returns:
        (frontmatter_dict, content_without_frontmatter).
        If no frontmatter found, returns ({}, original_content).
    """
    content = content.strip()
    if not content.startswith("---"):
        return {}, content

    try:
        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}, content

        yaml_content = parts[1].strip()
        remaining = parts[2].lstrip()

        if yaml_content:
            yaml_content = yaml_content.replace("\t", "    ")
            # Handle Obsidian-style #tags in YAML lists
            yaml_content = re.sub(r"(^\s*-\s+)(#.*)$", r"\1'\2'", yaml_content, flags=re.MULTILINE)
            frontmatter = yaml.safe_load(yaml_content) or {}
        else:
            frontmatter = {}

        return frontmatter, remaining
    except (yaml.YAMLError, Exception) as e:
        logger.warning(f"Failed to parse markdown frontmatter: {e}")
        return {}, content


def update_frontmatter(content: str, updates: dict) -> str:
    """Update or add YAML frontmatter to markdown content."""
    frontmatter, body = parse_frontmatter(content)

    for key, value in updates.items():
        if key in frontmatter and isinstance(frontmatter[key], list) and isinstance(value, list):
            combined = frontmatter[key] + [item for item in value if item not in frontmatter[key]]
            frontmatter[key] = combined
        elif key in frontmatter and isinstance(frontmatter[key], list) and not isinstance(value, list):
            if value not in frontmatter[key]:
                frontmatter[key].append(value)
        else:
            frontmatter[key] = value

    if frontmatter:
        yaml_content = yaml.dump(frontmatter, default_flow_style=False, sort_keys=False)
        return f"---\n{yaml_content}---\n\n{body}"
    return body


def extract_markdown_sections(content: str, section_level: int = 2) -> dict[str, str]:
    """
    Extract sections from markdown content by heading level.

    Returns:
        Dict mapping section titles to their content.
    """
    content = re.sub(r"^---.*?---\s*", "", content, flags=re.DOTALL)
    content = content.strip()
    content = re.sub(r"\r\n", "\n", content)

    heading_marker = "#" * section_level
    pattern = rf"\n(?={heading_marker} )"
    sections = {}

    for block in re.split(pattern, content):
        if block.startswith(f"{heading_marker} "):
            lines = block.split("\n")
            title = lines[0].replace(f"{heading_marker} ", "").strip()
            body = "\n".join(lines[1:]).strip()
            if title and body:
                sections[title] = body
    return sections


def parse_date_heading(content: str) -> datetime.date | None:
    """Parse date from markdown frontmatter or heading."""
    for pattern in [r"date:\s*(\d{4}-\d{2}-\d{2})", r"# (\d{4}-\d{2}-\d{2})"]:
        match = re.search(pattern, content)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y-%m-%d").date()
            except ValueError:
                continue
    return None


def backup_file(file_path: str, backup_dir: str | None = None) -> str | None:
    """Create a timestamped backup of a file. Returns backup path or None."""
    src = Path(file_path)
    if not src.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"{src.name}.backup.{timestamp}"

    if backup_dir:
        backup_path = Path(backup_dir) / backup_name
        backup_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        backup_path = src.parent / backup_name

    try:
        shutil.copy2(str(src), str(backup_path))
        return str(backup_path)
    except Exception:
        return None


def safe_write_with_backup(filepath: str, content: str, mode: str = "w", encoding: str = "utf-8") -> str | None:
    """Write to file with automatic backup of existing content."""
    backup_path = backup_file(filepath)
    safe_write(filepath, content, mode, encoding)
    return backup_path


def safe_move(src_path: str, dest_dir: str, create_dirs: bool = True) -> str:
    """Move a file to a destination directory, optionally creating it."""
    src = Path(src_path)
    dest = Path(dest_dir)

    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src_path}")
    if create_dirs:
        dest.mkdir(parents=True, exist_ok=True)
    elif not dest.exists():
        raise FileNotFoundError(f"Destination directory not found: {dest_dir}")

    dest_path = dest / src.name
    shutil.move(str(src), str(dest_path))
    return str(dest_path)
