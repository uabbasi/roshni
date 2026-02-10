"""Persona loader — builds system prompts from markdown config files.

Reads identity markdown files from a config directory and assembles a
system prompt prefix that gives agents consistent personality, user context,
and operational boundaries.

The default file layout expected::

    config_dir/
        IDENTITY.md   — character definition, channel overrides, agent identities
        SOUL.md       — values and mission
        USER.md       — who the user is, preferences, goals
        AGENTS.md     — operational policies, permissions, delegation
        TOOLS.md      — environment-specific tool guidance and context

Each file is optional; missing files are silently skipped.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path

from loguru import logger


class PromptMode(Enum):
    """Controls how much of the persona is included in the system prompt."""

    FULL = "full"  # All sections
    COMPACT = "compact"  # Identity preamble + user + mode hints only
    MINIMAL = "minimal"  # Identity preamble only (1-2 sentences)


def _read_md(config_dir: Path, filename: str) -> str:
    """Read a markdown config file, returning empty string if missing."""
    path = config_dir / filename
    if not path.exists():
        logger.debug(f"Persona config not found: {path}")
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return ""


def extract_section(text: str, header: str, level: int = 2) -> str:
    """Extract a section from markdown text by header.

    Returns all content under the matching header until the next header
    of the same or higher level.

    Args:
        text: Full markdown text.
        header: Header text to match (case-insensitive).
        level: Header level (2 = ``##``, 3 = ``###``).

    Returns:
        Section content (without the header line itself), or empty string.
    """
    if not text or not header:
        return ""

    prefix = "#" * level + " "
    higher_prefixes = ["#" * i + " " for i in range(1, level + 1)]

    lines = text.split("\n")
    in_section = False
    section_lines: list[str] = []

    for line in lines:
        stripped = line.strip().lower()
        if stripped == f"{prefix.lower()}{header.lower()}":
            in_section = True
            continue
        if in_section:
            if any(line.startswith(p) for p in higher_prefixes):
                break
            section_lines.append(line)

    return "\n".join(section_lines).strip()


def _extract_preamble(identity_md: str) -> str:
    """Extract content before any ## heading (the character preamble)."""
    if not identity_md:
        return ""
    lines = identity_md.split("\n")
    preamble_lines: list[str] = []
    for line in lines:
        if line.startswith("## "):
            break
        preamble_lines.append(line)
    return "\n".join(preamble_lines).strip()


def get_system_prompt(
    config_dir: str | Path,
    *,
    agent_name: str | None = None,
    channel: str | None = None,
    include_soul: bool = True,
    include_user: bool = True,
    include_agents: bool = True,
    include_identity: bool = True,
    include_tools: bool = True,
    extra_sections: list[str] | None = None,
    include_timestamp: bool = True,
    identity_file: str = "IDENTITY.md",
    soul_file: str = "SOUL.md",
    user_file: str = "USER.md",
    agents_file: str = "AGENTS.md",
    tools_file: str = "TOOLS.md",
    mode: PromptMode = PromptMode.FULL,
) -> str:
    """Build a system prompt prefix from markdown config files.

    Combines identity files in order (voice first):
    1. identity preamble — who you are, how you sound
    2. soul — values and mission
    3. user — who the user is, preferences, goals
    4. agents — operational policies (permissions, delegation)
    5. tools — environment-specific tool guidance and context
    6. identity channel/agent overrides — presentation details
    7. any extra sections appended at the end

    The *mode* parameter controls how much is included:

    - ``FULL``: all sections (default, current behavior)
    - ``COMPACT``: identity preamble + user + tools only
      (skips SOUL.md, AGENTS.md, channel/agent overrides)
    - ``MINIMAL``: identity preamble only (1-2 sentences)

    Args:
        config_dir: Directory containing the markdown config files.
        agent_name: Agent identifier for agent-specific identity overrides.
        channel: Channel identifier for channel-specific overrides.
        include_soul: Include soul file content.
        include_user: Include user file content.
        include_agents: Include agents file content.
        include_identity: Include identity file content.
        include_tools: Include tools file content.
        extra_sections: Additional text blocks to append.
        include_timestamp: Prepend current date/time header.
        identity_file: Filename for the identity config.
        soul_file: Filename for the soul config.
        user_file: Filename for the user config.
        agents_file: Filename for the agents config.
        tools_file: Filename for the tools config.
        mode: Prompt mode controlling verbosity (FULL, COMPACT, MINIMAL).

    Returns:
        Combined system prompt prefix string.
    """
    config_path = Path(config_dir)
    sections: list[str] = []

    identity = _read_md(config_path, identity_file) if include_identity else ""

    if identity:
        preamble = _extract_preamble(identity)
        if preamble:
            sections.append(preamble)

    # MINIMAL mode: identity preamble only
    if mode == PromptMode.MINIMAL:
        if not sections:
            return ""
        prompt = "\n\n---\n\n".join(sections)
        if include_timestamp:
            now = datetime.now().astimezone()
            header = f"CURRENT DATE: {now.strftime('%Y-%m-%d')}\nCURRENT TIME: {now.strftime('%I:%M %p %Z')}\n"
            prompt = header + prompt
        return prompt

    # COMPACT mode: preamble + user + tools (skip soul, agents, channel/agent overrides)
    if mode == PromptMode.COMPACT:
        if include_user:
            user = _read_md(config_path, user_file)
            if user:
                sections.append(user)
        if include_tools:
            tools = _read_md(config_path, tools_file)
            if tools:
                sections.append(tools)
        if extra_sections:
            sections.extend(s for s in extra_sections if s)
        if not sections:
            return ""
        prompt = "\n\n---\n\n".join(sections)
        if include_timestamp:
            now = datetime.now().astimezone()
            header = f"CURRENT DATE: {now.strftime('%Y-%m-%d')}\nCURRENT TIME: {now.strftime('%I:%M %p %Z')}\n"
            prompt = header + prompt
        return prompt

    # FULL mode: everything
    if include_soul:
        soul = _read_md(config_path, soul_file)
        if soul:
            sections.append(soul)

    if include_user:
        user = _read_md(config_path, user_file)
        if user:
            sections.append(user)

    if include_agents:
        agents = _read_md(config_path, agents_file)
        if agents:
            sections.append(agents)

    if include_tools:
        tools = _read_md(config_path, tools_file)
        if tools:
            sections.append(tools)

    if identity:
        if channel:
            channel_block = extract_section(identity, channel, level=3)
            if channel_block:
                sections.append(f"## Channel: {channel}\n{channel_block}")
        if agent_name:
            # Look under ## Agent Identities → ### {agent_name}
            agent_identities = extract_section(identity, "Agent Identities", level=2)
            if agent_identities:
                agent_block = extract_section(
                    f"## Agent Identities\n{agent_identities}",
                    agent_name,
                    level=3,
                )
                if agent_block:
                    sections.append(f"## Your Identity\n{agent_block}")

    if extra_sections:
        sections.extend(s for s in extra_sections if s)

    if not sections:
        return ""

    prompt = "\n\n---\n\n".join(sections)

    if include_timestamp:
        now = datetime.now().astimezone()
        header = f"CURRENT DATE: {now.strftime('%Y-%m-%d')}\nCURRENT TIME: {now.strftime('%I:%M %p %Z')}\n"
        prompt = header + prompt

    return prompt
