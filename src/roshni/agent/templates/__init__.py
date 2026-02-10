"""Persona template loading utilities."""

from importlib import resources


def _read_template(filename: str) -> str:
    """Read a template file from this package."""
    ref = resources.files(__package__) / filename
    return ref.read_text(encoding="utf-8")


def get_identity_template(tone: str) -> str:
    """Get identity template for a given tone (friendly, professional, warm, witty)."""
    return _read_template(f"identity_{tone}.md")


def get_soul_template() -> str:
    return _read_template("soul_template.md")


def get_user_template() -> str:
    return _read_template("user_template.md")


def get_agents_template() -> str:
    """Get the agents operational policies template."""
    return _read_template("agents_template.md")


def get_tools_template() -> str:
    """Get the tools environment context template."""
    return _read_template("tools_template.md")


AVAILABLE_TONES = ["friendly", "professional", "warm", "witty"]
