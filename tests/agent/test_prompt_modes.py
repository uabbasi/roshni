"""Tests for prompt mode variants (PromptMode enum and get_system_prompt mode param)."""

import os

import pytest

from roshni.agent.persona import PromptMode, get_system_prompt


@pytest.fixture
def persona_dir(tmp_dir):
    """Create a persona directory with all four markdown files."""
    d = os.path.join(tmp_dir, "persona")
    os.makedirs(d)

    with open(os.path.join(d, "IDENTITY.md"), "w") as f:
        f.write("You are Roshni, a personal AI assistant.\n\n## Agent Identities\n### cfo\nYou handle finances.\n")

    with open(os.path.join(d, "SOUL.md"), "w") as f:
        f.write("## Values\nBe helpful, honest, and harmless.\n")

    with open(os.path.join(d, "USER.md"), "w") as f:
        f.write("## User Profile\nThe user is a software engineer.\n")

    with open(os.path.join(d, "AGENTS.md"), "w") as f:
        f.write("## Operational Policies\nAlways ask before taking actions.\n")

    return d


class TestPromptModeEnum:
    def test_full_value(self):
        assert PromptMode.FULL.value == "full"

    def test_compact_value(self):
        assert PromptMode.COMPACT.value == "compact"

    def test_minimal_value(self):
        assert PromptMode.MINIMAL.value == "minimal"


class TestGetSystemPromptFull:
    def test_full_includes_all_sections(self, persona_dir):
        prompt = get_system_prompt(persona_dir, mode=PromptMode.FULL)
        assert "Roshni" in prompt
        assert "Values" in prompt or "helpful" in prompt
        assert "User Profile" in prompt or "software engineer" in prompt
        assert "Operational Policies" in prompt or "ask before" in prompt

    def test_default_is_full(self, persona_dir):
        default_prompt = get_system_prompt(persona_dir)
        full_prompt = get_system_prompt(persona_dir, mode=PromptMode.FULL)
        assert default_prompt == full_prompt


class TestGetSystemPromptCompact:
    def test_compact_includes_identity_and_user(self, persona_dir):
        prompt = get_system_prompt(persona_dir, mode=PromptMode.COMPACT)
        assert "Roshni" in prompt
        assert "software engineer" in prompt

    def test_compact_skips_soul(self, persona_dir):
        prompt = get_system_prompt(persona_dir, mode=PromptMode.COMPACT)
        assert "helpful, honest" not in prompt

    def test_compact_skips_agents(self, persona_dir):
        prompt = get_system_prompt(persona_dir, mode=PromptMode.COMPACT)
        assert "Operational Policies" not in prompt

    def test_compact_shorter_than_full(self, persona_dir):
        full = get_system_prompt(persona_dir, mode=PromptMode.FULL, include_timestamp=False)
        compact = get_system_prompt(persona_dir, mode=PromptMode.COMPACT, include_timestamp=False)
        assert len(compact) < len(full)


class TestGetSystemPromptMinimal:
    def test_minimal_includes_identity_preamble(self, persona_dir):
        prompt = get_system_prompt(persona_dir, mode=PromptMode.MINIMAL)
        assert "Roshni" in prompt

    def test_minimal_skips_soul(self, persona_dir):
        prompt = get_system_prompt(persona_dir, mode=PromptMode.MINIMAL)
        assert "helpful, honest" not in prompt

    def test_minimal_skips_user(self, persona_dir):
        prompt = get_system_prompt(persona_dir, mode=PromptMode.MINIMAL)
        assert "software engineer" not in prompt

    def test_minimal_skips_agents(self, persona_dir):
        prompt = get_system_prompt(persona_dir, mode=PromptMode.MINIMAL)
        assert "Operational Policies" not in prompt

    def test_minimal_shorter_than_compact(self, persona_dir):
        compact = get_system_prompt(persona_dir, mode=PromptMode.COMPACT, include_timestamp=False)
        minimal = get_system_prompt(persona_dir, mode=PromptMode.MINIMAL, include_timestamp=False)
        assert len(minimal) < len(compact)
