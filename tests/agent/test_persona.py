"""Tests for roshni.agent.persona."""

import pytest

from roshni.agent.persona import extract_section, get_system_prompt


@pytest.fixture
def config_dir(tmp_path):
    """Create a temporary config directory with sample persona files."""
    (tmp_path / "IDENTITY.md").write_text(
        "# Hakim\n\nYou are a helpful assistant.\n\n"
        "## Channel Overrides\n\n### telegram\nBe concise.\n\n### cli\nBe verbose.\n\n"
        "## Agent Identities\n\n### averroes\nYou are a journal specialist.\n\n"
        "### khaldun\nYou are a financial advisor.\n"
    )
    (tmp_path / "SOUL.md").write_text("Be kind and curious.")
    (tmp_path / "USER.md").write_text("The user is a software engineer.")
    (tmp_path / "AGENTS.md").write_text("Delegate health questions to Averroes.")
    return tmp_path


class TestExtractSection:
    def test_extract_h2(self):
        text = "## First\nContent A\n## Second\nContent B"
        assert extract_section(text, "First", level=2) == "Content A"

    def test_extract_h3(self):
        text = "### Alpha\nAlpha stuff\n### Beta\nBeta stuff"
        assert extract_section(text, "Alpha", level=3) == "Alpha stuff"

    def test_missing_section(self):
        text = "## First\nContent"
        assert extract_section(text, "Missing", level=2) == ""

    def test_empty_text(self):
        assert extract_section("", "Anything") == ""

    def test_case_insensitive(self):
        text = "## My Section\nContent here"
        assert extract_section(text, "my section", level=2) == "Content here"


class TestGetSystemPrompt:
    def test_basic_prompt(self, config_dir):
        prompt = get_system_prompt(config_dir, include_timestamp=False)
        assert "helpful assistant" in prompt
        assert "Be kind" in prompt
        assert "software engineer" in prompt
        assert "Delegate health" in prompt

    def test_channel_override(self, config_dir):
        prompt = get_system_prompt(config_dir, channel="telegram", include_timestamp=False)
        assert "Be concise" in prompt

    def test_agent_identity(self, config_dir):
        prompt = get_system_prompt(config_dir, agent_name="averroes", include_timestamp=False)
        assert "journal specialist" in prompt

    def test_exclude_sections(self, config_dir):
        prompt = get_system_prompt(
            config_dir,
            include_soul=False,
            include_user=False,
            include_agents=False,
            include_timestamp=False,
        )
        assert "helpful assistant" in prompt
        assert "Be kind" not in prompt
        assert "software engineer" not in prompt

    def test_timestamp_header(self, config_dir):
        prompt = get_system_prompt(config_dir, include_timestamp=True)
        assert "CURRENT DATE:" in prompt
        assert "CURRENT TIME:" in prompt

    def test_extra_sections(self, config_dir):
        prompt = get_system_prompt(
            config_dir,
            extra_sections=["MEMORY: User prefers dark mode"],
            include_timestamp=False,
        )
        assert "dark mode" in prompt

    def test_missing_config_dir(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        prompt = get_system_prompt(empty_dir, include_timestamp=False)
        assert prompt == ""

    def test_custom_filenames(self, tmp_path):
        (tmp_path / "persona.md").write_text("Custom persona content")
        prompt = get_system_prompt(
            tmp_path,
            identity_file="persona.md",
            include_soul=False,
            include_user=False,
            include_agents=False,
            include_timestamp=False,
        )
        assert "Custom persona" in prompt
