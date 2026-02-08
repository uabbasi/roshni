"""Tests for the Obsidian vault search tool."""

import os

import pytest

from roshni.agent.tools.obsidian_tool import create_obsidian_tools


@pytest.fixture
def vault_dir(tmp_dir):
    """Create a temporary Obsidian vault with some files."""
    vault = os.path.join(tmp_dir, "vault")
    os.makedirs(vault)

    with open(os.path.join(vault, "meeting-notes.md"), "w") as f:
        f.write(
            "# Meeting Notes\n\nDiscussed the new API design with Sarah.\nAction items: finalize schema by Friday.\n"
        )

    with open(os.path.join(vault, "recipes.md"), "w") as f:
        f.write("# Recipes\n\n## Pasta Carbonara\nEggs, cheese, bacon, pasta.\nCook pasta al dente.\n")

    subfolder = os.path.join(vault, "journal")
    os.makedirs(subfolder)
    with open(os.path.join(subfolder, "2024-01-15.md"), "w") as f:
        f.write("# January 15\n\nGreat day. Had coffee with Alice.\nFeeling productive.\n")

    hidden = os.path.join(vault, ".obsidian")
    os.makedirs(hidden)
    with open(os.path.join(hidden, "config.md"), "w") as f:
        f.write("# Config\nThis should not appear in search results with API keyword.\n")

    return vault


@pytest.fixture
def search_tool(vault_dir):
    tools = create_obsidian_tools(vault_dir)
    return next(t for t in tools if t.name == "search_vault")


class TestSearchVault:
    def test_search_finds_match(self, search_tool):
        result = search_tool.execute({"query": "Sarah"})
        assert "Sarah" in result
        assert "meeting-notes.md" in result

    def test_search_case_insensitive(self, search_tool):
        result = search_tool.execute({"query": "sarah"})
        assert "Sarah" in result

    def test_search_no_match(self, search_tool):
        result = search_tool.execute({"query": "zzzznotfound"})
        assert "No notes matching" in result

    def test_search_subfolder(self, search_tool):
        result = search_tool.execute({"query": "Alice"})
        assert "Alice" in result
        assert "2024-01-15" in result

    def test_search_skips_hidden_dirs(self, search_tool):
        result = search_tool.execute({"query": "Config"})
        assert ".obsidian" not in result

    def test_search_empty_query(self, search_tool):
        result = search_tool.execute({"query": ""})
        assert "provide a search query" in result.lower()

    def test_schema(self, search_tool):
        schema = search_tool.to_litellm_schema()
        assert schema["function"]["name"] == "search_vault"
        assert "query" in schema["function"]["parameters"]["properties"]

    def test_nonexistent_vault(self, tmp_dir):
        tools = create_obsidian_tools(os.path.join(tmp_dir, "nonexistent"))
        result = tools[0].execute({"query": "test"})
        assert "not found" in result.lower()
