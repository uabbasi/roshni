"""Tests for the notes tool."""

import os

import pytest

from roshni.agent.tools.notes_tool import create_notes_tools


@pytest.fixture
def notes_dir(tmp_dir):
    """Create a temporary notes directory."""
    d = os.path.join(tmp_dir, "notes")
    os.makedirs(d)
    return d


@pytest.fixture
def notes_tools(notes_dir):
    """Create notes tools pointing at tmp dir."""
    return {t.name: t for t in create_notes_tools(notes_dir)}


class TestSaveNote:
    def test_save_basic(self, notes_tools, notes_dir):
        tool = notes_tools["save_note"]
        result = tool.execute({"content": "Buy milk"})
        assert "Note saved" in result
        # File should exist
        files = os.listdir(notes_dir)
        assert len(files) == 1
        assert files[0].endswith(".md")

    def test_save_with_title(self, notes_tools, notes_dir):
        tool = notes_tools["save_note"]
        result = tool.execute({"content": "Pick up at 3pm", "title": "School pickup"})
        assert "Note saved" in result
        files = os.listdir(notes_dir)
        assert len(files) == 1
        content = open(os.path.join(notes_dir, files[0])).read()
        assert "School pickup" in content
        assert "Pick up at 3pm" in content

    def test_save_creates_dir(self, tmp_dir):
        notes_dir = os.path.join(tmp_dir, "nonexistent", "notes")
        tools = {t.name: t for t in create_notes_tools(notes_dir)}
        result = tools["save_note"].execute({"content": "test"})
        assert "Note saved" in result
        assert os.path.isdir(notes_dir)


class TestRecallNotes:
    def test_recall_empty(self, notes_tools):
        tool = notes_tools["recall_notes"]
        result = tool.execute({"query": ""})
        assert "No notes found" in result

    def test_recall_recent(self, notes_tools, notes_dir):
        # Save a note first
        notes_tools["save_note"].execute({"content": "Test note content"})
        result = notes_tools["recall_notes"].execute({"query": ""})
        assert "Test note content" in result

    def test_recall_by_keyword(self, notes_tools, notes_dir):
        notes_tools["save_note"].execute({"content": "Meeting with Alice at noon", "title": "Meeting"})
        notes_tools["save_note"].execute({"content": "Buy groceries: eggs, bread", "title": "Shopping"})

        result = notes_tools["recall_notes"].execute({"query": "Alice"})
        assert "Alice" in result
        assert "groceries" not in result

    def test_recall_no_match(self, notes_tools, notes_dir):
        notes_tools["save_note"].execute({"content": "Something unrelated"})
        result = notes_tools["recall_notes"].execute({"query": "zzzznotfound"})
        assert "No notes matching" in result


class TestToolDefinition:
    def test_schema(self, notes_tools):
        tool = notes_tools["save_note"]
        schema = tool.to_litellm_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "save_note"
        assert "content" in schema["function"]["parameters"]["properties"]

    def test_execute_with_json_string(self, notes_tools, notes_dir):
        import json

        tool = notes_tools["save_note"]
        result = tool.execute(json.dumps({"content": "from json string"}))
        assert "Note saved" in result
