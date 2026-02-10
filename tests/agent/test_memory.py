"""Tests for MemoryManager and save_memory tool."""

from __future__ import annotations

import os
import threading
from datetime import date

import pytest

from roshni.agent.memory import VALID_SECTIONS, MemoryManager, create_save_memory_tool


@pytest.fixture
def memory_path(tmp_dir):
    return os.path.join(tmp_dir, "MEMORY.md")


@pytest.fixture
def manager(memory_path):
    return MemoryManager(memory_path)


class TestMemoryManager:
    def test_creates_file_on_init(self, memory_path):
        assert not os.path.exists(memory_path)
        MemoryManager(memory_path)
        assert os.path.exists(memory_path)
        text = open(memory_path).read()
        assert "# Agent Memory" in text
        for section in VALID_SECTIONS:
            assert f"## {section}" in text

    def test_save_to_valid_section(self, manager, memory_path):
        result = manager.save("preferences", "Use dark mode")
        assert "Saved to preferences" in result
        text = open(memory_path).read()
        assert "- Use dark mode" in text

    def test_save_multiple_items(self, manager, memory_path):
        manager.save("preferences", "Use dark mode")
        manager.save("preferences", "Prefer Python over JS")
        text = open(memory_path).read()
        assert "- Use dark mode" in text
        assert "- Prefer Python over JS" in text

    def test_save_to_different_sections(self, manager, memory_path):
        manager.save("preferences", "Like cats")
        manager.save("decisions", "Use PostgreSQL")
        text = open(memory_path).read()
        assert "- Like cats" in text
        assert "- Use PostgreSQL" in text

    def test_save_invalid_section(self, manager):
        result = manager.save("nonexistent", "something")
        assert "Error" in result
        assert "nonexistent" in result

    def test_save_empty_content(self, manager):
        result = manager.save("preferences", "   ")
        assert "Error" in result

    def test_get_context_empty(self, memory_path):
        mgr = MemoryManager(memory_path)
        ctx = mgr.get_context()
        # Freshly created file has headers but no items — still returns content
        # because the file isn't truly empty (it has section headers)
        assert "[MEMORY]" in ctx or ctx == ""

    def test_get_context_with_data(self, manager):
        manager.save("preferences", "Dark mode always")
        ctx = manager.get_context()
        assert "[MEMORY]" in ctx
        assert "Dark mode always" in ctx
        assert "[/MEMORY]" in ctx

    def test_detect_trigger_positive(self, manager):
        assert manager.detect_trigger("always use dark mode")
        assert manager.detect_trigger("Never send emails on Sunday")
        assert manager.detect_trigger("Remember that I prefer Python")
        assert manager.detect_trigger("From now on, use metric units")
        assert manager.detect_trigger("I prefer tea over coffee")
        assert manager.detect_trigger("Don't forget to check the calendar")

    def test_detect_trigger_negative(self, manager):
        assert not manager.detect_trigger("What's the weather today?")
        assert not manager.detect_trigger("Tell me a joke")
        assert not manager.detect_trigger("How many days until Friday?")

    def test_thread_safety(self, manager, memory_path):
        """Multiple threads saving concurrently shouldn't corrupt the file."""
        errors = []

        def save_items(section, count):
            try:
                for i in range(count):
                    manager.save(section, f"item-{section}-{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=save_items, args=("preferences", 10)),
            threading.Thread(target=save_items, args=("decisions", 10)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        text = open(memory_path).read()
        # All 20 items should be present
        assert text.count("- item-") == 20

    def test_nonexistent_path_creates_parents(self, tmp_dir):
        deep_path = os.path.join(tmp_dir, "a", "b", "c", "MEMORY.md")
        MemoryManager(deep_path)
        assert os.path.exists(deep_path)


class TestDailyNotes:
    def test_append_daily_note(self, manager, memory_path):
        result = manager.append_daily_note("Had a good meeting")
        assert "Noted" in result
        daily_path = os.path.join(os.path.dirname(memory_path), "memory", date.today().isoformat() + ".md")
        assert os.path.exists(daily_path)
        text = open(daily_path).read()
        assert "- Had a good meeting" in text

    def test_append_multiple_notes(self, manager):
        manager.append_daily_note("First note")
        manager.append_daily_note("Second note")
        ctx = manager.get_daily_context()
        assert "First note" in ctx
        assert "Second note" in ctx

    def test_daily_note_specific_day(self, manager, memory_path):
        day = date(2025, 1, 15)
        manager.append_daily_note("Past note", day=day)
        daily_path = os.path.join(os.path.dirname(memory_path), "memory", "2025-01-15.md")
        assert os.path.exists(daily_path)

    def test_get_daily_context_empty(self, manager):
        ctx = manager.get_daily_context()
        assert ctx == ""

    def test_get_daily_context_with_data(self, manager):
        manager.append_daily_note("Something happened")
        ctx = manager.get_daily_context()
        assert "[DAILY NOTES]" in ctx
        assert "Something happened" in ctx
        assert "[/DAILY NOTES]" in ctx

    def test_daily_note_empty_content(self, manager):
        result = manager.append_daily_note("   ")
        assert "Error" in result

    def test_daily_note_creates_directory(self, tmp_dir):
        deep_path = os.path.join(tmp_dir, "deep", "MEMORY.md")
        mgr = MemoryManager(deep_path)
        mgr.append_daily_note("A note")
        daily_dir = os.path.join(tmp_dir, "deep", "memory")
        assert os.path.isdir(daily_dir)


class TestSaveMemoryTool:
    def test_tool_creation(self, manager):
        tool = create_save_memory_tool(manager)
        assert tool.name == "save_memory"
        assert tool.permission == "write"
        assert tool.requires_approval is False
        assert not tool.needs_approval()

    def test_tool_execution(self, manager, memory_path):
        tool = create_save_memory_tool(manager)
        result = tool.execute({"section": "preferences", "content": "Likes coffee"})
        assert "Saved" in result
        text = open(memory_path).read()
        assert "Likes coffee" in text

    def test_tool_schema(self, manager):
        tool = create_save_memory_tool(manager)
        schema = tool.to_litellm_schema()
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] == "save_memory"
        assert "section" in fn["parameters"]["properties"]
        assert "content" in fn["parameters"]["properties"]
        assert "enum" in fn["parameters"]["properties"]["section"]
