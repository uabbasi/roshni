"""Tests for built-in weather/web tools."""

from __future__ import annotations

from roshni.agent.tools.builtin_tool import create_builtin_tools


def test_builtin_tools_exist():
    tools = {t.name: t for t in create_builtin_tools()}
    assert "get_weather" in tools
    assert "search_web" in tools
    assert "fetch_webpage" in tools


def test_builtin_input_validation_without_network():
    tools = {t.name: t for t in create_builtin_tools()}

    weather = tools["get_weather"].execute({"location": ""})
    assert "Please provide a location" in weather

    search = tools["search_web"].execute({"query": ""})
    assert "Please provide a search query" in search

    fetch = tools["fetch_webpage"].execute({"url": "example.com"})
    assert "URL must start with" in fetch
