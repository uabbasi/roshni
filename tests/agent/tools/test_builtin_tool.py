"""Tests for built-in weather/web tools."""

from __future__ import annotations

import datetime
from unittest.mock import MagicMock, patch

from roshni.agent.tools.builtin_tool import _web_search, create_builtin_tools


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


def test_search_web_schema_has_auto_fetch():
    tools = {t.name: t for t in create_builtin_tools()}
    props = tools["search_web"].parameters["properties"]
    assert "auto_fetch" in props
    assert props["auto_fetch"]["type"] == "boolean"


def test_auto_year_appended_when_missing():
    """Query without a 4-digit year gets current year appended."""
    current_year = str(datetime.datetime.now(datetime.UTC).year)
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = []

    with patch("roshni.agent.tools.builtin_tool.DDGS", return_value=mock_ddgs_instance):
        _web_search("best python frameworks", auto_fetch=False)

    query_sent = mock_ddgs_instance.text.call_args[0][0]
    assert current_year in query_sent


def test_auto_year_not_appended_when_present():
    """Query that already contains a year should not get another year."""
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = []

    with patch("roshni.agent.tools.builtin_tool.DDGS", return_value=mock_ddgs_instance):
        _web_search("python trends 2025", auto_fetch=False)

    query_sent = mock_ddgs_instance.text.call_args[0][0]
    assert query_sent == "python trends 2025"


def test_search_uses_ddgs():
    """DDGS().text() is called and results are formatted."""
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = [
        {"title": "Result 1", "href": "https://example.com/1", "body": "Body 1"},
        {"title": "Result 2", "href": "https://example.com/2", "body": "Body 2"},
    ]

    with patch("roshni.agent.tools.builtin_tool.DDGS", return_value=mock_ddgs_instance):
        result = _web_search("test query 2025", limit=5, auto_fetch=False)

    assert "Result 1" in result
    assert "Result 2" in result
    assert "https://example.com/1" in result
    mock_ddgs_instance.text.assert_called_once()


def test_auto_fetch_appends_content():
    """When auto_fetch=True, first result URL content is appended."""
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = [
        {"title": "Page", "href": "https://example.com/page", "body": "Summary"},
    ]

    with (
        patch("roshni.agent.tools.builtin_tool.DDGS", return_value=mock_ddgs_instance),
        patch(
            "roshni.agent.tools.builtin_tool._fetch_webpage",
            return_value="Fetched: https://example.com/page\n\nPage content here",
        ),
    ):
        result = _web_search("test 2025", auto_fetch=True)

    assert "---" in result
    assert "Page content here" in result


def test_auto_fetch_silently_skips_on_failure():
    """When auto_fetch fails, it should not raise or add error text."""
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = [
        {"title": "Page", "href": "https://example.com/page", "body": "Summary"},
    ]

    with (
        patch("roshni.agent.tools.builtin_tool.DDGS", return_value=mock_ddgs_instance),
        patch("roshni.agent.tools.builtin_tool._fetch_webpage", side_effect=Exception("Connection refused")),
    ):
        result = _web_search("test 2025", auto_fetch=True)

    assert "Page" in result
    assert "Connection refused" not in result


def test_auto_fetch_disabled():
    """When auto_fetch=False, no fetch is attempted."""
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = [
        {"title": "Page", "href": "https://example.com/page", "body": "Summary"},
    ]

    with (
        patch("roshni.agent.tools.builtin_tool.DDGS", return_value=mock_ddgs_instance),
        patch("roshni.agent.tools.builtin_tool._fetch_webpage") as mock_fetch,
    ):
        result = _web_search("test 2025", auto_fetch=False)

    mock_fetch.assert_not_called()
    assert "---" not in result


def test_no_results_message():
    """When DDGS returns empty results, a no-results message is shown."""
    mock_ddgs_instance = MagicMock()
    mock_ddgs_instance.text.return_value = []

    with patch("roshni.agent.tools.builtin_tool.DDGS", return_value=mock_ddgs_instance):
        result = _web_search("obscure nonexistent thing 2025", auto_fetch=False)

    assert "No web results found" in result
