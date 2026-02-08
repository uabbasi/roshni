"""Tests for Notion tool behavior."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
import yaml

from roshni.agent.tools.notion_tool import create_notion_tools
from roshni.core.config import Config
from roshni.core.secrets import SecretsManager, YamlFileProvider


def _make_secrets(tmp_dir: str, payload: dict) -> SecretsManager:
    path = os.path.join(tmp_dir, "secrets.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(payload, f)
    return SecretsManager(providers=[YamlFileProvider(path)])


@pytest.fixture
def cfg(tmp_dir):
    return Config(
        data_dir=tmp_dir,
        defaults={
            "integrations": {
                "notion": {
                    "enabled": True,
                    "database_id": "db123",
                    "title_property": "Name",
                }
            }
        },
    )


def test_notion_tools_require_token_and_database(tmp_dir):
    cfg = Config(data_dir=tmp_dir, defaults={"integrations": {"notion": {"enabled": True, "database_id": ""}}})
    secrets = _make_secrets(tmp_dir, {})
    with pytest.raises(ValueError, match=r"notion\.token"):
        create_notion_tools(cfg, secrets)


@patch("roshni.agent.tools.notion_tool.NotionClient")
def test_notion_tools_list_create_update(mock_client_cls, cfg, tmp_dir):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_client.resolve_title_property.return_value = "Name"
    mock_client.extract_page_title.return_value = "Project Plan"
    mock_client.list_pages.return_value = [
        {
            "id": "p1",
            "last_edited_time": "2026-02-01T10:00:00.000Z",
            "url": "https://notion.so/p1",
            "properties": {},
        }
    ]
    mock_client.create_page.return_value = {"id": "p2", "properties": {}}
    mock_client.update_page.return_value = {"id": "p2", "properties": {}}

    secrets = _make_secrets(tmp_dir, {"notion": {"token": "secret"}})
    tools = create_notion_tools(cfg, secrets)
    names = {t.name for t in tools}

    assert "notion_list_pages" in names
    assert "notion_search_pages" in names
    assert "notion_create_page" in names
    assert "notion_update_page" in names
    assert "notion_append_to_page" in names

    list_tool = next(t for t in tools if t.name == "notion_list_pages")
    out = list_tool.execute({"limit": 5})
    assert "Notion pages:" in out

    create_tool = next(t for t in tools if t.name == "notion_create_page")
    result = create_tool.execute({"title": "Project Plan", "tags": "work,priority"})
    assert "Page create done" in result
    mock_client.create_page.assert_called_once()
    assert mock_client.create_page.call_args.kwargs["tags"] == ["work", "priority"]

    update_tool = next(t for t in tools if t.name == "notion_update_page")
    _ = update_tool.execute({"page_id": "p2", "archived": "true"})
    mock_client.update_page.assert_called_once_with("p2", title=None, archived=True, tags=None, status=None)
