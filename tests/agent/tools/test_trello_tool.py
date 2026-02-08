"""Tests for Trello tool behavior."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
import yaml

from roshni.agent.tools.trello_tool import create_trello_tools
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
                "trello": {
                    "enabled": True,
                }
            }
        },
    )


def test_create_trello_tools_requires_credentials(cfg, tmp_dir):
    secrets = _make_secrets(tmp_dir, {})
    with pytest.raises(ValueError, match=r"trello\.api_key"):
        create_trello_tools(cfg, secrets)


@patch("roshni.agent.tools.trello_tool.TrelloClient")
def test_trello_tools_expose_crud_capabilities(mock_client_cls, cfg, tmp_dir):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_client.list_boards.return_value = [{"id": "b1", "name": "Ops", "closed": False}]
    mock_client.create_card.return_value = {"id": "c1", "name": "Fix bug"}

    secrets = _make_secrets(tmp_dir, {"trello": {"api_key": "key", "token": "tok"}})
    tools = create_trello_tools(cfg, secrets)
    names = {t.name for t in tools}

    assert "trello_list_boards" in names
    assert "trello_create_board" in names
    assert "trello_list_lists" in names
    assert "trello_create_list" in names
    assert "trello_list_cards" in names
    assert "trello_create_card" in names
    assert "trello_update_card" in names
    assert "trello_add_comment" in names
    assert "trello_create_label" in names

    list_tool = next(t for t in tools if t.name == "trello_list_boards")
    out = list_tool.execute({})
    assert "Boards:" in out
    assert "Ops" in out

    create_tool = next(t for t in tools if t.name == "trello_create_card")
    result = create_tool.execute(
        {
            "list_id": "l1",
            "name": "Fix bug",
            "label_ids": "lb1, lb2",
        }
    )
    assert "Card create done" in result
    mock_client.create_card.assert_called_once()
    kwargs = mock_client.create_card.call_args.kwargs
    assert kwargs["label_ids"] == ["lb1", "lb2"]


@patch("roshni.agent.tools.trello_tool.TrelloClient")
def test_disable_board_delete_removes_tool(mock_client_cls, tmp_dir):
    cfg = Config(
        data_dir=tmp_dir,
        defaults={
            "integrations": {
                "trello": {
                    "enabled": True,
                    "disable_board_delete": True,
                }
            }
        },
    )
    secrets = _make_secrets(tmp_dir, {"trello": {"api_key": "key", "token": "tok"}})

    tools = create_trello_tools(cfg, secrets)
    names = {t.name for t in tools}
    assert "trello_delete_board" not in names
