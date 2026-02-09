"""Tests for Trello tool behavior."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
import yaml

from roshni.agent.tools.trello_tool import (
    _enrich_card_urgency,
    _find_list_by_name,
    _fmt_card,
    _format_due_display,
    create_trello_tools,
)
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


# --- Urgency calculation tests ---


class TestEnrichCardUrgency:
    def test_overdue_card(self):
        now = datetime(2025, 6, 15, 12, 0, tzinfo=UTC)
        card = {"name": "Old task", "due": "2025-06-12T12:00:00Z"}
        result = _enrich_card_urgency(card, now)
        assert result["is_overdue"] is True
        assert result["overdue_days"] == 3
        assert result["days_until_due"] < 0

    def test_due_today(self):
        now = datetime(2025, 6, 15, 8, 0, tzinfo=UTC)
        card = {"name": "Today task", "due": "2025-06-15T20:00:00Z"}
        result = _enrich_card_urgency(card, now)
        assert result["is_overdue"] is False
        assert 0 < result["days_until_due"] < 1

    def test_due_tomorrow(self):
        now = datetime(2025, 6, 15, 12, 0, tzinfo=UTC)
        card = {"name": "Tomorrow task", "due": "2025-06-16T18:00:00Z"}
        result = _enrich_card_urgency(card, now)
        assert result["is_overdue"] is False
        assert 1 < result["days_until_due"] < 2

    def test_due_in_future(self):
        now = datetime(2025, 6, 15, 12, 0, tzinfo=UTC)
        card = {"name": "Future task", "due": "2025-06-20T12:00:00Z"}
        result = _enrich_card_urgency(card, now)
        assert result["is_overdue"] is False
        assert result["days_until_due"] == pytest.approx(5.0)

    def test_no_due_date(self):
        card = {"name": "No due"}
        result = _enrich_card_urgency(card)
        assert "days_until_due" not in result
        assert "is_overdue" not in result

    def test_invalid_due_date(self):
        card = {"name": "Bad due", "due": "not-a-date"}
        result = _enrich_card_urgency(card)
        assert "days_until_due" not in result


class TestFormatDueDisplay:
    def test_overdue(self):
        card = {"days_until_due": -3.5, "is_overdue": True, "overdue_days": 3}
        assert _format_due_display(card) == "OVERDUE (3d ago)"

    def test_today(self):
        card = {"days_until_due": 0.5, "is_overdue": False}
        assert _format_due_display(card) == "TODAY"

    def test_tomorrow(self):
        card = {"days_until_due": 1.2, "is_overdue": False}
        assert _format_due_display(card) == "tomorrow"

    def test_future(self):
        card = {"days_until_due": 5.7, "is_overdue": False}
        assert _format_due_display(card) == "in 5d"

    def test_no_urgency_data(self):
        card = {"name": "plain"}
        assert _format_due_display(card) == ""


# --- trello_today briefing test ---


@patch("roshni.agent.tools.trello_tool.TrelloClient")
def test_trello_today_produces_briefing(mock_client_cls, cfg, tmp_dir):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    now = datetime.now(UTC)
    overdue_due = (now - timedelta(days=2)).isoformat()
    today_due = (now + timedelta(hours=6)).isoformat()

    mock_client.list_lists.return_value = [
        {"id": "l1", "name": "Today", "closed": False},
        {"id": "l2", "name": "This Week", "closed": False},
    ]
    mock_client.list_cards.side_effect = [
        [
            {"id": "c1", "name": "Overdue task", "due": overdue_due},
            {"id": "c2", "name": "Today task", "due": today_due},
        ],
        [
            {"id": "c3", "name": "Weekly task", "due": None},
        ],
    ]

    secrets = _make_secrets(tmp_dir, {"trello": {"api_key": "key", "token": "tok"}})
    tools = create_trello_tools(cfg, secrets)
    today_tool = next(t for t in tools if t.name == "trello_today")

    # Need to pass board_id since no default_board_id in config
    result = today_tool.execute({"board_id": "b1"})

    assert "## Today" in result
    assert "## This Week" in result
    assert "Overdue task" in result
    assert "Today task" in result
    assert "Weekly task" in result
    # Overdue should appear before today task
    assert result.index("Overdue task") < result.index("Today task")


# --- find_list_by_name tests ---


class TestFindListByName:
    def test_exact_match(self):
        client = MagicMock()
        client.list_lists.return_value = [
            {"id": "l1", "name": "Backlog"},
            {"id": "l2", "name": "Today"},
            {"id": "l3", "name": "Done"},
        ]
        result = _find_list_by_name(client, "b1", "Today")
        assert result is not None
        assert result["id"] == "l2"

    def test_case_insensitive_exact(self):
        client = MagicMock()
        client.list_lists.return_value = [
            {"id": "l1", "name": "TODAY"},
        ]
        result = _find_list_by_name(client, "b1", "today")
        assert result is not None
        assert result["id"] == "l1"

    def test_partial_match(self):
        client = MagicMock()
        client.list_lists.return_value = [
            {"id": "l1", "name": "Backlog Items"},
            {"id": "l2", "name": "This Week Tasks"},
        ]
        result = _find_list_by_name(client, "b1", "Week")
        assert result is not None
        assert result["id"] == "l2"

    def test_no_match(self):
        client = MagicMock()
        client.list_lists.return_value = [
            {"id": "l1", "name": "Backlog"},
        ]
        result = _find_list_by_name(client, "b1", "Nonexistent")
        assert result is None


# --- Checklist rendering test ---


class TestChecklistRendering:
    def test_fmt_card_with_checklists(self):
        card = {
            "id": "c1",
            "name": "Shopping card",
            "idList": "l1",
            "idBoard": "b1",
            "due": None,
            "url": "https://trello.com/c/abc",
            "closed": False,
            "checklists": [
                {
                    "name": "Shopping",
                    "checkItems": [
                        {"name": "Milk", "state": "complete"},
                        {"name": "Eggs", "state": "incomplete"},
                        {"name": "Bread", "state": "complete"},
                        {"name": "Butter", "state": "incomplete"},
                        {"name": "Cheese", "state": "incomplete"},
                    ],
                }
            ],
        }
        result = _fmt_card(card)
        assert "checklist 'Shopping': 2/5 complete" in result
        assert "[x] Milk" in result
        assert "[ ] Eggs" in result
        assert "[x] Bread" in result

    def test_fmt_card_no_checklists(self):
        card = {
            "id": "c1",
            "name": "Simple card",
            "idList": "l1",
            "idBoard": "b1",
            "due": None,
            "url": "https://trello.com/c/abc",
            "closed": False,
        }
        result = _fmt_card(card)
        assert "checklist" not in result

    def test_fmt_card_empty_checklist(self):
        card = {
            "id": "c1",
            "name": "Empty checklist card",
            "idList": "l1",
            "idBoard": "b1",
            "due": None,
            "url": "https://trello.com/c/abc",
            "closed": False,
            "checklists": [{"name": "Todo", "checkItems": []}],
        }
        result = _fmt_card(card)
        assert "checklist 'Todo': 0/0 complete" in result
