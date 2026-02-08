"""Tests for roshni.integrations.trello."""

from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from roshni.core.exceptions import APIError
from roshni.integrations.trello import TrelloClient


class _DummyResp:
    def __init__(self, payload: object):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_init_requires_credentials():
    with pytest.raises(ValueError):
        TrelloClient(api_key="", token="token")
    with pytest.raises(ValueError):
        TrelloClient(api_key="key", token="")


@patch("roshni.integrations.trello.urllib.request.urlopen")
def test_list_boards(mock_urlopen):
    mock_urlopen.return_value = _DummyResp(
        [
            {"id": "b1", "name": "Board 1", "closed": False},
        ]
    )

    client = TrelloClient(api_key="k", token="t")
    boards = client.list_boards()

    assert len(boards) == 1
    req = mock_urlopen.call_args[0][0]
    assert req.get_method() == "GET"
    assert "/members/me/boards" in req.full_url
    assert "key=k" in req.full_url
    assert "token=t" in req.full_url


@patch("roshni.integrations.trello.urllib.request.urlopen")
def test_create_card_includes_labels(mock_urlopen):
    mock_urlopen.return_value = _DummyResp({"id": "c1", "name": "Test card"})

    client = TrelloClient(api_key="k", token="t")
    card = client.create_card(
        list_id="list123",
        name="Test card",
        desc="desc",
        due="2026-02-20",
        label_ids=["l1", "l2"],
    )

    assert card["id"] == "c1"
    req = mock_urlopen.call_args[0][0]
    assert req.get_method() == "POST"
    assert "/cards" in req.full_url
    assert "idList=list123" in req.full_url
    assert "idLabels=l1%2Cl2" in req.full_url


@patch("roshni.integrations.trello.urllib.request.urlopen")
def test_update_card_clears_due_with_null(mock_urlopen):
    mock_urlopen.return_value = _DummyResp({"id": "c1"})

    client = TrelloClient(api_key="k", token="t")
    _ = client.update_card("c1", due="")

    req = mock_urlopen.call_args[0][0]
    assert req.get_method() == "PUT"
    assert "due=null" in req.full_url


@patch("roshni.integrations.trello.urllib.request.urlopen")
def test_search_cards_returns_cards_array(mock_urlopen):
    mock_urlopen.return_value = _DummyResp(
        {
            "cards": [
                {"id": "c1", "name": "One"},
                {"id": "c2", "name": "Two"},
            ]
        }
    )

    client = TrelloClient(api_key="k", token="t")
    cards = client.search_cards("bug", board_id="b1", limit=5)

    assert [c["id"] for c in cards] == ["c1", "c2"]


def test_http_error_raises_api_error():
    client = TrelloClient(api_key="k", token="t")

    def _raise(_req, timeout):
        raise urllib.error.HTTPError(
            url="https://api.trello.com/1/cards",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(b'{"message":"invalid token"}'),
        )

    with patch("roshni.integrations.trello.urllib.request.urlopen", side_effect=_raise):
        with pytest.raises(APIError, match="Trello API 401"):
            client.list_boards()
