"""Tests for roshni.integrations.notion."""

from __future__ import annotations

import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from roshni.core.exceptions import APIError
from roshni.integrations.notion import NotionClient


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
        NotionClient(token="", database_id="db")
    with pytest.raises(ValueError):
        NotionClient(token="tok", database_id="")


@patch("roshni.integrations.notion.urllib.request.urlopen")
def test_resolve_title_property_from_database(mock_urlopen):
    mock_urlopen.return_value = _DummyResp(
        {
            "properties": {
                "Name": {"type": "title"},
                "Tags": {"type": "multi_select"},
            }
        }
    )

    client = NotionClient(token="tok", database_id="db")
    prop = client.resolve_title_property()

    assert prop == "Name"
    req = mock_urlopen.call_args[0][0]
    assert req.get_method() == "GET"
    assert "/databases/db" in req.full_url


@patch("roshni.integrations.notion.urllib.request.urlopen")
def test_search_pages(mock_urlopen):
    captured_payloads: list[dict] = []

    def _side_effect(req, timeout):
        if req.get_method() == "GET":
            return _DummyResp({"properties": {"Name": {"type": "title"}}})
        payload = json.loads(req.data.decode("utf-8"))
        captured_payloads.append(payload)
        return _DummyResp({"results": [{"id": "p1"}]})

    mock_urlopen.side_effect = _side_effect

    client = NotionClient(token="tok", database_id="db")
    pages = client.search_pages("hello", limit=5)

    assert pages == [{"id": "p1"}]
    assert captured_payloads
    assert captured_payloads[0]["filter"]["property"] == "Name"
    assert captured_payloads[0]["filter"]["title"]["contains"] == "hello"


@patch("roshni.integrations.notion.urllib.request.urlopen")
def test_create_page_with_tags_and_status(mock_urlopen):
    payloads: list[dict] = []

    def _side_effect(req, timeout):
        if req.get_method() == "GET":
            return _DummyResp(
                {
                    "properties": {
                        "Name": {"type": "title"},
                        "Tags": {"type": "multi_select"},
                        "Status": {"type": "status"},
                    }
                }
            )
        payloads.append(json.loads(req.data.decode("utf-8")))
        return _DummyResp({"id": "page1"})

    mock_urlopen.side_effect = _side_effect

    client = NotionClient(token="tok", database_id="db")
    page = client.create_page(title="My Page", content="Body", tags=["work", "urgent"], status="In Progress")

    assert page["id"] == "page1"
    create_payload = payloads[-1]
    props = create_payload["properties"]
    assert props["Name"]["title"][0]["text"]["content"] == "My Page"
    assert props["Tags"]["multi_select"][0]["name"] == "work"
    assert props["Status"]["status"]["name"] == "In Progress"
    assert create_payload["children"][0]["type"] == "paragraph"


def test_extract_page_title():
    page = {
        "properties": {
            "Name": {
                "type": "title",
                "title": [{"plain_text": "Hello"}, {"plain_text": " World"}],
            }
        }
    }
    title = NotionClient.extract_page_title(page, "Name")
    assert title == "Hello World"


def test_http_error_raises_api_error():
    client = NotionClient(token="tok", database_id="db")

    def _raise(_req, timeout):
        raise urllib.error.HTTPError(
            url="https://api.notion.com/v1/pages",
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=io.BytesIO(b'{"message":"forbidden"}'),
        )

    with patch("roshni.integrations.notion.urllib.request.urlopen", side_effect=_raise):
        with pytest.raises(APIError, match="Notion API 403"):
            client.get_database()
