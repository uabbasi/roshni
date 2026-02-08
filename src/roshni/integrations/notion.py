"""Notion API client for database-backed note workflows.

Uses Notion's public REST API with bearer-token auth.
No external dependencies beyond the standard library.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from roshni.core.exceptions import APIError

DEFAULT_API_BASE = "https://api.notion.com/v1"
DEFAULT_NOTION_VERSION = "2022-06-28"


class NotionClient:
    """Small Notion client centered on one target database."""

    def __init__(
        self,
        token: str,
        database_id: str,
        *,
        title_property: str = "",
        timeout: int = 20,
        notion_version: str = DEFAULT_NOTION_VERSION,
        api_base: str = DEFAULT_API_BASE,
    ):
        if not token or not database_id:
            raise ValueError("token and database_id are required")
        self.token = token
        self.database_id = database_id
        self.title_property = title_property
        self.timeout = timeout
        self.notion_version = notion_version
        self.api_base = api_base.rstrip("/")
        self._database_cache: dict[str, Any] | None = None

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.api_base}/{path.lstrip('/')}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query, doseq=True)}"

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Notion-Version": self.notion_version,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url=url, method=method.upper(), data=data, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else ""
            raise APIError(f"Notion API {e.code}: {body or e.reason}") from e
        except urllib.error.URLError as e:
            raise APIError(f"Notion API request failed: {e}") from e

        if not raw:
            return {}
        return json.loads(raw.decode("utf-8", errors="ignore"))

    def get_database(self, *, force_refresh: bool = False) -> dict[str, Any]:
        if self._database_cache is None or force_refresh:
            self._database_cache = self._request("GET", f"/databases/{self.database_id}")
        return dict(self._database_cache)

    def resolve_title_property(self) -> str:
        if self.title_property:
            return self.title_property

        database = self.get_database()
        properties = database.get("properties", {}) if isinstance(database, dict) else {}
        for prop_name, prop_def in properties.items():
            if isinstance(prop_def, dict) and prop_def.get("type") == "title":
                self.title_property = prop_name
                return prop_name

        # Fallback used by many templates.
        self.title_property = "Name"
        return self.title_property

    @staticmethod
    def _title_value(title: str) -> list[dict[str, Any]]:
        return [{"type": "text", "text": {"content": title}}]

    @staticmethod
    def _tags_value(tags: list[str]) -> list[dict[str, str]]:
        return [{"name": tag.strip()} for tag in tags if tag.strip()]

    def list_pages(self, limit: int = 10) -> list[dict[str, Any]]:
        result = self._request(
            "POST",
            f"/databases/{self.database_id}/query",
            payload={"page_size": max(1, min(int(limit), 100))},
        )
        pages = result.get("results", []) if isinstance(result, dict) else []
        return pages if isinstance(pages, list) else []

    def search_pages(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        title_prop = self.resolve_title_property()
        result = self._request(
            "POST",
            f"/databases/{self.database_id}/query",
            payload={
                "page_size": max(1, min(int(limit), 100)),
                "filter": {
                    "property": title_prop,
                    "title": {"contains": query.strip()},
                },
            },
        )
        pages = result.get("results", []) if isinstance(result, dict) else []
        return pages if isinstance(pages, list) else []

    def create_page(
        self,
        title: str,
        *,
        content: str = "",
        tags: list[str] | None = None,
        status: str = "",
    ) -> dict[str, Any]:
        title_prop = self.resolve_title_property()
        props: dict[str, Any] = {
            title_prop: {"title": self._title_value(title)},
        }

        database = self.get_database()
        db_props = database.get("properties", {}) if isinstance(database, dict) else {}

        tags = tags or []
        if tags and "Tags" in db_props and isinstance(db_props.get("Tags"), dict):
            tags_def = db_props["Tags"]
            tags_type = tags_def.get("type")
            if tags_type == "multi_select":
                props["Tags"] = {"multi_select": self._tags_value(tags)}

        if status and "Status" in db_props and isinstance(db_props.get("Status"), dict):
            status_def = db_props["Status"]
            status_type = status_def.get("type")
            if status_type == "status":
                props["Status"] = {"status": {"name": status}}
            elif status_type == "select":
                props["Status"] = {"select": {"name": status}}

        payload: dict[str, Any] = {
            "parent": {"database_id": self.database_id},
            "properties": props,
        }

        if content.strip():
            payload["children"] = [
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": content.strip()},
                            }
                        ]
                    },
                }
            ]

        return self._request("POST", "/pages", payload=payload)

    def update_page(
        self,
        page_id: str,
        *,
        title: str | None = None,
        archived: bool | None = None,
        tags: list[str] | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        props: dict[str, Any] = {}

        if title is not None:
            title_prop = self.resolve_title_property()
            props[title_prop] = {"title": self._title_value(title)}

        if tags is not None:
            database = self.get_database()
            db_props = database.get("properties", {}) if isinstance(database, dict) else {}
            if "Tags" in db_props and isinstance(db_props.get("Tags"), dict):
                tags_def = db_props["Tags"]
                tags_type = tags_def.get("type")
                if tags_type == "multi_select":
                    props["Tags"] = {"multi_select": self._tags_value(tags)}

        if status is not None:
            database = self.get_database()
            db_props = database.get("properties", {}) if isinstance(database, dict) else {}
            if "Status" in db_props and isinstance(db_props.get("Status"), dict):
                status_def = db_props["Status"]
                status_type = status_def.get("type")
                if status_type == "status":
                    props["Status"] = {"status": {"name": status}}
                elif status_type == "select":
                    props["Status"] = {"select": {"name": status}}

        if props:
            payload["properties"] = props

        if archived is not None:
            payload["archived"] = bool(archived)

        return self._request("PATCH", f"/pages/{page_id}", payload=payload)

    def get_page(self, page_id: str) -> dict[str, Any]:
        return self._request("GET", f"/pages/{page_id}")

    def append_paragraph(self, page_id: str, text: str) -> dict[str, Any]:
        payload = {
            "children": [
                {
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [
                            {
                                "type": "text",
                                "text": {"content": text.strip()},
                            }
                        ]
                    },
                }
            ]
        }
        return self._request("PATCH", f"/blocks/{page_id}/children", payload=payload)

    @staticmethod
    def extract_page_title(page: dict[str, Any], title_property: str = "") -> str:
        properties = page.get("properties", {}) if isinstance(page, dict) else {}
        if not isinstance(properties, dict):
            return "Untitled"

        if title_property and isinstance(properties.get(title_property), dict):
            value = properties[title_property].get("title", [])
            if isinstance(value, list) and value:
                title = "".join(part.get("plain_text", "") for part in value if isinstance(part, dict)).strip()
                return title or "Untitled"

        for prop in properties.values():
            if isinstance(prop, dict) and prop.get("type") == "title":
                value = prop.get("title", [])
                if isinstance(value, list):
                    title = "".join(part.get("plain_text", "") for part in value if isinstance(part, dict)).strip()
                    return title or "Untitled"

        return "Untitled"
