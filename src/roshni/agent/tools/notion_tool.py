"""Notion knowledge tools."""

from __future__ import annotations

from typing import Any

from roshni.agent.permissions import PermissionTier, filter_tools_by_tier
from roshni.agent.tools import ToolDefinition
from roshni.core.config import Config
from roshni.core.secrets import SecretsManager
from roshni.integrations.notion import NotionClient


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def _parse_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    return None


def _fmt_pages(client: NotionClient, pages: list[dict[str, Any]]) -> str:
    if not pages:
        return "No Notion pages found."
    title_prop = client.resolve_title_property()
    lines = ["Notion pages:"]
    for page in pages:
        title = client.extract_page_title(page, title_prop)
        page_id = page.get("id")
        edited = page.get("last_edited_time", "unknown")
        url = page.get("url", "")
        lines.append(f"- {title} [id={page_id}] last_edited={edited} url={url}")
    return "\n".join(lines)


def _fmt_page(client: NotionClient, page: dict[str, Any]) -> str:
    if not page:
        return "No Notion page found."
    title = client.extract_page_title(page, client.resolve_title_property())
    lines = [
        f"Page: {title}",
        f"- id: {page.get('id')}",
        f"- url: {page.get('url', 'n/a')}",
        f"- archived: {page.get('archived', False)}",
        f"- last_edited: {page.get('last_edited_time', 'unknown')}",
    ]
    return "\n".join(lines)


def _fmt_result(label: str, payload: dict[str, Any], client: NotionClient) -> str:
    if not isinstance(payload, dict):
        return f"{label} done."
    page_id = payload.get("id")
    title = client.extract_page_title(payload, client.resolve_title_property())
    suffix = ""
    if page_id:
        suffix += f" id={page_id}"
    if title and title != "Untitled":
        suffix += f" title={title}"
    return f"{label} done.{suffix}".strip()


def create_notion_tools(
    config: Config,
    secrets: SecretsManager,
    tier: PermissionTier = PermissionTier.INTERACT,
) -> list[ToolDefinition]:
    notion_cfg = config.get("integrations.notion", {}) or {}
    token = secrets.get("notion.token", "")
    database_id = notion_cfg.get("database_id", "")
    title_property = notion_cfg.get("title_property", "")

    if not token or not database_id:
        raise ValueError("Notion is enabled but notion.token or integrations.notion.database_id is missing")

    client = NotionClient(token=token, database_id=database_id, title_property=title_property)

    def update_page(page_id: str, title: str = "", archived: str = "", tags: str = "", status: str = "") -> str:
        archived_value = _parse_optional_bool(archived)
        payload = client.update_page(
            page_id,
            title=title if title.strip() else None,
            archived=archived_value,
            tags=_split_csv(tags) if tags.strip() else None,
            status=status if status.strip() else None,
        )
        return _fmt_result("Page update", payload, client)

    tools = [
        ToolDefinition(
            name="notion_list_pages",
            description="List recent pages from the configured Notion database.",
            parameters={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max pages (1-100)"},
                },
                "required": [],
            },
            function=lambda limit=10: _fmt_pages(client, client.list_pages(limit=int(limit))),
            permission="read",
        ),
        ToolDefinition(
            name="notion_search_pages",
            description="Search pages by title in the configured Notion database.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search text"},
                    "limit": {"type": "integer", "description": "Max pages (1-100)"},
                },
                "required": ["query"],
            },
            function=lambda query, limit=10: _fmt_pages(client, client.search_pages(query=query, limit=int(limit))),
            permission="read",
        ),
        ToolDefinition(
            name="notion_get_page",
            description="Get metadata for one Notion page by ID.",
            parameters={
                "type": "object",
                "properties": {
                    "page_id": {"type": "string", "description": "Notion page ID"},
                },
                "required": ["page_id"],
            },
            function=lambda page_id: _fmt_page(client, client.get_page(page_id)),
            permission="read",
        ),
        ToolDefinition(
            name="notion_create_page",
            description="Create a page in Notion with optional body text, tags, and status.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Page title"},
                    "content": {"type": "string", "description": "Optional initial page content"},
                    "tags": {"type": "string", "description": "Optional comma-separated tags"},
                    "status": {"type": "string", "description": "Optional status name"},
                },
                "required": ["title"],
            },
            function=lambda title, content="", tags="", status="": _fmt_result(
                "Page create",
                client.create_page(title=title, content=content, tags=_split_csv(tags), status=status),
                client,
            ),
            permission="write",
        ),
        ToolDefinition(
            name="notion_update_page",
            description="Update Notion page title/status/tags/archive state.",
            parameters={
                "type": "object",
                "properties": {
                    "page_id": {"type": "string", "description": "Page ID"},
                    "title": {"type": "string", "description": "New title"},
                    "archived": {"type": "string", "description": "true/false to archive or unarchive"},
                    "tags": {"type": "string", "description": "Comma-separated tags"},
                    "status": {"type": "string", "description": "Status value"},
                },
                "required": ["page_id"],
            },
            function=update_page,
            permission="write",
        ),
        ToolDefinition(
            name="notion_append_to_page",
            description="Append a text paragraph block to an existing Notion page.",
            parameters={
                "type": "object",
                "properties": {
                    "page_id": {"type": "string", "description": "Page ID"},
                    "text": {"type": "string", "description": "Text to append"},
                },
                "required": ["page_id", "text"],
            },
            function=lambda page_id, text: _fmt_result("Append block", client.append_paragraph(page_id, text), client),
            permission="write",
        ),
        ToolDefinition(
            name="notion_delete_page",
            description="Delete (archive) a Notion page permanently.",
            parameters={
                "type": "object",
                "properties": {
                    "page_id": {"type": "string", "description": "Page ID to delete"},
                },
                "required": ["page_id"],
            },
            function=lambda page_id: _fmt_result("Page delete", client.update_page(page_id, archived=True), client),
            permission="admin",
        ),
    ]

    return filter_tools_by_tier(tools, tier)
