"""Trello tools for project/task management."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from roshni.agent.permissions import PermissionTier, filter_tools_by_tier
from roshni.agent.tools import ToolDefinition
from roshni.core.config import Config
from roshni.core.secrets import SecretsManager
from roshni.integrations.trello import TrelloClient


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


def _enrich_card_urgency(card: dict[str, Any], now: datetime | None = None) -> dict[str, Any]:
    """Parse ISO due date and add urgency fields: days_until_due, is_overdue, overdue_days."""
    now = now or datetime.now(UTC)
    due_str = card.get("due")
    if not due_str:
        return card
    try:
        due_dt = datetime.fromisoformat(due_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return card
    delta = (due_dt - now).total_seconds() / 86400
    card["days_until_due"] = delta
    card["is_overdue"] = delta < 0
    card["overdue_days"] = max(0, int(-delta)) if delta < 0 else 0
    return card


def _format_due_display(card: dict[str, Any]) -> str:
    """Return human-readable urgency string for a card's due date."""
    if "days_until_due" not in card:
        return ""
    days = card["days_until_due"]
    if card.get("is_overdue"):
        return f"OVERDUE ({card['overdue_days']}d ago)"
    if days < 1:
        return "TODAY"
    if days < 2:
        return "tomorrow"
    return f"in {int(days)}d"


def _find_list_by_name(client: TrelloClient, board_id: str, name: str) -> dict[str, Any] | None:
    """Find a list by name: exact match first, then case-insensitive partial."""
    lists = client.list_lists(board_id)
    # Exact match (case-insensitive)
    for lst in lists:
        if lst.get("name", "").lower() == name.lower():
            return lst
    # Partial match (case-insensitive)
    for lst in lists:
        if name.lower() in lst.get("name", "").lower():
            return lst
    return None


def _get_trello_today(client: TrelloClient, board_id: str) -> str:
    """Build a daily briefing from 'Today' and 'This Week' lists."""
    now = datetime.now(UTC)
    sections: list[str] = []

    for list_name in ("Today", "This Week"):
        lst = _find_list_by_name(client, board_id, list_name)
        if not lst:
            continue
        cards = client.list_cards(lst["id"], limit=50)
        if not cards:
            continue
        for c in cards:
            _enrich_card_urgency(c, now)

        # Sort: overdue first (by overdue_days desc), then today, then rest
        def _sort_key(c: dict[str, Any]) -> tuple[int, float]:
            if c.get("is_overdue"):
                return (0, -c.get("overdue_days", 0))
            if "days_until_due" in c and c["days_until_due"] < 1:
                return (1, c["days_until_due"])
            return (2, c.get("days_until_due", 999))

        cards.sort(key=_sort_key)
        lines = [f"## {list_name} ({len(cards)} cards)"]
        for c in cards:
            urgency = _format_due_display(c)
            suffix = f" [{urgency}]" if urgency else ""
            lines.append(f"- {c.get('name', '(untitled)')}{suffix}")
        sections.append("\n".join(lines))

    if not sections:
        return "No 'Today' or 'This Week' lists found on this board."
    return "\n\n".join(sections)


def _fmt_boards(items: list[dict[str, Any]]) -> str:
    if not items:
        return "No boards found."
    lines = ["Boards:"]
    for b in items:
        lines.append(f"- {b.get('name', '(unnamed)')} [id={b.get('id')}] closed={b.get('closed', False)}")
    return "\n".join(lines)


def _fmt_lists(items: list[dict[str, Any]]) -> str:
    if not items:
        return "No lists found."
    lines = ["Lists:"]
    for lst in items:
        lines.append(f"- {lst.get('name', '(unnamed)')} [id={lst.get('id')}] closed={lst.get('closed', False)}")
    return "\n".join(lines)


def _fmt_cards(items: list[dict[str, Any]]) -> str:
    if not items:
        return "No cards found."
    now = datetime.now(UTC)
    lines = ["Cards:"]
    for c in items:
        _enrich_card_urgency(c, now)
        urgency = _format_due_display(c)
        due = urgency if urgency else (c.get("due") or "none")
        labels = ", ".join(
            lbl.get("name") or lbl.get("color", "") for lbl in c.get("labels", []) if isinstance(lbl, dict)
        )
        labels = labels or "none"
        lines.append(
            f"- {c.get('name', '(untitled)')} [id={c.get('id')}] "
            f"due={due} labels={labels} closed={c.get('closed', False)}"
        )
    return "\n".join(lines)


def _fmt_labels(items: list[dict[str, Any]]) -> str:
    if not items:
        return "No labels found."
    lines = ["Labels:"]
    for lbl in items:
        lines.append(f"- {lbl.get('name') or '(unnamed)'} [id={lbl.get('id')}] color={lbl.get('color') or 'none'}")
    return "\n".join(lines)


def _fmt_card(card: dict[str, Any]) -> str:
    if not card:
        return "No card found."

    lines = [
        f"Card: {card.get('name', '(untitled)')}",
        f"- id: {card.get('id')}",
        f"- list_id: {card.get('idList')}",
        f"- board_id: {card.get('idBoard')}",
        f"- due: {card.get('due') or 'none'}",
        f"- url: {card.get('url') or 'n/a'}",
        f"- closed: {card.get('closed', False)}",
    ]

    desc = (card.get("desc") or "").strip()
    if desc:
        lines.append("- description:")
        lines.append(desc)

    labels = card.get("labels", []) or []
    if labels:
        label_text = ", ".join(lbl.get("name") or lbl.get("color", "") for lbl in labels if isinstance(lbl, dict))
        lines.append(f"- labels: {label_text or 'none'}")

    checklists = card.get("checklists", []) or []
    for cl in checklists:
        if not isinstance(cl, dict):
            continue
        items = cl.get("checkItems", []) or []
        done = sum(1 for it in items if isinstance(it, dict) and it.get("state") == "complete")
        lines.append(f"- checklist '{cl.get('name', '(unnamed)')}': {done}/{len(items)} complete")
        for it in items:
            if not isinstance(it, dict):
                continue
            mark = "x" if it.get("state") == "complete" else " "
            lines.append(f"  [{mark}] {it.get('name', '')}")

    actions = card.get("actions", []) or []
    comments: list[str] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        data = action.get("data", {})
        text = data.get("text")
        if text:
            comments.append(str(text))
    if comments:
        lines.append("- recent_comments:")
        for c in comments[:10]:
            lines.append(f"  - {c}")

    return "\n".join(lines)


def _fmt_result(label: str, payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return f"{label} done."
    item_id = payload.get("id", "")
    item_name = payload.get("name", "")
    suffix = f" id={item_id}" if item_id else ""
    if item_name:
        suffix += f" name={item_name}"
    return f"{label} done.{suffix}".strip()


def create_trello_tools(
    config: Config,
    secrets: SecretsManager,
    tier: PermissionTier = PermissionTier.INTERACT,
) -> list[ToolDefinition]:
    trello_cfg = config.get("integrations.trello", {}) or {}
    api_key = secrets.get("trello.api_key", "")
    token = secrets.get("trello.token", "")
    if not api_key or not token:
        raise ValueError("Trello is enabled but trello.api_key/trello.token are missing")

    client = TrelloClient(api_key=api_key, token=token)

    def update_board(board_id: str, name: str = "", desc: str = "", closed: str = "") -> str:
        changes: dict[str, Any] = {}
        if name.strip():
            changes["name"] = name.strip()
        if desc.strip():
            changes["desc"] = desc
        closed_value = _parse_optional_bool(closed)
        if closed_value is not None:
            changes["closed"] = closed_value
        if not changes:
            return "No board changes provided."
        return _fmt_result("Board update", client.update_board(board_id, **changes))

    def update_list(list_id: str, name: str = "", pos: str = "", closed: str = "") -> str:
        changes: dict[str, Any] = {}
        if name.strip():
            changes["name"] = name.strip()
        if pos.strip():
            changes["pos"] = pos.strip()
        closed_value = _parse_optional_bool(closed)
        if closed_value is not None:
            changes["closed"] = closed_value
        if not changes:
            return "No list changes provided."
        return _fmt_result("List update", client.update_list(list_id, **changes))

    def update_card(
        card_id: str,
        name: str | None = None,
        desc: str | None = None,
        due: str | None = None,
        id_list: str | None = None,
        closed: str | None = None,
        label_ids: str | None = None,
    ) -> str:
        changes: dict[str, Any] = {}
        if name is not None and name.strip():
            changes["name"] = name.strip()
        if desc is not None:
            changes["desc"] = desc
        if due is not None:
            changes["due"] = due
        if id_list is not None and id_list.strip():
            changes["idList"] = id_list.strip()
        closed_value = _parse_optional_bool(closed)
        if closed_value is not None:
            changes["closed"] = closed_value
        labels = _split_csv(label_ids or "")
        if labels:
            changes["idLabels"] = labels
        if not changes:
            return "No card changes provided."
        return _fmt_result("Card update", client.update_card(card_id, **changes))

    tools: list[ToolDefinition] = [
        ToolDefinition(
            name="trello_list_boards",
            description="List Trello boards for the authenticated user.",
            parameters={
                "type": "object",
                "properties": {
                    "include_closed": {"type": "boolean", "description": "Include closed boards"},
                },
                "required": [],
            },
            function=lambda include_closed=False: _fmt_boards(client.list_boards(bool(include_closed))),
            permission="read",
        ),
        ToolDefinition(
            name="trello_create_board",
            description="Create a new Trello board.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Board name"},
                    "desc": {"type": "string", "description": "Board description"},
                },
                "required": ["name"],
            },
            function=lambda name, desc="": _fmt_result("Board create", client.create_board(name=name, desc=desc)),
            permission="write",
        ),
        ToolDefinition(
            name="trello_update_board",
            description="Update a Trello board (name/description/closed).",
            parameters={
                "type": "object",
                "properties": {
                    "board_id": {"type": "string", "description": "Board ID"},
                    "name": {"type": "string", "description": "New board name"},
                    "desc": {"type": "string", "description": "New board description"},
                    "closed": {"type": "string", "description": "true/false to close or reopen board"},
                },
                "required": ["board_id"],
            },
            function=update_board,
            permission="write",
        ),
        ToolDefinition(
            name="trello_delete_board",
            description="Delete a Trello board permanently.",
            parameters={
                "type": "object",
                "properties": {
                    "board_id": {"type": "string", "description": "Board ID"},
                },
                "required": ["board_id"],
            },
            function=lambda board_id: _fmt_result("Board delete", client.delete_board(board_id)),
            permission="admin",
        ),
        ToolDefinition(
            name="trello_list_lists",
            description="List lists in a Trello board.",
            parameters={
                "type": "object",
                "properties": {
                    "board_id": {"type": "string", "description": "Board ID"},
                    "include_closed": {"type": "boolean", "description": "Include archived lists"},
                },
                "required": ["board_id"],
            },
            function=lambda board_id, include_closed=False: _fmt_lists(
                client.list_lists(board_id, bool(include_closed))
            ),
            permission="read",
        ),
        ToolDefinition(
            name="trello_create_list",
            description="Create a list in a Trello board.",
            parameters={
                "type": "object",
                "properties": {
                    "board_id": {"type": "string", "description": "Board ID"},
                    "name": {"type": "string", "description": "List name"},
                    "pos": {"type": "string", "description": "List position (top, bottom, or number)"},
                },
                "required": ["board_id", "name"],
            },
            function=lambda board_id, name, pos="bottom": _fmt_result(
                "List create", client.create_list(board_id=board_id, name=name, pos=pos)
            ),
            permission="write",
        ),
        ToolDefinition(
            name="trello_update_list",
            description="Update a Trello list (name/position/closed).",
            parameters={
                "type": "object",
                "properties": {
                    "list_id": {"type": "string", "description": "List ID"},
                    "name": {"type": "string", "description": "New list name"},
                    "pos": {"type": "string", "description": "New position"},
                    "closed": {"type": "string", "description": "true/false to archive or unarchive"},
                },
                "required": ["list_id"],
            },
            function=update_list,
            permission="write",
        ),
        ToolDefinition(
            name="trello_list_cards",
            description="List cards in a Trello list.",
            parameters={
                "type": "object",
                "properties": {
                    "list_id": {"type": "string", "description": "List ID"},
                    "limit": {"type": "integer", "description": "Max cards (1-100)"},
                    "include_closed": {"type": "boolean", "description": "Include archived cards"},
                },
                "required": ["list_id"],
            },
            function=lambda list_id, limit=20, include_closed=False: _fmt_cards(
                client.list_cards(list_id=list_id, include_closed=bool(include_closed), limit=int(limit))
            ),
            permission="read",
        ),
        ToolDefinition(
            name="trello_get_card",
            description="Get full details for a Trello card (includes recent comments).",
            parameters={
                "type": "object",
                "properties": {
                    "card_id": {"type": "string", "description": "Card ID"},
                },
                "required": ["card_id"],
            },
            function=lambda card_id: _fmt_card(client.get_card(card_id)),
            permission="read",
        ),
        ToolDefinition(
            name="trello_search_cards",
            description="Search cards by text, optionally scoped to one board.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "board_id": {"type": "string", "description": "Optional board ID"},
                    "limit": {"type": "integer", "description": "Max cards (1-25)"},
                },
                "required": ["query"],
            },
            function=lambda query, board_id="", limit=10: _fmt_cards(
                client.search_cards(query=query, board_id=board_id, limit=int(limit))
            ),
            permission="read",
        ),
        ToolDefinition(
            name="trello_create_card",
            description="Create a card in a list, with optional due date and labels.",
            parameters={
                "type": "object",
                "properties": {
                    "list_id": {"type": "string", "description": "List ID"},
                    "name": {"type": "string", "description": "Card title"},
                    "desc": {"type": "string", "description": "Card description"},
                    "due": {"type": "string", "description": "ISO date/time due value"},
                    "label_ids": {"type": "string", "description": "Comma-separated label IDs"},
                    "pos": {"type": "string", "description": "Card position (top, bottom, or number)"},
                },
                "required": ["list_id", "name"],
            },
            function=lambda list_id, name, desc="", due="", label_ids="", pos="top": _fmt_result(
                "Card create",
                client.create_card(
                    list_id=list_id,
                    name=name,
                    desc=desc,
                    due=due,
                    label_ids=_split_csv(label_ids),
                    pos=pos,
                ),
            ),
            permission="write",
        ),
        ToolDefinition(
            name="trello_update_card",
            description="Update a card's name/description/due/list/status/labels.",
            parameters={
                "type": "object",
                "properties": {
                    "card_id": {"type": "string", "description": "Card ID"},
                    "name": {"type": "string", "description": "New card title"},
                    "desc": {"type": "string", "description": "New description"},
                    "due": {"type": "string", "description": "Due date/time (empty string clears due date)"},
                    "id_list": {"type": "string", "description": "Move card to this list ID"},
                    "closed": {"type": "string", "description": "true/false to archive or unarchive"},
                    "label_ids": {"type": "string", "description": "Comma-separated label IDs"},
                },
                "required": ["card_id"],
            },
            function=update_card,
            permission="write",
        ),
        ToolDefinition(
            name="trello_move_card",
            description="Move a card to another list.",
            parameters={
                "type": "object",
                "properties": {
                    "card_id": {"type": "string", "description": "Card ID"},
                    "list_id": {"type": "string", "description": "Destination list ID"},
                    "pos": {"type": "string", "description": "Position in destination list"},
                },
                "required": ["card_id", "list_id"],
            },
            function=lambda card_id, list_id, pos="top": _fmt_result(
                "Card move", client.move_card(card_id=card_id, list_id=list_id, pos=pos)
            ),
            permission="write",
        ),
        ToolDefinition(
            name="trello_add_comment",
            description="Add a comment to a Trello card.",
            parameters={
                "type": "object",
                "properties": {
                    "card_id": {"type": "string", "description": "Card ID"},
                    "text": {"type": "string", "description": "Comment text"},
                },
                "required": ["card_id", "text"],
            },
            function=lambda card_id, text: _fmt_result("Comment add", client.add_comment(card_id, text)),
            permission="write",
        ),
        ToolDefinition(
            name="trello_archive_card",
            description="Archive (close) a Trello card.",
            parameters={
                "type": "object",
                "properties": {
                    "card_id": {"type": "string", "description": "Card ID"},
                },
                "required": ["card_id"],
            },
            function=lambda card_id: _fmt_result("Card archive", client.archive_card(card_id)),
            permission="write",
        ),
        ToolDefinition(
            name="trello_delete_card",
            description="Delete a Trello card permanently.",
            parameters={
                "type": "object",
                "properties": {
                    "card_id": {"type": "string", "description": "Card ID"},
                },
                "required": ["card_id"],
            },
            function=lambda card_id: _fmt_result("Card delete", client.delete_card(card_id)),
            permission="admin",
        ),
        ToolDefinition(
            name="trello_list_labels",
            description="List labels on a board.",
            parameters={
                "type": "object",
                "properties": {
                    "board_id": {"type": "string", "description": "Board ID"},
                },
                "required": ["board_id"],
            },
            function=lambda board_id: _fmt_labels(client.list_labels(board_id)),
            permission="read",
        ),
        ToolDefinition(
            name="trello_create_label",
            description="Create a label on a board.",
            parameters={
                "type": "object",
                "properties": {
                    "board_id": {"type": "string", "description": "Board ID"},
                    "name": {"type": "string", "description": "Label name"},
                    "color": {
                        "type": "string",
                        "description": "Label color (green,yellow,orange,red,purple,blue,sky,lime,pink,black)",
                    },
                },
                "required": ["board_id", "name"],
            },
            function=lambda board_id, name, color="": _fmt_result(
                "Label create", client.create_label(board_id=board_id, name=name, color=color)
            ),
            permission="write",
        ),
        ToolDefinition(
            name="trello_update_label",
            description="Update a label's name or color.",
            parameters={
                "type": "object",
                "properties": {
                    "label_id": {"type": "string", "description": "Label ID"},
                    "name": {"type": "string", "description": "Label name"},
                    "color": {"type": "string", "description": "Label color"},
                },
                "required": ["label_id"],
            },
            function=lambda label_id, name="", color="": _fmt_result(
                "Label update",
                client.update_label(
                    label_id,
                    **{k: v for k, v in {"name": name.strip(), "color": color.strip()}.items() if v},
                ),
            ),
            permission="write",
        ),
        ToolDefinition(
            name="trello_delete_label",
            description="Delete a board label permanently.",
            parameters={
                "type": "object",
                "properties": {
                    "label_id": {"type": "string", "description": "Label ID"},
                },
                "required": ["label_id"],
            },
            function=lambda label_id: _fmt_result("Label delete", client.delete_label(label_id)),
            permission="admin",
        ),
        ToolDefinition(
            name="trello_add_label_to_card",
            description="Attach an existing label to a card.",
            parameters={
                "type": "object",
                "properties": {
                    "card_id": {"type": "string", "description": "Card ID"},
                    "label_id": {"type": "string", "description": "Label ID"},
                },
                "required": ["card_id", "label_id"],
            },
            function=lambda card_id, label_id: _fmt_result("Add label", client.add_label_to_card(card_id, label_id)),
            permission="write",
        ),
        ToolDefinition(
            name="trello_remove_label_from_card",
            description="Remove a label from a card.",
            parameters={
                "type": "object",
                "properties": {
                    "card_id": {"type": "string", "description": "Card ID"},
                    "label_id": {"type": "string", "description": "Label ID"},
                },
                "required": ["card_id", "label_id"],
            },
            function=lambda card_id, label_id: _fmt_result(
                "Remove label", client.remove_label_from_card(card_id, label_id)
            ),
            permission="write",
        ),
        ToolDefinition(
            name="trello_today",
            description="Daily briefing: cards from 'Today' and 'This Week' lists, sorted by urgency.",
            parameters={
                "type": "object",
                "properties": {
                    "board_id": {
                        "type": "string",
                        "description": "Board ID (defaults to configured default_board_id)",
                    },
                },
                "required": [],
            },
            function=lambda board_id="": _get_trello_today(client, board_id or trello_cfg.get("default_board_id", "")),
            permission="read",
        ),
        ToolDefinition(
            name="trello_find_list",
            description="Find a list by name on a board (exact match first, then partial).",
            parameters={
                "type": "object",
                "properties": {
                    "board_id": {"type": "string", "description": "Board ID"},
                    "name": {"type": "string", "description": "List name to search for"},
                },
                "required": ["board_id", "name"],
            },
            function=lambda board_id, name: (
                _fmt_lists([found])
                if (found := _find_list_by_name(client, board_id, name))
                else "No matching list found."
            ),
            permission="read",
        ),
        ToolDefinition(
            name="trello_find_card",
            description="Find a card by name, optionally scoped to a board.",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Card name to search for"},
                    "board_id": {"type": "string", "description": "Optional board ID to scope search"},
                },
                "required": ["name"],
            },
            function=lambda name, board_id="": _fmt_cards(client.search_cards(query=name, board_id=board_id, limit=10)),
            permission="read",
        ),
    ]

    tools = filter_tools_by_tier(tools, tier)

    # Option to disable destructive board deletion at config level.
    if bool(trello_cfg.get("disable_board_delete", False)):
        tools = [t for t in tools if t.name != "trello_delete_board"]

    return tools
