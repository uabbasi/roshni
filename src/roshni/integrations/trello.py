"""Trello REST API client.

Thin wrapper around Trello's HTTP API with key/token auth.
No external dependencies beyond the standard library.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from roshni.core.exceptions import APIError

DEFAULT_API_BASE = "https://api.trello.com/1"


class TrelloClient:
    """Minimal Trello client supporting boards, lists, cards, labels, and comments."""

    def __init__(self, api_key: str, token: str, timeout: int = 15, api_base: str = DEFAULT_API_BASE):
        if not api_key or not token:
            raise ValueError("api_key and token are required")
        self.api_key = api_key
        self.token = token
        self.timeout = timeout
        self.api_base = api_base.rstrip("/")

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        query: dict[str, Any] = {"key": self.api_key, "token": self.token}
        if params:
            query.update({k: v for k, v in params.items() if v is not None})

        encoded_query = urllib.parse.urlencode(self._normalize_params(query), doseq=True)
        url = f"{self.api_base}/{path.lstrip('/')}"
        if encoded_query:
            url = f"{url}?{encoded_query}"

        data = None
        headers: dict[str, str] = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(url=url, data=data, method=method.upper(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else ""
            raise APIError(f"Trello API {e.code}: {body or e.reason}") from e
        except urllib.error.URLError as e:
            raise APIError(f"Trello API request failed: {e}") from e

        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8", errors="ignore"))
        except json.JSONDecodeError:
            return {"raw": raw.decode("utf-8", errors="ignore")}

    @staticmethod
    def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for key, value in params.items():
            if isinstance(value, bool):
                normalized[key] = "true" if value else "false"
            elif isinstance(value, (list, tuple)):
                normalized[key] = ",".join(str(v) for v in value)
            else:
                normalized[key] = value
        return normalized

    # Boards
    def list_boards(self, include_closed: bool = False) -> list[dict[str, Any]]:
        boards = self._request(
            "GET",
            "/members/me/boards",
            params={"fields": "id,name,desc,url,closed", "filter": "all" if include_closed else "open"},
        )
        return boards if isinstance(boards, list) else []

    def create_board(self, name: str, desc: str = "") -> dict[str, Any]:
        return self._request("POST", "/boards", params={"name": name, "desc": desc})

    def update_board(self, board_id: str, **changes: Any) -> dict[str, Any]:
        return self._request("PUT", f"/boards/{board_id}", params=changes)

    def delete_board(self, board_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/boards/{board_id}")

    # Lists
    def list_lists(self, board_id: str, include_closed: bool = False) -> list[dict[str, Any]]:
        lists = self._request(
            "GET",
            f"/boards/{board_id}/lists",
            params={"fields": "id,name,closed,pos,idBoard", "filter": "all" if include_closed else "open"},
        )
        return lists if isinstance(lists, list) else []

    def create_list(self, board_id: str, name: str, pos: str = "bottom") -> dict[str, Any]:
        return self._request("POST", f"/boards/{board_id}/lists", params={"name": name, "pos": pos})

    def update_list(self, list_id: str, **changes: Any) -> dict[str, Any]:
        return self._request("PUT", f"/lists/{list_id}", params=changes)

    # Cards
    def list_cards(self, list_id: str, include_closed: bool = False, limit: int = 50) -> list[dict[str, Any]]:
        cards = self._request(
            "GET",
            f"/lists/{list_id}/cards",
            params={
                "fields": "id,name,desc,due,idList,idBoard,labels,url,closed,dateLastActivity",
                "limit": max(1, min(int(limit), 100)),
                "filter": "all" if include_closed else "open",
            },
        )
        return cards if isinstance(cards, list) else []

    def get_card(self, card_id: str) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/cards/{card_id}",
            params={
                "fields": "id,name,desc,due,idList,idBoard,labels,url,closed,dateLastActivity",
                "actions": "commentCard",
                "actions_limit": 20,
            },
        )

    def create_card(
        self,
        list_id: str,
        name: str,
        desc: str = "",
        due: str = "",
        label_ids: list[str] | None = None,
        pos: str = "top",
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"idList": list_id, "name": name, "desc": desc, "pos": pos}
        if due:
            params["due"] = due
        if label_ids:
            params["idLabels"] = label_ids
        return self._request("POST", "/cards", params=params)

    def update_card(self, card_id: str, **changes: Any) -> dict[str, Any]:
        # Trello expects due="null" to clear due dates.
        if "due" in changes and changes["due"] == "":
            changes["due"] = "null"
        return self._request("PUT", f"/cards/{card_id}", params=changes)

    def archive_card(self, card_id: str) -> dict[str, Any]:
        return self.update_card(card_id, closed=True)

    def delete_card(self, card_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/cards/{card_id}")

    def move_card(self, card_id: str, list_id: str, pos: str = "top") -> dict[str, Any]:
        return self.update_card(card_id, idList=list_id, pos=pos)

    def add_comment(self, card_id: str, text: str) -> dict[str, Any]:
        return self._request("POST", f"/cards/{card_id}/actions/comments", params={"text": text})

    # Labels
    def list_labels(self, board_id: str) -> list[dict[str, Any]]:
        labels = self._request("GET", f"/boards/{board_id}/labels", params={"fields": "id,name,color"})
        return labels if isinstance(labels, list) else []

    def create_label(self, board_id: str, name: str, color: str = "") -> dict[str, Any]:
        return self._request("POST", "/labels", params={"idBoard": board_id, "name": name, "color": color})

    def update_label(self, label_id: str, **changes: Any) -> dict[str, Any]:
        return self._request("PUT", f"/labels/{label_id}", params=changes)

    def delete_label(self, label_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/labels/{label_id}")

    def add_label_to_card(self, card_id: str, label_id: str) -> dict[str, Any]:
        return self._request("POST", f"/cards/{card_id}/idLabels", params={"value": label_id})

    def remove_label_from_card(self, card_id: str, label_id: str) -> dict[str, Any]:
        return self._request("DELETE", f"/cards/{card_id}/idLabels/{label_id}")

    # Search
    def search_cards(self, query: str, board_id: str = "", limit: int = 10) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "query": query,
            "modelTypes": "cards",
            "card_fields": "id,name,desc,due,idList,idBoard,labels,url,closed,dateLastActivity",
            "cards_limit": max(1, min(int(limit), 25)),
        }
        if board_id:
            params["idBoards"] = board_id

        result = self._request("GET", "/search", params=params)
        cards = result.get("cards", []) if isinstance(result, dict) else []
        return cards if isinstance(cards, list) else []
