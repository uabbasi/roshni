"""Gmail API (OAuth) functions.

These use the Gmail REST API via ``googleapiclient`` and require an
authenticated ``service`` object (from ``GoogleOAuth.get_service("gmail", "v1")``).

Each function accepts the service as its first argument — no global
state or singleton.  Return values are JSON strings matching the
contract expected by the agent tool layer.
"""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from email.mime.text import MIMEText
from email.utils import parseaddr

from loguru import logger


def search_gmail(service, query: str, max_results: int = 10) -> str:
    """Search Gmail messages using Gmail search syntax."""
    try:
        result = service.users().messages().list(userId="me", q=query, maxResults=max_results).execute()

        messages = result.get("messages", [])
        if not messages:
            return json.dumps({"messages": [], "count": 0, "query": query})

        formatted = []
        for msg_ref in messages:
            msg = service.users().messages().get(userId="me", id=msg_ref["id"], format="metadata").execute()

            headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}

            formatted.append(
                {
                    "id": msg["id"],
                    "subject": headers.get("Subject", "(No subject)"),
                    "from": headers.get("From", ""),
                    "date": headers.get("Date", ""),
                    "snippet": msg.get("snippet", ""),
                    "labels": msg.get("labelIds", []),
                    "unread": "UNREAD" in msg.get("labelIds", []),
                }
            )

        return json.dumps({"messages": formatted, "count": len(formatted), "query": query})

    except Exception as e:
        logger.error(f"Gmail search failed: {e}")
        return json.dumps({"error": str(e)})


def create_gmail_draft(service, to: str, subject: str, body: str) -> str:
    """Create a Gmail draft (user sends manually from Gmail)."""
    try:
        message = MIMEText(body)
        message["to"] = to
        message["subject"] = subject

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        result = service.users().drafts().create(userId="me", body={"message": {"raw": raw}}).execute()

        return json.dumps(
            {
                "success": True,
                "action": "create_gmail_draft",
                "mode": "new",
                "draft_id": result.get("id"),
                "thread_id": (result.get("message") or {}).get("threadId"),
                "to": to,
                "subject": subject,
                "evidence": {
                    "draft_id": result.get("id"),
                    "thread_id": (result.get("message") or {}).get("threadId"),
                    "to": to,
                    "subject": subject,
                },
                "message": "Draft created — open Gmail to review and send.",
            }
        )

    except Exception as e:
        logger.error(f"Gmail draft failed: {e}")
        return json.dumps(
            {
                "success": False,
                "action": "create_gmail_draft",
                "mode": "new",
                "error": str(e),
                "retryable": True,
            }
        )


def create_gmail_reply_draft(service, message_id: str, body: str, subject_override: str = "") -> str:
    """Create a Gmail reply draft in the original thread."""
    try:
        msg = service.users().messages().get(userId="me", id=message_id, format="metadata").execute()

        payload_headers = msg.get("payload", {}).get("headers", [])
        headers = {h.get("name", ""): h.get("value", "") for h in payload_headers}

        from_header = headers.get("Reply-To") or headers.get("From", "")
        recipient = parseaddr(from_header)[1] or from_header
        original_subject = headers.get("Subject", "").strip()
        if subject_override.strip():
            subject = subject_override.strip()
        elif original_subject.lower().startswith("re:"):
            subject = original_subject
        else:
            subject = f"Re: {original_subject}" if original_subject else "Re:"

        original_message_id_header = headers.get("Message-Id", "").strip()
        references = headers.get("References", "").strip()
        if original_message_id_header:
            references = (
                f"{references} {original_message_id_header}".strip() if references else original_message_id_header
            )

        reply = MIMEText(body)
        if recipient:
            reply["to"] = recipient
        reply["subject"] = subject
        if original_message_id_header:
            reply["In-Reply-To"] = original_message_id_header
            reply["References"] = references

        raw = base64.urlsafe_b64encode(reply.as_bytes()).decode()
        result = (
            service.users()
            .drafts()
            .create(
                userId="me",
                body={
                    "message": {
                        "raw": raw,
                        "threadId": msg.get("threadId"),
                    }
                },
            )
            .execute()
        )

        return json.dumps(
            {
                "success": True,
                "action": "create_gmail_reply_draft",
                "mode": "reply",
                "draft_id": result.get("id"),
                "thread_id": (result.get("message") or {}).get("threadId") or msg.get("threadId"),
                "reply_to_message_id": message_id,
                "to": recipient,
                "subject": subject,
                "evidence": {
                    "draft_id": result.get("id"),
                    "thread_id": (result.get("message") or {}).get("threadId") or msg.get("threadId"),
                    "reply_to_message_id": message_id,
                    "to": recipient,
                    "subject": subject,
                },
                "message": "Reply draft created in the original Gmail thread.",
            }
        )

    except Exception as e:
        logger.error(f"Gmail reply draft failed: {e}")
        return json.dumps(
            {
                "success": False,
                "action": "create_gmail_reply_draft",
                "mode": "reply",
                "reply_to_message_id": message_id,
                "error": str(e),
                "retryable": True,
            }
        )


def get_gmail_summary(service) -> str:
    """Get Gmail inbox summary — starred unread + important unread (separate buckets)."""
    try:
        # Get unread count
        unread = service.users().messages().list(userId="me", q="is:unread", maxResults=1).execute()
        unread_count = unread.get("resultSizeEstimate", 0)

        # Threads with existing drafts are usually already being handled.
        # Suppress fresh draft threads; only resurface when stale.
        draft_thread_last_ts_ms: dict[str, int] = {}
        draft_stale_after_hours = 24
        try:
            drafts_resp = service.users().drafts().list(userId="me", maxResults=100).execute()
            for draft in drafts_resp.get("drafts", []):
                message = draft.get("message") or {}
                thread_id = message.get("threadId")
                if not thread_id:
                    continue
                internal_date_raw = message.get("internalDate")
                try:
                    internal_date_ms = int(internal_date_raw) if internal_date_raw is not None else 0
                except (TypeError, ValueError):
                    internal_date_ms = 0
                prev = draft_thread_last_ts_ms.get(thread_id, 0)
                if internal_date_ms > prev:
                    draft_thread_last_ts_ms[thread_id] = internal_date_ms
        except Exception as e:
            logger.debug(f"Gmail draft thread scan skipped: {e}")

        now_ms = int(datetime.now(UTC).timestamp() * 1000)

        def _fetch_messages(query: str, max_results: int = 5) -> tuple[list[dict], int, int]:
            result = service.users().messages().list(userId="me", q=query, maxResults=max_results).execute()
            msgs = []
            suppressed_due_to_fresh_draft = 0
            stale_draft_threads = 0
            for msg_ref in result.get("messages", []):
                msg = service.users().messages().get(userId="me", id=msg_ref["id"], format="metadata").execute()
                thread_id = msg.get("threadId")
                has_draft = bool(thread_id and thread_id in draft_thread_last_ts_ms)
                status = "new_unread"
                if has_draft and thread_id:
                    draft_age_hours = (now_ms - draft_thread_last_ts_ms[thread_id]) / (1000 * 60 * 60)
                    if draft_age_hours < draft_stale_after_hours:
                        suppressed_due_to_fresh_draft += 1
                        continue
                    status = "stale_draft"
                    stale_draft_threads += 1
                headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}
                msgs.append(
                    {
                        "id": msg["id"],
                        "thread_id": thread_id,
                        "subject": headers.get("Subject", "(No subject)"),
                        "from": headers.get("From", ""),
                        "date": headers.get("Date", ""),
                        "snippet": msg.get("snippet", ""),
                        "status": status,
                    }
                )
            return msgs, suppressed_due_to_fresh_draft, stale_draft_threads

        # Bucket 1: starred unread — always surface these
        starred_msgs, starred_suppressed, starred_stale = _fetch_messages("is:starred is:unread", max_results=5)
        starred_ids = {m["id"] for m in starred_msgs}

        # Bucket 2: important unread (Gmail's ML filter) — Hakim triages these
        important_msgs, important_suppressed, important_stale = _fetch_messages(
            "is:important is:unread -is:starred", max_results=10
        )
        # Exclude any that are also starred (already in bucket 1)
        important_msgs = [m for m in important_msgs if m["id"] not in starred_ids]

        return json.dumps(
            {
                "unread_count": unread_count,
                "starred_unread": starred_msgs,
                "starred_count": len(starred_msgs),
                "important_unread": important_msgs,
                "important_count": len(important_msgs),
                "suppressed_due_to_existing_draft": starred_suppressed + important_suppressed,
                "stale_draft_threads_count": starred_stale + important_stale,
                "draft_stale_after_hours": draft_stale_after_hours,
                "triage_note": (
                    "Starred = always surface. "
                    "Important = use judgment — only surface if time-sensitive, "
                    "action-required, or from a known contact about something that can't wait."
                ),
            }
        )

    except Exception as e:
        logger.error(f"Gmail summary failed: {e}")
        return json.dumps({"error": str(e)})
