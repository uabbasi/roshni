"""Gmail tools.

Two backends:
  1. **OAuth** (preferred) — uses Gmail REST API via ``GoogleOAuth``.
     Provides ``search_gmail``, ``create_gmail_draft``,
     ``create_gmail_reply_draft``, ``get_gmail_summary``.
  2. **IMAP/SMTP** (fallback) — uses App Passwords.
     Provides ``check_email``, ``send_email``.

Local-file ``create_email_draft`` is always available regardless of backend.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, Field

from roshni.agent.permissions import PermissionTier, filter_tools_by_tier
from roshni.agent.tools import ToolDefinition
from roshni.core.config import Config
from roshni.core.secrets import SecretsManager

# ---------- Gmail OAuth Pydantic schemas ----------


class SearchGmailInput(BaseModel):
    query: str = Field(description="Gmail search query (same syntax as Gmail search box)")
    max_results: int = Field(default=10, description="Maximum number of messages to return")


class CreateGmailDraftInput(BaseModel):
    to: str = Field(description="Recipient email address")
    subject: str = Field(description="Email subject")
    body: str = Field(description="Email body (plain text)")


class CreateGmailReplyDraftInput(BaseModel):
    message_id: str = Field(description="Gmail message ID to reply to")
    body: str = Field(description="Reply body (plain text)")
    subject_override: str = Field(
        default="",
        description="Optional subject override. Leave empty to auto-prefix original subject with Re: if needed.",
    )


class GetGmailSummaryInput(BaseModel):
    pass


# ---------- Local draft helpers ----------


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return cleaned[:48] or "draft"


def _create_email_draft(
    recipient: str,
    subject: str,
    body: str,
    *,
    drafts_dir: str,
    cc: str = "",
    bcc: str = "",
) -> str:
    """Save an email draft to local storage."""
    base = Path(drafts_dir).expanduser()
    base.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    filename = f"{now.strftime('%Y-%m-%d_%H%M%S')}_{_slugify(subject)}.md"
    path = base / filename

    lines = [
        "# Email Draft",
        "",
        f"- To: {recipient}",
        f"- Subject: {subject}",
        f"- CC: {cc or '(none)'}",
        f"- BCC: {bcc or '(none)'}",
        f"- Saved: {now.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Body",
        "",
        body.strip(),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return f"Draft saved: {path}"


# ---------- IMAP/SMTP fallback helpers ----------


def _check_email(
    *,
    address: str = "",
    app_password: str = "",
    count: int = 10,
    unread_only: bool = False,
) -> str:
    """Fetch recent emails via Gmail IMAP."""
    from roshni.integrations.gmail import GmailReader

    try:
        reader = GmailReader(address=address, app_password=app_password)
        emails = reader.fetch_recent(count=count, unread_only=unread_only)
        if not emails:
            return "No emails found." if not unread_only else "No unread emails."

        lines: list[str] = []
        for i, msg in enumerate(emails, 1):
            unread_marker = " [UNREAD]" if msg["unread"] == "yes" else ""
            lines.append(f"**{i}. {msg['subject']}**{unread_marker}")
            lines.append(f"   From: {msg['from']}")
            lines.append(f"   Date: {msg['date']}")
            if msg["snippet"]:
                lines.append(f"   Preview: {msg['snippet']}")
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        return f"Failed to check email: {e}"


def _send_email(
    recipient: str,
    subject: str,
    body: str,
    *,
    sender: str = "",
    app_password: str = "",
) -> str:
    """Send an email via Gmail SMTP."""
    from roshni.integrations.gmail import GmailSender

    try:
        gmail = GmailSender(sender=sender, app_password=app_password)
        gmail.send(recipient=recipient, subject=subject, html_body=body, text_body=body)
        return f"Email sent to {recipient}: {subject}"
    except Exception as e:
        return f"Failed to send email: {e}"


# ---------- OAuth tool builder ----------


def _build_oauth_tools(gmail_service) -> list[ToolDefinition]:
    """Build 4 OAuth-backed Gmail tools using a pre-authenticated service."""
    from roshni.integrations.gmail_api import (
        create_gmail_draft,
        create_gmail_reply_draft,
        get_gmail_summary,
        search_gmail,
    )

    return [
        ToolDefinition.from_function(
            func=lambda query, max_results=10: search_gmail(gmail_service, query, max_results),
            name="search_gmail",
            description=(
                "Search Gmail messages using Gmail search syntax. "
                "Examples: 'from:boss@company.com', 'subject:invoice', 'is:unread after:2026/01/01'. "
                "Returns subject, sender, date, and snippet for each message."
            ),
            args_schema=SearchGmailInput,
            service_name="gmail",
        ),
        ToolDefinition.from_function(
            func=lambda to, subject, body: create_gmail_draft(gmail_service, to, subject, body),
            name="create_gmail_draft",
            description=(
                "Create a NEW Gmail draft (starts a new thread). "
                "Use this when composing a brand-new email. "
                "For replies to an existing message, use create_gmail_reply_draft instead."
            ),
            args_schema=CreateGmailDraftInput,
            permission="send",
            requires_approval=False,
            service_name="gmail",
        ),
        ToolDefinition.from_function(
            func=lambda message_id, body, subject_override="": create_gmail_reply_draft(
                gmail_service, message_id, body, subject_override
            ),
            name="create_gmail_reply_draft",
            description=(
                "Create a Gmail REPLY draft in an existing thread using a specific message_id. "
                "Use this when the user asks to reply/follow up on an existing email."
            ),
            args_schema=CreateGmailReplyDraftInput,
            permission="send",
            requires_approval=False,
            service_name="gmail",
        ),
        ToolDefinition.from_function(
            func=lambda: get_gmail_summary(gmail_service),
            name="get_gmail_summary",
            description=(
                "Get Gmail inbox summary with two buckets: "
                "starred_unread (always surface to user) and important_unread (triage for urgency — "
                "only surface if time-sensitive, action-required, or financial/medical/legal). "
                "Use for: 'any new email', 'inbox summary', 'urgent messages'"
            ),
            args_schema=GetGmailSummaryInput,
            service_name="gmail",
        ),
    ]


# ---------- Factory ----------


def create_gmail_tools(
    config: Config,
    secrets: SecretsManager,
    tier: PermissionTier = PermissionTier.INTERACT,
    google_oauth=None,
) -> list[ToolDefinition]:
    """Create Gmail tools using config-driven safety settings.

    Args:
        config: Application config.
        secrets: Secrets manager for IMAP/SMTP credentials.
        tier: Permission tier to filter tools by.
        google_oauth: Optional ``GoogleOAuth`` instance.  When provided and
            a Gmail service can be obtained, the 4 OAuth tools are used
            instead of the IMAP/SMTP fallback tools.
    """
    tools: list[ToolDefinition] = []

    gmail_cfg = config.get("integrations.gmail", {}) or {}
    drafts_dir = config.get("paths.email_drafts_dir", "~/.roshni/drafts/email")

    # --- Try OAuth path first ---
    gmail_service = None
    if google_oauth is not None:
        try:
            gmail_service = google_oauth.get_service("gmail", "v1")
        except Exception as e:
            logger.warning(f"Gmail OAuth service unavailable, falling back to IMAP: {e}")

    if gmail_service:
        tools.extend(_build_oauth_tools(gmail_service))
    else:
        # --- IMAP/SMTP fallback ---
        address = secrets.get("gmail.address", "")
        app_password = secrets.get("gmail.app_password", "")
        if address and app_password:
            tools.append(
                ToolDefinition(
                    name="check_email",
                    description=(
                        "Check the user's Gmail inbox. Returns recent emails with subject, sender, date, "
                        "and a short preview. Can filter to unread only."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "count": {
                                "type": "integer",
                                "description": "Number of recent emails to fetch (default 10, max 25)",
                            },
                            "unread_only": {
                                "type": "boolean",
                                "description": "If true, only show unread emails",
                            },
                        },
                        "required": [],
                    },
                    function=lambda count=10, unread_only=False: _check_email(
                        address=address,
                        app_password=app_password,
                        count=min(int(count), 25),
                        unread_only=bool(unread_only),
                    ),
                    permission="read",
                    requires_approval=False,
                    timeout=30.0,
                    service_name="gmail",
                )
            )

        allow_send = bool(gmail_cfg.get("allow_send", False))
        if allow_send and address and app_password:
            tools.append(
                ToolDefinition(
                    name="send_email",
                    description=(
                        "Send an email immediately. High-risk action: "
                        "use only when the user explicitly asks to send now."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "recipient": {"type": "string", "description": "Email address to send to"},
                            "subject": {"type": "string", "description": "Email subject line"},
                            "body": {"type": "string", "description": "Email body text"},
                        },
                        "required": ["recipient", "subject", "body"],
                    },
                    function=lambda recipient, subject, body: _send_email(
                        recipient=recipient,
                        subject=subject,
                        body=body,
                        sender=address,
                        app_password=app_password,
                    ),
                    permission="send",
                )
            )

    # Local-file draft — always available regardless of backend
    tools.append(
        ToolDefinition(
            name="create_email_draft",
            description=(
                "Create an email draft (does NOT send). Use this for all email composition by default. "
                "The draft is saved locally for review."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "recipient": {"type": "string", "description": "Recipient email address"},
                    "subject": {"type": "string", "description": "Email subject line"},
                    "body": {"type": "string", "description": "Email body text"},
                    "cc": {"type": "string", "description": "Optional CC addresses"},
                    "bcc": {"type": "string", "description": "Optional BCC addresses"},
                },
                "required": ["recipient", "subject", "body"],
            },
            function=lambda recipient, subject, body, cc="", bcc="": _create_email_draft(
                recipient=recipient,
                subject=subject,
                body=body,
                drafts_dir=drafts_dir,
                cc=cc,
                bcc=bcc,
            ),
            permission="write",
            requires_approval=False,
        )
    )

    return filter_tools_by_tier(tools, tier)
