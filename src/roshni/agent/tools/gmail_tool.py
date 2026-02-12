"""Gmail tools.

Safety-first behavior:
  - Always supports local draft creation.
  - Sending is optional and disabled by default.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from roshni.agent.permissions import PermissionTier, filter_tools_by_tier
from roshni.agent.tools import ToolDefinition
from roshni.core.config import Config
from roshni.core.secrets import SecretsManager


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


def create_gmail_tools(
    config: Config,
    secrets: SecretsManager,
    tier: PermissionTier = PermissionTier.INTERACT,
) -> list[ToolDefinition]:
    """Create Gmail tools using config-driven safety settings."""
    tools: list[ToolDefinition] = []

    gmail_cfg = config.get("integrations.gmail", {}) or {}
    drafts_dir = config.get("paths.email_drafts_dir", "~/.roshni/drafts/email")

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

    allow_send = bool(gmail_cfg.get("allow_send", False))
    address = secrets.get("gmail.address", "")
    app_password = secrets.get("gmail.app_password", "")
    if allow_send and address and app_password:
        tools.append(
            ToolDefinition(
                name="send_email",
                description=(
                    "Send an email immediately. High-risk action: use only when the user explicitly asks to send now."
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

    return filter_tools_by_tier(tools, tier)
