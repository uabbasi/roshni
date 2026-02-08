"""Gmail tool — send emails via the existing GmailSender."""

from __future__ import annotations

from roshni.agent.tools import ToolDefinition
from roshni.core.secrets import SecretsManager


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


def create_gmail_tools(secrets: SecretsManager) -> list[ToolDefinition]:
    """Create email sending tool."""
    address = secrets.get("gmail.address", "")
    app_password = secrets.get("gmail.app_password", "")

    if not address or not app_password:
        return []  # Can't create tools without credentials

    return [
        ToolDefinition(
            name="send_email",
            description=(
                "Send an email on behalf of the user. Use this when the user asks you to "
                "email someone. Always confirm the recipient, subject, and body before sending."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "recipient": {
                        "type": "string",
                        "description": "Email address to send to",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line",
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body text",
                    },
                },
                "required": ["recipient", "subject", "body"],
            },
            function=lambda recipient, subject, body: _send_email(
                recipient, subject, body, sender=address, app_password=app_password
            ),
        ),
    ]
