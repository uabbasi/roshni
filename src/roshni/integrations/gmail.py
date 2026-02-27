"""Gmail SMTP sender and IMAP reader.

Minimal, dependency-free Gmail integration (uses App Password).
No OAuth dance required -- App Passwords never expire.

No external dependencies beyond the stdlib.
"""

from __future__ import annotations

import email as email_mod
import email.header
import email.utils
import imaplib
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from loguru import logger

DEFAULT_SMTP_SERVER = "smtp.gmail.com"
DEFAULT_SMTP_PORT = 587
DEFAULT_IMAP_SERVER = "imap.gmail.com"
DEFAULT_IMAP_PORT = 993


class GmailSender:
    """Send emails via Gmail SMTP with App Password.

    Args:
        sender: Gmail address (e.g. ``you@gmail.com``).
        app_password: Gmail App Password (not your account password).
        smtp_server: SMTP host.
        smtp_port: SMTP port (587 for STARTTLS).
    """

    def __init__(
        self,
        sender: str,
        app_password: str,
        smtp_server: str = DEFAULT_SMTP_SERVER,
        smtp_port: int = DEFAULT_SMTP_PORT,
    ):
        if not sender or not app_password:
            raise ValueError("sender and app_password are required")

        self.sender = sender
        self.app_password = app_password
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port

    def send(
        self,
        recipient: str,
        subject: str,
        html_body: str,
        text_body: str | None = None,
    ) -> bool:
        """Send an email.

        Args:
            recipient: Destination email address.
            subject: Email subject line.
            html_body: HTML content.
            text_body: Optional plain-text fallback.

        Returns:
            ``True`` if sent successfully.

        Raises:
            smtplib.SMTPAuthenticationError: Bad credentials.
            smtplib.SMTPException: Other SMTP errors.
        """
        msg = MIMEMultipart("alternative")
        msg["From"] = self.sender
        msg["To"] = recipient
        msg["Subject"] = subject

        if text_body:
            msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
            server.starttls()
            server.login(self.sender, self.app_password)
            server.send_message(msg)

        logger.info(f"Email sent to {recipient}: {subject}")
        return True


def _decode_header(raw: str | None) -> str:
    """Decode an RFC 2047 encoded header value to a plain string."""
    if not raw:
        return ""
    parts = email.header.decode_header(raw)
    decoded: list[str] = []
    for data, charset in parts:
        if isinstance(data, bytes):
            decoded.append(data.decode(charset or "utf-8", errors="replace"))
        else:
            decoded.append(data)
    return " ".join(decoded)


def _extract_text(msg: email_mod.message.Message) -> str:
    """Extract plain-text body from an email message, falling back to HTML."""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, errors="replace")
        # Fallback: try HTML
        for part in msg.walk():
            ct = part.get_content_type()
            if ct == "text/html":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    return payload.decode(charset, errors="replace")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="replace")
    return ""


class GmailReader:
    """Read emails from Gmail via IMAP with App Password.

    Args:
        address: Gmail address (e.g. ``you@gmail.com``).
        app_password: Gmail App Password (same one used for SMTP).
        imap_server: IMAP host.
        imap_port: IMAP port (993 for SSL).
    """

    def __init__(
        self,
        address: str,
        app_password: str,
        imap_server: str = DEFAULT_IMAP_SERVER,
        imap_port: int = DEFAULT_IMAP_PORT,
    ):
        if not address or not app_password:
            raise ValueError("address and app_password are required")

        self.address = address
        self.app_password = app_password
        self.imap_server = imap_server
        self.imap_port = imap_port

    def fetch_recent(
        self,
        count: int = 10,
        mailbox: str = "INBOX",
        unread_only: bool = False,
        snippet_chars: int = 200,
    ) -> list[dict[str, str]]:
        """Fetch recent emails and return summaries.

        Args:
            count: Number of recent messages to fetch.
            mailbox: IMAP mailbox name (default ``INBOX``).
            unread_only: If ``True``, only fetch unread messages.
            snippet_chars: Max characters for the body snippet.

        Returns:
            List of dicts with keys: ``from``, ``to``, ``subject``,
            ``date``, ``snippet``, ``unread``.
        """
        conn = imaplib.IMAP4_SSL(self.imap_server, self.imap_port)
        try:
            conn.login(self.address, self.app_password)
            conn.select(mailbox, readonly=True)

            criteria = "UNSEEN" if unread_only else "ALL"
            status, data = conn.search(None, criteria)
            if status != "OK" or not data[0]:
                return []

            msg_ids = data[0].split()
            # Take the most recent `count` messages
            recent_ids = msg_ids[-count:]
            recent_ids.reverse()  # newest first

            results: list[dict[str, str]] = []
            for msg_id in recent_ids:
                status, msg_data = conn.fetch(msg_id, "(FLAGS RFC822)")
                if status != "OK" or not msg_data or not msg_data[0]:
                    continue

                raw_email = msg_data[0][1]
                if not isinstance(raw_email, bytes):
                    continue

                msg = email_mod.message_from_bytes(raw_email)

                # Check flags for unread status
                flags_data = msg_data[0][0] if msg_data[0] else b""
                is_unread = b"\\Seen" not in flags_data if isinstance(flags_data, bytes) else True

                body = _extract_text(msg)
                snippet = body[:snippet_chars].strip() if body else ""

                results.append(
                    {
                        "from": _decode_header(msg.get("From")),
                        "to": _decode_header(msg.get("To")),
                        "subject": _decode_header(msg.get("Subject")),
                        "date": msg.get("Date", ""),
                        "snippet": snippet,
                        "unread": "yes" if is_unread else "no",
                    }
                )

            logger.info(f"Fetched {len(results)} emails from {mailbox}")
            return results
        finally:
            try:
                conn.logout()
            except Exception:
                pass
