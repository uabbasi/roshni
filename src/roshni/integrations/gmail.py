"""Gmail SMTP sender.

A minimal, dependency-free SMTP sender for Gmail (uses App Password).
No OAuth dance required -- App Passwords never expire.

No external dependencies beyond the stdlib.
"""

from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from loguru import logger

DEFAULT_SMTP_SERVER = "smtp.gmail.com"
DEFAULT_SMTP_PORT = 587


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
