"""Tests for roshni.integrations.gmail."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from roshni.integrations.gmail import GmailReader, GmailSender


class TestGmailSender:
    def test_init_requires_credentials(self):
        with pytest.raises(ValueError, match="sender and app_password are required"):
            GmailSender(sender="", app_password="pass")

        with pytest.raises(ValueError, match="sender and app_password are required"):
            GmailSender(sender="me@gmail.com", app_password="")

    def test_init_defaults(self):
        sender = GmailSender(sender="me@gmail.com", app_password="secret")
        assert sender.smtp_server == "smtp.gmail.com"
        assert sender.smtp_port == 587

    def test_init_custom_server(self):
        sender = GmailSender(
            sender="me@gmail.com",
            app_password="secret",
            smtp_server="smtp.custom.com",
            smtp_port=465,
        )
        assert sender.smtp_server == "smtp.custom.com"
        assert sender.smtp_port == 465

    @patch("roshni.integrations.gmail.smtplib.SMTP")
    def test_send_html_only(self, mock_smtp_cls):
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        sender = GmailSender(sender="me@gmail.com", app_password="secret")
        result = sender.send(
            recipient="you@gmail.com",
            subject="Test",
            html_body="<h1>Hello</h1>",
        )

        assert result is True
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("me@gmail.com", "secret")
        mock_server.send_message.assert_called_once()

    @patch("roshni.integrations.gmail.smtplib.SMTP")
    def test_send_with_text_fallback(self, mock_smtp_cls):
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        sender = GmailSender(sender="me@gmail.com", app_password="secret")
        result = sender.send(
            recipient="you@gmail.com",
            subject="Test",
            html_body="<h1>Hello</h1>",
            text_body="Hello",
        )

        assert result is True
        # Verify the message was sent (text + html parts)
        msg = mock_server.send_message.call_args[0][0]
        payloads = msg.get_payload()
        assert len(payloads) == 2  # text + html

    @patch("roshni.integrations.gmail.smtplib.SMTP")
    def test_send_auth_failure_propagates(self, mock_smtp_cls):
        import smtplib

        mock_server = MagicMock()
        mock_server.login.side_effect = smtplib.SMTPAuthenticationError(535, b"Bad creds")
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        sender = GmailSender(sender="me@gmail.com", app_password="wrong")
        with pytest.raises(smtplib.SMTPAuthenticationError):
            sender.send("you@gmail.com", "Test", "<p>Hi</p>")


class TestGmailReader:
    def test_init_requires_credentials(self):
        with pytest.raises(ValueError, match="address and app_password are required"):
            GmailReader(address="", app_password="pass")

        with pytest.raises(ValueError, match="address and app_password are required"):
            GmailReader(address="me@gmail.com", app_password="")

    def test_init_defaults(self):
        reader = GmailReader(address="me@gmail.com", app_password="secret")
        assert reader.imap_server == "imap.gmail.com"
        assert reader.imap_port == 993

    def test_init_custom_server(self):
        reader = GmailReader(
            address="me@gmail.com",
            app_password="secret",
            imap_server="imap.custom.com",
            imap_port=143,
        )
        assert reader.imap_server == "imap.custom.com"
        assert reader.imap_port == 143

    @patch("roshni.integrations.gmail.imaplib.IMAP4_SSL")
    def test_fetch_recent_returns_emails(self, mock_imap_cls):
        mock_conn = MagicMock()
        mock_imap_cls.return_value = mock_conn
        mock_conn.login.return_value = ("OK", [])
        mock_conn.select.return_value = ("OK", [b"3"])
        mock_conn.search.return_value = ("OK", [b"1 2 3"])

        # Build a simple RFC822 email message
        from email.mime.text import MIMEText

        msg = MIMEText("Hello, this is the body.")
        msg["From"] = "sender@example.com"
        msg["To"] = "me@gmail.com"
        msg["Subject"] = "Test Subject"
        msg["Date"] = "Thu, 27 Feb 2026 10:00:00 +0000"
        raw = msg.as_bytes()

        # IMAP fetch returns (flags_line, raw_email) tuples
        mock_conn.fetch.return_value = ("OK", [(b"1 (FLAGS (\\Seen) RFC822 {1234})", raw)])

        reader = GmailReader(address="me@gmail.com", app_password="secret")
        results = reader.fetch_recent(count=2)

        assert len(results) == 2
        assert results[0]["from"] == "sender@example.com"
        assert results[0]["subject"] == "Test Subject"
        assert "Hello" in results[0]["snippet"]
        assert results[0]["unread"] == "no"  # \Seen flag present

    @patch("roshni.integrations.gmail.imaplib.IMAP4_SSL")
    def test_fetch_recent_empty_inbox(self, mock_imap_cls):
        mock_conn = MagicMock()
        mock_imap_cls.return_value = mock_conn
        mock_conn.login.return_value = ("OK", [])
        mock_conn.select.return_value = ("OK", [b"0"])
        mock_conn.search.return_value = ("OK", [b""])

        reader = GmailReader(address="me@gmail.com", app_password="secret")
        results = reader.fetch_recent()

        assert results == []

    @patch("roshni.integrations.gmail.imaplib.IMAP4_SSL")
    def test_fetch_unread_only(self, mock_imap_cls):
        mock_conn = MagicMock()
        mock_imap_cls.return_value = mock_conn
        mock_conn.login.return_value = ("OK", [])
        mock_conn.select.return_value = ("OK", [b"5"])
        mock_conn.search.return_value = ("OK", [b""])

        reader = GmailReader(address="me@gmail.com", app_password="secret")
        results = reader.fetch_recent(unread_only=True)

        # Verify UNSEEN criteria was used
        mock_conn.search.assert_called_once_with(None, "UNSEEN")
        assert results == []
