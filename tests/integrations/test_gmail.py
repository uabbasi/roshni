"""Tests for roshni.integrations.gmail."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from roshni.integrations.gmail import GmailSender


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
