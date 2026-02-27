"""Tests for Gmail tool behavior (draft-first + optional send + check email)."""

from __future__ import annotations

import os
from unittest.mock import patch

import yaml

from roshni.agent.permissions import PermissionTier
from roshni.agent.tools.gmail_tool import create_gmail_tools
from roshni.core.config import Config
from roshni.core.secrets import SecretsManager, YamlFileProvider


def _make_secrets(tmp_dir: str, payload: dict) -> SecretsManager:
    path = os.path.join(tmp_dir, "secrets.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(payload, f)
    return SecretsManager(providers=[YamlFileProvider(path)])


def test_draft_tool_enabled_by_default(tmp_dir):
    drafts_dir = os.path.join(tmp_dir, "drafts")
    config = Config(
        data_dir=tmp_dir,
        defaults={
            "integrations": {"gmail": {"enabled": True, "allow_send": False}},
            "paths": {"email_drafts_dir": drafts_dir},
        },
    )
    secrets = _make_secrets(tmp_dir, {})

    tools = create_gmail_tools(config, secrets)
    names = {t.name for t in tools}
    assert "create_email_draft" in names
    assert "send_email" not in names

    draft_tool = next(t for t in tools if t.name == "create_email_draft")
    result = draft_tool.execute({"recipient": "test@example.com", "subject": "Hi", "body": "Hello"})
    assert "Draft saved:" in result
    assert os.listdir(drafts_dir)


def test_send_tool_only_when_allowed_and_credentials_present(tmp_dir):
    config = Config(
        data_dir=tmp_dir,
        defaults={
            "integrations": {"gmail": {"enabled": True, "allow_send": True}},
            "paths": {"email_drafts_dir": os.path.join(tmp_dir, "drafts")},
        },
    )
    secrets = _make_secrets(
        tmp_dir,
        {"gmail": {"address": "me@gmail.com", "app_password": "app-pass"}},
    )

    tools = create_gmail_tools(config, secrets, tier=PermissionTier.FULL)
    names = {t.name for t in tools}
    assert "create_email_draft" in names
    assert "send_email" in names


def test_check_email_tool_available_with_credentials(tmp_dir):
    """check_email tool should appear when gmail credentials exist."""
    config = Config(
        data_dir=tmp_dir,
        defaults={
            "integrations": {"gmail": {"enabled": True}},
            "paths": {"email_drafts_dir": os.path.join(tmp_dir, "drafts")},
        },
    )
    secrets = _make_secrets(
        tmp_dir,
        {"gmail": {"address": "me@gmail.com", "app_password": "app-pass"}},
    )

    tools = create_gmail_tools(config, secrets)
    names = {t.name for t in tools}
    assert "check_email" in names

    check_tool = next(t for t in tools if t.name == "check_email")
    assert check_tool.permission == "read"
    assert check_tool.requires_approval is False


def test_check_email_tool_missing_without_credentials(tmp_dir):
    """check_email tool should NOT appear without gmail credentials."""
    config = Config(
        data_dir=tmp_dir,
        defaults={
            "integrations": {"gmail": {"enabled": True}},
            "paths": {"email_drafts_dir": os.path.join(tmp_dir, "drafts")},
        },
    )
    secrets = _make_secrets(tmp_dir, {})

    tools = create_gmail_tools(config, secrets)
    names = {t.name for t in tools}
    assert "check_email" not in names


@patch("roshni.agent.tools.gmail_tool._check_email")
def test_check_email_tool_invokes_reader(mock_check, tmp_dir):
    """check_email tool should call _check_email with bound credentials."""
    mock_check.return_value = "**1. Test Subject**\n   From: sender@example.com\n"

    config = Config(
        data_dir=tmp_dir,
        defaults={
            "integrations": {"gmail": {"enabled": True}},
            "paths": {"email_drafts_dir": os.path.join(tmp_dir, "drafts")},
        },
    )
    secrets = _make_secrets(
        tmp_dir,
        {"gmail": {"address": "me@gmail.com", "app_password": "app-pass"}},
    )

    tools = create_gmail_tools(config, secrets)
    check_tool = next(t for t in tools if t.name == "check_email")
    result = check_tool.execute({"count": 5, "unread_only": True})
    assert "Test Subject" in result
