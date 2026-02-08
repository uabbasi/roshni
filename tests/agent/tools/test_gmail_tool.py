"""Tests for Gmail tool behavior (draft-first + optional send)."""

from __future__ import annotations

import os

import yaml

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

    tools = create_gmail_tools(config, secrets)
    names = {t.name for t in tools}
    assert "create_email_draft" in names
    assert "send_email" in names
