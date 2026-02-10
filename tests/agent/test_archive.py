"""Tests for pre-compaction archival in DefaultAgent."""

import os

import pytest

from roshni.agent.default import DefaultAgent
from roshni.core.config import Config
from roshni.core.secrets import SecretsManager


@pytest.fixture
def config(tmp_dir):
    return Config(
        data_dir=tmp_dir,
        defaults={"llm": {"provider": "openai", "model": "gpt-5.2-chat-latest"}},
    )


@pytest.fixture
def secrets():
    return SecretsManager(providers=[])


class TestArchiveConversation:
    def test_archive_creates_file(self, config, secrets, tmp_dir):
        archive_dir = os.path.join(tmp_dir, "archive")
        agent = DefaultAgent(config=config, secrets=secrets, archive_dir=archive_dir)

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        agent._archive_conversation(messages)

        # Should have created a .md file
        files = os.listdir(archive_dir)
        assert len(files) == 1
        assert files[0].endswith(".md")

        # Verify content format
        content = open(os.path.join(archive_dir, files[0])).read()
        assert "## user" in content
        assert "Hello" in content
        assert "## assistant" in content
        assert "Hi there!" in content

    def test_archive_skipped_when_no_dir(self, config, secrets):
        agent = DefaultAgent(config=config, secrets=secrets, archive_dir=None)
        # Should not raise
        agent._archive_conversation([{"role": "user", "content": "test"}])

    def test_archive_handles_write_error_gracefully(self, config, secrets):
        agent = DefaultAgent(config=config, secrets=secrets, archive_dir="/nonexistent/path/that/fails")
        # Should log warning, not raise
        agent._archive_conversation([{"role": "user", "content": "test"}])

    def test_archive_creates_parent_dirs(self, config, secrets, tmp_dir):
        archive_dir = os.path.join(tmp_dir, "deep", "nested", "archive")
        agent = DefaultAgent(config=config, secrets=secrets, archive_dir=archive_dir)
        agent._archive_conversation([{"role": "user", "content": "test"}])
        assert os.path.isdir(archive_dir)

    def test_archive_includes_session_id(self, config, secrets, tmp_dir):
        archive_dir = os.path.join(tmp_dir, "archive")
        agent = DefaultAgent(config=config, secrets=secrets, archive_dir=archive_dir)
        agent._active_session_id = "abc12345-full-session-id"
        agent._archive_conversation([{"role": "user", "content": "test"}])

        files = os.listdir(archive_dir)
        assert len(files) == 1
        assert "abc12345" in files[0]
