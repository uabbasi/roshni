"""Tests for the CLI entry point."""

from click.testing import CliRunner

from roshni.core.cli import main


class TestCliGroup:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Roshni" in result.output
        assert "init" in result.output
        assert "run" in result.output
        assert "chat" in result.output

    def test_version(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestInitCommand:
    def test_init_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["init", "--help"])
        assert result.exit_code == 0
        assert "Set up" in result.output


class TestRunCommand:
    def test_run_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "Telegram" in result.output


class TestChatCommand:
    def test_chat_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["chat", "--help"])
        assert result.exit_code == 0
        assert "terminal" in result.output.lower()
