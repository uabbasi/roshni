"""Tests for VaultManager."""

import os

from roshni.agent.vault import VaultManager


class TestInit:
    def test_default_agent_dir(self, tmp_dir):
        vm = VaultManager(tmp_dir)
        assert vm.vault_path.as_posix().endswith(os.path.basename(tmp_dir))
        assert vm.agent_dir == "jarvis"

    def test_custom_agent_dir(self, tmp_dir):
        vm = VaultManager(tmp_dir, agent_dir="atlas")
        assert vm.agent_dir == "atlas"


class TestDirectoryProperties:
    def test_base_dir(self, tmp_dir):
        vm = VaultManager(tmp_dir, agent_dir="bot")
        assert vm.base_dir == vm.vault_path / "bot"

    def test_subdirectories(self, tmp_dir):
        vm = VaultManager(tmp_dir, agent_dir="bot")
        assert vm.persona_dir == vm.base_dir / "persona"
        assert vm.memory_dir == vm.base_dir / "memory"
        assert vm.tasks_dir == vm.base_dir / "tasks"
        assert vm.projects_dir == vm.base_dir / "projects"
        assert vm.people_dir == vm.base_dir / "people"
        assert vm.ideas_dir == vm.base_dir / "ideas"
        assert vm.admin_dir == vm.base_dir / "admin"


class TestScaffold:
    def test_creates_all_directories(self, tmp_dir):
        vm = VaultManager(tmp_dir)
        vm.scaffold()
        for subdir in ("persona", "memory", "tasks", "projects", "people", "ideas", "admin"):
            assert (vm.base_dir / subdir).is_dir()

    def test_creates_archive(self, tmp_dir):
        vm = VaultManager(tmp_dir)
        vm.scaffold()
        assert (vm.tasks_dir / "_archive").is_dir()

    def test_creates_index(self, tmp_dir):
        vm = VaultManager(tmp_dir)
        vm.scaffold()
        index = vm.tasks_dir / "_index.md"
        assert index.exists()
        assert "Task Index" in index.read_text()

    def test_creates_audit(self, tmp_dir):
        vm = VaultManager(tmp_dir)
        vm.scaffold()
        audit = vm.admin_dir / "audit.md"
        assert audit.exists()
        assert "Audit Log" in audit.read_text()

    def test_idempotent(self, tmp_dir):
        vm = VaultManager(tmp_dir)
        vm.scaffold()
        vm.scaffold()  # should not raise
        assert (vm.tasks_dir / "_index.md").exists()


class TestLogAction:
    def test_appends_entry(self, tmp_dir):
        vm = VaultManager(tmp_dir)
        vm.scaffold()
        vm.log_action("save", "save_person", "name=Alice")
        audit = (vm.admin_dir / "audit.md").read_text()
        assert "**save**" in audit
        assert "`save_person`" in audit
        assert "name=Alice" in audit

    def test_creates_audit_if_missing(self, tmp_dir):
        vm = VaultManager(tmp_dir)
        # Don't scaffold — log_action should still work
        vm.log_action("test", "test_tool")
        assert (vm.admin_dir / "audit.md").exists()


class TestSearchAll:
    def test_finds_matching_files(self, tmp_dir):
        vm = VaultManager(tmp_dir)
        vm.scaffold()
        # Create a test person file
        (vm.people_dir / "alice.md").write_text("---\nname: Alice\n---\nWorks at Acme Corp\n")
        (vm.ideas_dir / "robot.md").write_text("---\ntitle: Robot\n---\nBuild a robot\n")

        results = vm.search_all("alice")
        assert len(results) == 1
        assert results[0]["section"] == "people"
        assert "alice.md" in results[0]["path"]

    def test_empty_query_returns_empty(self, tmp_dir):
        vm = VaultManager(tmp_dir)
        vm.scaffold()
        assert vm.search_all("") == []

    def test_no_match_returns_empty(self, tmp_dir):
        vm = VaultManager(tmp_dir)
        vm.scaffold()
        (vm.people_dir / "bob.md").write_text("Bob is great\n")
        assert vm.search_all("zzzznotfound") == []

    def test_respects_limit(self, tmp_dir):
        vm = VaultManager(tmp_dir)
        vm.scaffold()
        for i in range(5):
            (vm.people_dir / f"person{i}.md").write_text(f"Person {i} is a keyword match\n")
        results = vm.search_all("keyword", limit=2)
        assert len(results) == 2
