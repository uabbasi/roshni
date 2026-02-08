"""Tests for vault tools — people, projects, ideas, search."""

from roshni.agent.permissions import PermissionTier
from roshni.agent.tools.vault_tools import create_vault_tools
from roshni.agent.vault import VaultManager


def _make_vault(tmp_dir):
    vm = VaultManager(tmp_dir, agent_dir="test")
    vm.scaffold()
    return vm


class TestCreateVaultTools:
    def test_returns_all_tools_at_interact(self, tmp_dir):
        vm = _make_vault(tmp_dir)
        tools = create_vault_tools(vm, tier=PermissionTier.INTERACT)
        names = {t.name for t in tools}
        # Read tools always present
        assert "list_people" in names
        assert "get_person" in names
        assert "search_people" in names
        assert "list_projects" in names
        assert "get_project" in names
        assert "list_ideas" in names
        assert "search_ideas" in names
        assert "search_vault_all" in names
        # Write tools present at INTERACT
        assert "save_person" in names
        assert "save_project" in names
        assert "save_idea" in names

    def test_observe_excludes_write_tools(self, tmp_dir):
        vm = _make_vault(tmp_dir)
        tools = create_vault_tools(vm, tier=PermissionTier.OBSERVE)
        names = {t.name for t in tools}
        assert "list_people" in names
        assert "save_person" not in names
        assert "save_project" not in names
        assert "save_idea" not in names

    def test_none_returns_empty(self, tmp_dir):
        vm = _make_vault(tmp_dir)
        tools = create_vault_tools(vm, tier=PermissionTier.NONE)
        assert tools == []


class TestPeopleTools:
    def test_list_people_empty(self, tmp_dir):
        vm = _make_vault(tmp_dir)
        tools = {t.name: t for t in create_vault_tools(vm)}
        result = tools["list_people"].execute({})
        assert "No entries found" in result

    def test_save_and_get_person(self, tmp_dir):
        vm = _make_vault(tmp_dir)
        tools = {t.name: t for t in create_vault_tools(vm)}
        result = tools["save_person"].execute({"name": "Alice Smith", "notes": "Works at Acme"})
        assert "Saved" in result

        result = tools["get_person"].execute({"name": "alice-smith"})
        assert "Works at Acme" in result

    def test_save_person_with_tags(self, tmp_dir):
        vm = _make_vault(tmp_dir)
        tools = {t.name: t for t in create_vault_tools(vm)}
        tools["save_person"].execute({"name": "Bob", "notes": "A friend", "tags": ["friend", "neighbor"]})
        result = tools["get_person"].execute({"name": "bob"})
        assert "friend" in result

    def test_list_people_after_save(self, tmp_dir):
        vm = _make_vault(tmp_dir)
        tools = {t.name: t for t in create_vault_tools(vm)}
        tools["save_person"].execute({"name": "Charlie", "notes": "test"})
        result = tools["list_people"].execute({})
        assert "charlie" in result

    def test_search_people(self, tmp_dir):
        vm = _make_vault(tmp_dir)
        tools = {t.name: t for t in create_vault_tools(vm)}
        tools["save_person"].execute({"name": "Dana", "notes": "Works at Acme Corp"})
        result = tools["search_people"].execute({"query": "Acme"})
        assert "dana" in result.lower()

    def test_get_person_not_found(self, tmp_dir):
        vm = _make_vault(tmp_dir)
        tools = {t.name: t for t in create_vault_tools(vm)}
        result = tools["get_person"].execute({"name": "nonexistent"})
        assert "Not found" in result


class TestProjectTools:
    def test_save_and_get_project(self, tmp_dir):
        vm = _make_vault(tmp_dir)
        tools = {t.name: t for t in create_vault_tools(vm)}
        result = tools["save_project"].execute({"title": "My Project", "notes": "Building something"})
        assert "Saved" in result

        result = tools["get_project"].execute({"name": "my-project"})
        assert "Building something" in result

    def test_list_projects(self, tmp_dir):
        vm = _make_vault(tmp_dir)
        tools = {t.name: t for t in create_vault_tools(vm)}
        tools["save_project"].execute({"title": "Alpha", "notes": "First project"})
        result = tools["list_projects"].execute({})
        assert "alpha" in result


class TestIdeaTools:
    def test_save_and_list_ideas(self, tmp_dir):
        vm = _make_vault(tmp_dir)
        tools = {t.name: t for t in create_vault_tools(vm)}
        result = tools["save_idea"].execute({"title": "Robot Butler", "notes": "Automate chores"})
        assert "Saved" in result

        result = tools["list_ideas"].execute({})
        assert "robot-butler" in result

    def test_search_ideas(self, tmp_dir):
        vm = _make_vault(tmp_dir)
        tools = {t.name: t for t in create_vault_tools(vm)}
        tools["save_idea"].execute({"title": "Solar Panels", "notes": "Green energy for home"})
        result = tools["search_ideas"].execute({"query": "energy"})
        assert "solar" in result.lower()


class TestSearchVaultAll:
    def test_cross_section_search(self, tmp_dir):
        vm = _make_vault(tmp_dir)
        tools = {t.name: t for t in create_vault_tools(vm)}
        tools["save_person"].execute({"name": "Eve", "notes": "Researches quantum computing"})
        tools["save_idea"].execute({"title": "Quantum App", "notes": "Use quantum computing APIs"})

        result = tools["search_vault_all"].execute({"query": "quantum"})
        assert "people" in result.lower() or "ideas" in result.lower()

    def test_search_no_match(self, tmp_dir):
        vm = _make_vault(tmp_dir)
        tools = {t.name: t for t in create_vault_tools(vm)}
        result = tools["search_vault_all"].execute({"query": "xyznotfound"})
        assert "No results" in result

    def test_audit_log_on_save(self, tmp_dir):
        vm = _make_vault(tmp_dir)
        tools = {t.name: t for t in create_vault_tools(vm)}
        tools["save_person"].execute({"name": "Frank", "notes": "test"})
        audit = (vm.admin_dir / "audit.md").read_text()
        assert "save_person" in audit
        assert "Frank" in audit
