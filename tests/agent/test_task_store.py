"""Tests for the task store."""

import threading
from datetime import datetime, timedelta

import pytest

from roshni.agent.task_store import TaskPriority, TaskStatus, TaskStore


@pytest.fixture
def tasks_dir(tmp_path):
    d = tmp_path / "tasks"
    d.mkdir()
    return d


@pytest.fixture
def store(tasks_dir):
    return TaskStore(tasks_dir)


class TestCreate:
    def test_create_basic(self, store, tasks_dir):
        t = store.create("Buy groceries")
        assert t.title == "Buy groceries"
        assert t.status == TaskStatus.OPEN
        assert t.priority == TaskPriority.MEDIUM
        # File should exist
        files = [f for f in tasks_dir.glob("*.md") if not f.name.startswith("_")]
        assert len(files) == 1

    def test_create_with_options(self, store):
        t = store.create(
            "Fix bug",
            description="Crash on startup",
            priority="high",
            project="roshni",
            tags=["bug", "urgent"],
            due="2026-03-01T00:00:00",
        )
        assert t.priority == TaskPriority.HIGH
        assert t.project == "roshni"
        assert t.tags == ["bug", "urgent"]
        assert t.due is not None
        assert t.body == "Crash on startup"

    def test_id_format(self, store):
        t = store.create("Test task")
        today = datetime.now().strftime("%Y%m%d")
        assert t.id.startswith(f"t-{today}-")
        # Should be zero-padded
        seq = t.id.split("-")[-1]
        assert len(seq) == 3
        assert seq == "001"

    def test_sequential_ids(self, store):
        t1 = store.create("First")
        t2 = store.create("Second")
        # Both should have same date prefix, sequential numbers
        assert t1.id.endswith("-001")
        assert t2.id.endswith("-002")

    def test_creates_dir_if_missing(self, tmp_path):
        new_dir = tmp_path / "nonexistent" / "tasks"
        s = TaskStore(new_dir)
        t = s.create("Auto-created dir")
        assert new_dir.is_dir()
        assert t.title == "Auto-created dir"

    def test_concurrent_create_produces_unique_ids(self, store):
        created_ids: list[str] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def _create(i: int) -> None:
            try:
                t = store.create(f"Task {i}")
                with lock:
                    created_ids.append(t.id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_create, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(created_ids) == 20
        assert len(set(created_ids)) == 20


class TestGet:
    def test_get_existing(self, store):
        created = store.create("Find me")
        found = store.get(created.id)
        assert found is not None
        assert found.id == created.id
        assert found.title == "Find me"

    def test_get_missing(self, store):
        assert store.get("t-00000000-999") is None


class TestUpdate:
    def test_update_title(self, store):
        t = store.create("Original")
        updated = store.update(t.id, title="Renamed")
        assert updated.title == "Renamed"
        # Re-read from disk
        reloaded = store.get(t.id)
        assert reloaded.title == "Renamed"

    def test_update_updates_timestamp(self, store):
        t = store.create("Timestamped")
        original_updated = t.updated
        updated = store.update(t.id, title="Changed")
        assert updated.updated >= original_updated

    def test_update_description(self, store):
        t = store.create("Has body", description="original text")
        updated = store.update(t.id, description="new text")
        assert updated.body == "new text"

    def test_update_missing_raises(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.update("t-00000000-999", title="Nope")

    def test_update_unknown_field_raises(self, store):
        t = store.create("Fields")
        with pytest.raises(ValueError, match="Unknown"):
            store.update(t.id, nonexistent="value")


class TestDelete:
    def test_delete_existing(self, store, tasks_dir):
        t = store.create("Delete me")
        assert store.delete(t.id) is True
        assert store.get(t.id) is None
        files = [f for f in tasks_dir.glob("*.md") if not f.name.startswith("_")]
        assert len(files) == 0

    def test_delete_missing(self, store):
        assert store.delete("t-00000000-999") is False


class TestListTasks:
    def test_list_all(self, store):
        store.create("A")
        store.create("B")
        store.create("C")
        tasks = store.list_tasks()
        assert len(tasks) == 3

    def test_filter_by_status(self, store):
        store.create("Open task")
        t2 = store.create("Will progress")
        store.transition(t2.id, "in_progress")
        tasks = store.list_tasks(status="in_progress")
        assert len(tasks) == 1
        assert tasks[0].id == t2.id

    def test_filter_by_project(self, store):
        store.create("A", project="alpha")
        store.create("B", project="beta")
        tasks = store.list_tasks(project="alpha")
        assert len(tasks) == 1
        assert tasks[0].project == "alpha"

    def test_filter_by_tag(self, store):
        store.create("Tagged", tags=["important"])
        store.create("Untagged")
        tasks = store.list_tasks(tag="important")
        assert len(tasks) == 1
        assert "important" in tasks[0].tags

    def test_limit(self, store):
        for i in range(5):
            store.create(f"Task {i}")
        tasks = store.list_tasks(limit=3)
        assert len(tasks) == 3

    def test_empty_dir(self, tmp_path):
        s = TaskStore(tmp_path / "empty")
        assert s.list_tasks() == []


class TestSearch:
    def test_search_title(self, store):
        store.create("Buy milk")
        store.create("Fix server")
        results = store.search("milk")
        assert len(results) == 1
        assert results[0].title == "Buy milk"

    def test_search_body(self, store):
        store.create("Task one", description="Contains keyword foobar in body")
        store.create("Task two", description="Nothing special")
        results = store.search("foobar")
        assert len(results) == 1

    def test_search_tags(self, store):
        store.create("Tagged", tags=["finance"])
        store.create("Other")
        results = store.search("finance")
        assert len(results) == 1

    def test_search_no_match(self, store):
        store.create("Something")
        assert store.search("zzzznotfound") == []


class TestTransition:
    def test_open_to_in_progress(self, store):
        t = store.create("Start me")
        result = store.transition(t.id, "in_progress")
        assert result.status == TaskStatus.IN_PROGRESS

    def test_in_progress_to_done(self, store):
        t = store.create("Finish me")
        store.transition(t.id, "in_progress")
        result = store.transition(t.id, "done")
        assert result.status == TaskStatus.DONE

    def test_done_to_open_reopen(self, store):
        t = store.create("Reopen me")
        store.transition(t.id, "in_progress")
        store.transition(t.id, "done")
        result = store.transition(t.id, "open")
        assert result.status == TaskStatus.OPEN

    def test_invalid_open_to_done(self, store):
        t = store.create("Skip ahead")
        with pytest.raises(ValueError, match="Invalid transition"):
            store.transition(t.id, "done")

    def test_invalid_done_to_in_progress(self, store):
        t = store.create("Direct progress")
        store.transition(t.id, "in_progress")
        store.transition(t.id, "done")
        with pytest.raises(ValueError, match="Invalid transition"):
            store.transition(t.id, "in_progress")

    def test_transition_missing_task(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.transition("t-00000000-999", "done")


class TestGetActionable:
    def test_no_deps(self, store):
        store.create("Free task")
        actionable = store.get_actionable()
        assert len(actionable) == 1

    def test_unresolved_dep_excluded(self, store):
        t1 = store.create("Blocker")
        t2 = store.create("Blocked")
        store.update(t2.id, depends_on=[t1.id])
        actionable = store.get_actionable()
        # Only t1 should be actionable
        ids = [t.id for t in actionable]
        assert t1.id in ids
        assert t2.id not in ids

    def test_resolved_dep_included(self, store):
        t1 = store.create("Blocker")
        t2 = store.create("Blocked")
        store.update(t2.id, depends_on=[t1.id])
        store.transition(t1.id, "in_progress")
        store.transition(t1.id, "done")
        actionable = store.get_actionable()
        ids = [t.id for t in actionable]
        assert t2.id in ids


class TestSummarizeCompleted:
    def test_archive_old_done(self, store, tasks_dir):
        t = store.create("Old done task")
        store.transition(t.id, "in_progress")
        store.transition(t.id, "done")
        # Manually backdate the updated timestamp (YAML quotes ISO strings)
        path = next(p for p in tasks_dir.glob("*.md") if not p.name.startswith("_"))
        content = path.read_text()
        current_ts = store.get(t.id).updated.isoformat(timespec="seconds")
        old_date = (datetime.now() - timedelta(days=60)).isoformat(timespec="seconds")
        # YAML may quote the value with single quotes
        content = content.replace(f"'{current_ts}'", f"'{old_date}'")
        content = content.replace(f"updated: {current_ts}", f"updated: {old_date}")
        path.write_text(content)

        result = store.summarize_completed(older_than_days=30)
        assert "Archived 1" in result
        # File should be in _archive
        archive_files = list((tasks_dir / "_archive").glob("*.md"))
        assert len(archive_files) == 1

    def test_recent_done_not_archived(self, store):
        t = store.create("Recent done")
        store.transition(t.id, "in_progress")
        store.transition(t.id, "done")
        result = store.summarize_completed(older_than_days=30)
        assert "No completed tasks" in result

    def test_empty_dir(self, tmp_path):
        s = TaskStore(tmp_path / "empty")
        result = s.summarize_completed()
        assert "No tasks directory" in result


class TestRebuildIndex:
    def test_generates_index(self, store, tasks_dir):
        store.create("Open task")
        t2 = store.create("Done task")
        store.transition(t2.id, "in_progress")
        store.transition(t2.id, "done")
        store.rebuild_index()

        index = tasks_dir / "_index.md"
        assert index.exists()
        content = index.read_text()
        assert "Task Index" in content
        assert "**open**: 1" in content
        assert "**done**: 1" in content

    def test_index_empty(self, store, tasks_dir):
        store.rebuild_index()
        index = tasks_dir / "_index.md"
        assert index.exists()
        content = index.read_text()
        assert "total**: 0" in content
