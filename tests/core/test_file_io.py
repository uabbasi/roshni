"""Tests for roshni.core.utils.file_io."""

import os

import pytest

from roshni.core.utils.file_io import (
    backup_file,
    extract_markdown_sections,
    parse_date_heading,
    parse_frontmatter,
    safe_move,
    safe_write,
    safe_write_with_backup,
    update_frontmatter,
)


class TestSafeWrite:
    def test_creates_file(self, tmp_dir):
        path = os.path.join(tmp_dir, "sub", "file.txt")
        safe_write(path, "hello")
        with open(path) as f:
            assert f.read() == "hello"

    def test_creates_parent_dirs(self, tmp_dir):
        path = os.path.join(tmp_dir, "a", "b", "c.txt")
        safe_write(path, "nested")
        assert os.path.exists(path)


class TestFrontmatter:
    def test_parse(self):
        content = "---\ntitle: Test\ntags: [a, b]\n---\n\nBody text"
        fm, body = parse_frontmatter(content)
        assert fm["title"] == "Test"
        assert "a" in fm["tags"]
        assert body == "Body text"

    def test_no_frontmatter(self):
        content = "# Just a heading\n\nContent"
        fm, body = parse_frontmatter(content)
        assert fm == {}
        assert body == content

    def test_obsidian_tags(self):
        content = "---\ntags:\n  - #daily\n  - #health\n---\n\nBody"
        fm, _body = parse_frontmatter(content)
        assert "#daily" in fm["tags"]

    def test_update_adds_new_key(self):
        content = "---\ntitle: Old\n---\n\nBody"
        result = update_frontmatter(content, {"author": "test"})
        fm, _ = parse_frontmatter(result)
        assert fm["author"] == "test"
        assert fm["title"] == "Old"

    def test_update_merges_lists(self):
        content = "---\ntags: [a, b]\n---\n\nBody"
        result = update_frontmatter(content, {"tags": ["c"]})
        fm, _ = parse_frontmatter(result)
        assert set(fm["tags"]) == {"a", "b", "c"}


class TestMarkdownSections:
    def test_extract_h2(self):
        content = "## Section 1\nContent 1\n\n## Section 2\nContent 2"
        sections = extract_markdown_sections(content, section_level=2)
        assert "Section 1" in sections
        assert "Section 2" in sections


class TestDateParsing:
    def test_frontmatter_date(self):
        content = "---\ndate: 2024-06-15\n---\n\nBody"
        d = parse_date_heading(content)
        assert d is not None
        assert d.isoformat() == "2024-06-15"

    def test_heading_date(self):
        content = "# 2024-03-01\n\nBody"
        d = parse_date_heading(content)
        assert d is not None
        assert d.isoformat() == "2024-03-01"

    def test_no_date(self):
        assert parse_date_heading("No date here") is None


class TestBackup:
    def test_backup_creates_copy(self, tmp_dir):
        path = os.path.join(tmp_dir, "file.txt")
        safe_write(path, "original")
        backup_path = backup_file(path)
        assert backup_path is not None
        assert os.path.exists(backup_path)
        with open(backup_path) as f:
            assert f.read() == "original"

    def test_backup_nonexistent(self, tmp_dir):
        assert backup_file(os.path.join(tmp_dir, "nope.txt")) is None

    def test_safe_write_with_backup(self, tmp_dir):
        path = os.path.join(tmp_dir, "file.txt")
        safe_write(path, "v1")
        backup_path = safe_write_with_backup(path, "v2")
        assert backup_path is not None
        with open(path) as f:
            assert f.read() == "v2"
        with open(backup_path) as f:
            assert f.read() == "v1"


class TestSafeMove:
    def test_move(self, tmp_dir):
        src = os.path.join(tmp_dir, "src.txt")
        safe_write(src, "data")
        dest_dir = os.path.join(tmp_dir, "dest")
        new_path = safe_move(src, dest_dir)
        assert os.path.exists(new_path)
        assert not os.path.exists(src)

    def test_move_nonexistent_raises(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            safe_move(os.path.join(tmp_dir, "nope"), tmp_dir)
