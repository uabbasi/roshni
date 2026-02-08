"""Tests for roshni.journal.models."""

import pytest

from roshni.journal.models import Document, RetrievalStrategy, SearchResult


class TestDocument:
    def test_create_basic(self):
        doc = Document(content="Hello world", metadata={"source": "test.md"})
        assert doc.content == "Hello world"
        assert doc.metadata["source"] == "test.md"

    def test_create_empty_metadata(self):
        doc = Document(content="Hello world")
        assert doc.metadata == {}

    def test_empty_content_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            Document(content="", metadata={})

    def test_none_content_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            Document(content=None, metadata={})

    def test_non_dict_metadata_raises(self):
        with pytest.raises(ValueError, match="dictionary"):
            Document(content="hello", metadata="not a dict")

    def test_repr_short_content(self):
        doc = Document(content="short", metadata={"source": "file.md"})
        assert "file.md" in repr(doc)
        assert "short" in repr(doc)

    def test_repr_long_content_truncated(self):
        doc = Document(content="x" * 100, metadata={"source": "file.md"})
        assert "..." in repr(doc)


class TestSearchResult:
    def test_create(self):
        doc = Document(content="test content", metadata={"source": "test.md"})
        result = SearchResult(document=doc, score=0.95)
        assert result.score == 0.95
        assert result.strategy == RetrievalStrategy.KEYWORD

    def test_repr(self):
        doc = Document(content="test", metadata={"source": "file.md"})
        result = SearchResult(document=doc, score=0.85, strategy=RetrievalStrategy.HYBRID)
        assert "file.md" in repr(result)
        assert "0.850" in repr(result)


class TestRetrievalStrategy:
    def test_all_strategies_exist(self):
        strategies = [s.value for s in RetrievalStrategy]
        assert "semantic" in strategies
        assert "keyword" in strategies
        assert "hybrid" in strategies
        assert "mmr" in strategies
        assert "auto" in strategies
