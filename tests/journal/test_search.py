"""Tests for roshni.journal.search (TF-IDF TextSearcher)."""

import pytest

from roshni.journal.config import SearchConfig
from roshni.journal.models import Document, RetrievalStrategy
from roshni.journal.search import TextSearcher


@pytest.fixture
def sample_documents():
    return [
        Document(content="Python is a great programming language for data science", metadata={"source": "python.md"}),
        Document(content="JavaScript is used for web development and frontend", metadata={"source": "js.md"}),
        Document(
            content="Machine learning and deep learning are subfields of artificial intelligence",
            metadata={"source": "ml.md"},
        ),
        Document(content="Data science uses Python and R for statistical analysis", metadata={"source": "ds.md"}),
        Document(content="Web frameworks like Django and Flask are built with Python", metadata={"source": "web.md"}),
    ]


class TestTextSearcher:
    def test_build_index(self, sample_documents):
        searcher = TextSearcher()
        searcher.build_index(sample_documents)
        assert searcher.is_built
        assert searcher.document_count == 5

    def test_search_empty_index(self):
        searcher = TextSearcher()
        results = searcher.search("python")
        assert results == []

    def test_build_empty_list(self):
        searcher = TextSearcher()
        searcher.build_index([])
        assert not searcher.is_built

    def test_search_finds_relevant(self, sample_documents):
        searcher = TextSearcher()
        searcher.build_index(sample_documents)
        results = searcher.search("Python programming")
        assert len(results) > 0
        # Python-related docs should appear in results
        sources = [r.document.metadata["source"] for r in results]
        assert "python.md" in sources

    def test_search_returns_search_results(self, sample_documents):
        searcher = TextSearcher()
        searcher.build_index(sample_documents)
        results = searcher.search("data science")
        assert all(isinstance(r.score, float) for r in results)
        assert all(r.strategy == RetrievalStrategy.KEYWORD for r in results)

    def test_search_respects_top_k(self, sample_documents):
        searcher = TextSearcher()
        searcher.build_index(sample_documents)
        results = searcher.search("Python", top_k=2)
        assert len(results) <= 2

    def test_search_scores_descending(self, sample_documents):
        searcher = TextSearcher()
        searcher.build_index(sample_documents)
        results = searcher.search("Python")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_custom_config(self, sample_documents):
        config = SearchConfig(max_results=3, tfidf_max_features=500)
        searcher = TextSearcher(config=config)
        searcher.build_index(sample_documents)
        results = searcher.search("Python")
        assert len(results) <= 3

    def test_zero_score_excluded(self, sample_documents):
        searcher = TextSearcher()
        searcher.build_index(sample_documents)
        results = searcher.search("quantum physics blockchain")
        # All results should have positive scores
        assert all(r.score > 0 for r in results)
