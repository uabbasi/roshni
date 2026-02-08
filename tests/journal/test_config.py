"""Tests for roshni.journal.config."""

from roshni.journal.config import ProcessingConfig, SearchConfig


class TestProcessingConfig:
    def test_defaults(self):
        config = ProcessingConfig()
        assert config.chunk_size == 2000
        assert config.chunk_overlap == 400
        assert ".md" in config.file_extensions
        assert "chats" in config.skip_directories

    def test_custom_values(self):
        config = ProcessingConfig(chunk_size=1000, chunk_overlap=200)
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200


class TestSearchConfig:
    def test_defaults(self):
        config = SearchConfig()
        assert config.max_results == 25
        assert config.similarity_threshold == 0.7
        assert config.semantic_weight == 0.7
        assert config.tfidf_ngram_range == (1, 2)

    def test_custom_values(self):
        config = SearchConfig(max_results=10, semantic_weight=0.5)
        assert config.max_results == 10
        assert config.semantic_weight == 0.5
