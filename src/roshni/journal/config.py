"""Configuration dataclasses for journal processing and search.

These are pure data containers with sensible defaults.
Override them from YAML config, env vars, or constructor args.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProcessingConfig:
    """Settings for document chunking and ingestion.

    Attributes:
        chunk_size: Target characters per chunk.
        chunk_overlap: Overlap between adjacent chunks for context continuity.
        file_extensions: Which file types to process.
        skip_directories: Directory names to skip during recursive walks.
    """

    chunk_size: int = 2000
    chunk_overlap: int = 400
    file_extensions: list[str] = field(default_factory=lambda: [".md", ".txt"])
    skip_directories: list[str] = field(default_factory=lambda: ["chats", "retro-chats"])


@dataclass
class SearchConfig:
    """Settings for search and retrieval.

    Attributes:
        max_results: Maximum documents to return per query.
        similarity_threshold: Minimum score for semantic results.
        tfidf_max_features: Vocabulary cap for TF-IDF vectorizer.
        tfidf_ngram_range: N-gram range (min, max) for TF-IDF.
        tfidf_min_df: Minimum document frequency for TF-IDF terms.
        tfidf_max_df: Maximum document frequency ratio for TF-IDF terms.
        semantic_weight: Weight for semantic scores in hybrid search (0-1).
    """

    max_results: int = 25
    similarity_threshold: float = 0.7
    tfidf_max_features: int = 10000
    tfidf_ngram_range: tuple[int, int] = (1, 2)
    tfidf_min_df: int = 2
    tfidf_max_df: float = 0.85
    semantic_weight: float = 0.7  # 70% semantic, 30% keyword in hybrid
