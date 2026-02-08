"""Core data models for journal/document systems.

Provides framework-agnostic document and search result types.
No dependency on LangChain or any specific vector store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RetrievalStrategy(Enum):
    """Retrieval strategies for document search."""

    SEMANTIC = "semantic"  # Pure semantic similarity (embeddings)
    KEYWORD = "keyword"  # Pure keyword/TF-IDF search
    HYBRID = "hybrid"  # Combined semantic + keyword
    MMR = "mmr"  # Maximal Marginal Relevance (diversity)
    RERANKED = "reranked"  # Re-ranked with cross-encoder
    AUTO = "auto"  # Automatically select best strategy


@dataclass
class Document:
    """A document with content and metadata.

    Framework-agnostic representation. Converters to/from
    LangChain, LlamaIndex, etc. live in the consumer code.

    Attributes:
        content: The text content of the document.
        metadata: Arbitrary metadata (source path, date, section, etc.).
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.content or not isinstance(self.content, str):
            raise ValueError("Content must be a non-empty string")
        if not isinstance(self.metadata, dict):
            raise ValueError("Metadata must be a dictionary")

    def __repr__(self) -> str:
        source = self.metadata.get("source", "unknown")
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document(source='{source}', content='{preview}')"


@dataclass
class SearchResult:
    """A search result with relevance score.

    Attributes:
        document: The matched document.
        score: Relevance score (higher = more relevant, scale varies by strategy).
        strategy: Which retrieval strategy produced this result.
    """

    document: Document
    score: float
    strategy: RetrievalStrategy = RetrievalStrategy.KEYWORD

    def __repr__(self) -> str:
        source = self.document.metadata.get("source", "unknown")
        return f"SearchResult(source='{source}', score={self.score:.3f})"
