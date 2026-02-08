"""Journal processing and search framework.

Provides document models, a JournalStore protocol for pluggable
backends, TF-IDF text search, and configuration dataclasses.
"""

from .config import ProcessingConfig, SearchConfig
from .models import Document, RetrievalStrategy, SearchResult
from .store import JournalStore

__all__ = [
    "Document",
    "JournalStore",
    "ProcessingConfig",
    "RetrievalStrategy",
    "SearchConfig",
    "SearchResult",
]
