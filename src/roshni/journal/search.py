"""TF-IDF text search engine.

Provides keyword-based search over a corpus of Documents.
This is the "keyword" half of hybrid search — combine with
a vector store's semantic search for full hybrid retrieval.

Requires scikit-learn (``pip install roshni[journal]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import SearchConfig
from .models import Document, RetrievalStrategy, SearchResult

if TYPE_CHECKING:
    pass


def _require_sklearn():
    """Lazy import with clear error message."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        return TfidfVectorizer, cosine_similarity
    except ImportError:
        raise ImportError(
            "scikit-learn is required for text search. Install with: pip install roshni[journal]"
        ) from None


class TextSearcher:
    """TF-IDF keyword search over a document corpus.

    Build the index once with ``build_index()``, then search with ``search()``.
    The index can be rebuilt incrementally or from scratch.

    Example::

        searcher = TextSearcher()
        searcher.build_index(documents)
        results = searcher.search("retirement planning", top_k=10)
    """

    def __init__(self, config: SearchConfig | None = None):
        self.config = config or SearchConfig()
        self._vectorizer = None
        self._tfidf_matrix = None
        self._documents: list[Document] = []

    @property
    def is_built(self) -> bool:
        """Whether the index has been built."""
        return self._tfidf_matrix is not None

    @property
    def document_count(self) -> int:
        """Number of documents in the index."""
        return len(self._documents)

    def build_index(self, documents: list[Document]) -> None:
        """Build TF-IDF index from documents.

        Args:
            documents: List of Documents to index.
        """
        if not documents:
            return

        TfidfVectorizer, _ = _require_sklearn()

        self._documents = list(documents)
        texts = [doc.content for doc in self._documents]

        self._vectorizer = TfidfVectorizer(
            max_features=self.config.tfidf_max_features,
            stop_words="english",
            ngram_range=self.config.tfidf_ngram_range,
            min_df=self.config.tfidf_min_df,
            max_df=self.config.tfidf_max_df,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Search the index for documents matching the query.

        Args:
            query: Search query string.
            top_k: Max results to return. Defaults to config.max_results.

        Returns:
            List of SearchResult sorted by relevance (descending).
        """
        if not self.is_built:
            return []

        _, cosine_similarity = _require_sklearn()
        top_k = top_k or self.config.max_results

        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        # Get top-k indices sorted by score descending
        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score > 0:
                results.append(
                    SearchResult(
                        document=self._documents[idx],
                        score=score,
                        strategy=RetrievalStrategy.KEYWORD,
                    )
                )

        return results
