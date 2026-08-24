from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer


def _top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or scores.size == 0:
        return np.empty(0, dtype=np.int64)
    finite = np.flatnonzero(np.isfinite(scores) & (scores > 0))
    if finite.size == 0:
        return np.empty(0, dtype=np.int64)
    k = min(k, finite.size)
    local = np.argpartition(scores[finite], -k)[-k:]
    chosen = finite[local]
    return chosen[np.argsort(scores[chosen])[::-1]]


class BM25Index:
    """Sparse BM25 index used by the GitHub Docs retrieval backends."""

    def __init__(
        self,
        texts: list[str],
        max_features: int = 60_000,
        min_df: int = 2,
        max_df: float = 1.0,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.vectorizer = CountVectorizer(
            lowercase=True,
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z0-9][a-zA-Z0-9_.:/-]+\b",
            ngram_range=(1, 2),
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            dtype=np.float32,
        )
        counts = self.vectorizer.fit_transform(texts).tocsr()
        self.counts = counts
        doc_lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
        avg_length = float(doc_lengths.mean()) or 1.0
        document_frequency = np.diff(counts.tocsc().indptr).astype(np.float32)
        n_docs = counts.shape[0]
        self.idf = np.log1p(
            (n_docs - document_frequency + 0.5) / (document_frequency + 0.5)
        )

        weighted = counts.copy().astype(np.float32)
        row_norm = k1 * (1.0 - b + b * doc_lengths / avg_length)
        repeated_norm = np.repeat(row_norm, np.diff(weighted.indptr))
        weighted.data = (
            weighted.data
            * (k1 + 1.0)
            / (weighted.data + repeated_norm)
            * self.idf[weighted.indices]
        ).astype(np.float32)
        self.weighted = weighted.tocsr()

    def query_vector(self, query: str) -> sparse.csr_matrix:
        counts = self.vectorizer.transform([query]).tocsr()
        if counts.nnz:
            counts.data[:] = 1.0
        return counts

    def scores(self, query: str) -> np.ndarray:
        query_vector = self.query_vector(query)
        if query_vector.nnz == 0:
            return np.zeros(self.weighted.shape[0], dtype=np.float32)
        return (self.weighted @ query_vector.T).toarray().ravel().astype(np.float32)

    def search(
        self,
        query: str,
        k: int,
        allowed: np.ndarray | None = None,
    ) -> list[tuple[int, float]]:
        scores = self.scores(query)
        if allowed is not None:
            mask = np.zeros(scores.shape[0], dtype=bool)
            mask[allowed] = True
            scores[~mask] = -np.inf
        indices = _top_indices(scores, k)
        return [(int(index), float(scores[index])) for index in indices]
