from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import hnswlib
import numpy as np
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import normalize

from .models import Document


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
        self.idf = np.log1p((n_docs - document_frequency + 0.5) / (document_frequency + 0.5))

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


class DenseHNSWIndex:
    def __init__(self, vectors: np.ndarray, ef_search: int = 96) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.dim = int(self.vectors.shape[1])
        self.ef_search = ef_search
        self.global_index = self._build(self.vectors, np.arange(len(self.vectors), dtype=np.int64))
        self.route_indices: dict[int, tuple[hnswlib.Index, np.ndarray]] = {}

    def _build(self, vectors: np.ndarray, labels: np.ndarray) -> hnswlib.Index:
        index = hnswlib.Index(space="cosine", dim=self.dim)
        index.init_index(
            max_elements=max(1, len(labels)),
            ef_construction=160,
            M=16,
            random_seed=17,
        )
        if len(labels):
            # hnswlib's parallel insertion order can vary across runs even with
            # a fixed random seed. Single-threaded construction makes locked
            # benchmark artifacts reproducible.
            index.add_items(vectors, labels, num_threads=1)
        index.set_ef(max(self.ef_search, 20))
        return index

    def add_route_indices(self, route_labels: np.ndarray) -> None:
        self.route_indices.clear()
        for route_id in sorted(int(value) for value in np.unique(route_labels)):
            doc_indices = np.flatnonzero(route_labels == route_id).astype(np.int64)
            route_index = self._build(self.vectors[doc_indices], doc_indices)
            self.route_indices[route_id] = (route_index, doc_indices)

    @staticmethod
    def _query_one(index: hnswlib.Index, vector: np.ndarray, k: int, size: int) -> list[tuple[int, float]]:
        if k <= 0 or size <= 0 or not np.any(vector):
            return []
        labels, distances = index.knn_query(
            vector.reshape(1, -1),
            k=min(k, size),
            num_threads=1,
        )
        return [
            (int(label), float(1.0 - distance))
            for label, distance in zip(labels[0], distances[0])
        ]

    def search(
        self,
        vector: np.ndarray,
        k: int,
        route_ids: Iterable[int] | None = None,
    ) -> list[tuple[int, float]]:
        if route_ids is None:
            return self._query_one(self.global_index, vector, k, len(self.vectors))
        merged: list[tuple[int, float]] = []
        for route_id in route_ids:
            pair = self.route_indices.get(int(route_id))
            if pair is None:
                continue
            index, doc_indices = pair
            merged.extend(self._query_one(index, vector, k, len(doc_indices)))
        merged.sort(key=lambda item: item[1], reverse=True)
        return merged[:k]


@dataclass
class RoutingLayer:
    labels: np.ndarray
    summaries: list[str]
    keywords: list[list[str]]
    representatives: list[list[int]]
    selector: TfidfVectorizer
    selector_matrix: sparse.csr_matrix

    @property
    def route_count(self) -> int:
        return len(self.summaries)

    def select(self, query: str, top_routes: int) -> list[int]:
        query_vector = self.selector.transform([query])
        scores = (self.selector_matrix @ query_vector.T).toarray().ravel()
        if not np.any(scores):
            # A deterministic fallback avoids query/qrel leakage.
            sizes = np.bincount(self.labels, minlength=self.route_count)
            return [int(index) for index in np.argsort(sizes)[::-1][:top_routes]]
        top = np.argsort(scores)[::-1][: min(top_routes, len(scores))]
        return [int(index) for index in top]

    def allowed_documents(self, route_ids: Iterable[int]) -> np.ndarray:
        route_ids_array = np.fromiter((int(value) for value in route_ids), dtype=np.int32)
        return np.flatnonzero(np.isin(self.labels, route_ids_array)).astype(np.int64)

    def write_markdown(self, output_dir: Path, documents: list[Document]) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        root_lines = [
            "# ROUTING",
            "",
            "This file is a generated, corpus-only routing layer. Choose the best route by its keywords and representative titles, then inspect the matching child manifest.",
            "",
        ]
        for route_id, summary in enumerate(self.summaries):
            size = int(np.sum(self.labels == route_id))
            root_lines.extend(
                [
                    f"- [route_{route_id:03d}.md](routes/route_{route_id:03d}.md) ({size} files)",
                    f"  - {summary}",
                ]
            )
        (output_dir / "ROUTING.md").write_text("\n".join(root_lines) + "\n", encoding="utf-8")
        routes_dir = output_dir / "routes"
        routes_dir.mkdir(exist_ok=True)
        for route_id in range(self.route_count):
            lines = [f"# Route {route_id:03d}", "", self.summaries[route_id], "", "## Files", ""]
            for doc_index in np.flatnonzero(self.labels == route_id):
                document = documents[int(doc_index)]
                safe_title = document.title.replace("\n", " ").strip()
                lines.append(f"- `{document.doc_id}` — {safe_title}")
            (routes_dir / f"route_{route_id:03d}.md").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )


class StaticDocumentGraph:
    ID_PATTERN = re.compile(r"\b(?:swg|nas)[a-zA-Z0-9]+\b", re.IGNORECASE)

    def __init__(self, documents: list[Document], max_degree: int = 64) -> None:
        started = time.perf_counter()
        stem_to_index = {Path(document.doc_id).stem.lower(): index for index, document in enumerate(documents)}
        neighbor_sets: list[set[int]] = [set() for _ in documents]
        for source_index, document in enumerate(documents):
            for raw_reference in self.ID_PATTERN.findall(document.text):
                target_index = stem_to_index.get(raw_reference.lower())
                if target_index is None or target_index == source_index:
                    continue
                neighbor_sets[source_index].add(target_index)
                neighbor_sets[target_index].add(source_index)
        # Apply the degree cap as an undirected greedy edge selection so the
        # stored graph remains symmetric and edge statistics are exact.
        edges = sorted(
            (source, target)
            for source, values in enumerate(neighbor_sets)
            for target in values
            if source < target
        )
        capped_sets: list[set[int]] = [set() for _ in documents]
        for source, target in edges:
            if len(capped_sets[source]) >= max_degree or len(capped_sets[target]) >= max_degree:
                continue
            capped_sets[source].add(target)
            capped_sets[target].add(source)
        self.neighbors = [np.asarray(sorted(values), dtype=np.int64) for values in capped_sets]
        self.edge_count = len(edges) if not edges else sum(len(values) for values in self.neighbors) // 2
        self.non_isolated = sum(bool(len(values)) for values in self.neighbors)
        self.build_seconds = time.perf_counter() - started


class RetrievalIndex:
    def __init__(
        self,
        documents: list[Document],
        dense_dimensions: int = 96,
        route_count: int = 48,
        random_seed: int = 17,
    ) -> None:
        self.documents = documents
        self.doc_ids = [document.doc_id for document in documents]
        self.id_to_index = {doc_id: index for index, doc_id in enumerate(self.doc_ids)}
        index_texts = [document.index_text for document in documents]
        started = time.perf_counter()
        # Tiny fixtures need min_df=1; the public corpus uses min_df=2 to bound
        # the vocabulary and remove one-off OCR/noise tokens.
        self.bm25 = BM25Index(index_texts, min_df=1 if len(documents) < 100 else 2)

        self.tfidf = TfidfTransformer(sublinear_tf=True, norm="l2")
        tfidf_matrix = self.tfidf.fit_transform(self.bm25.counts)
        max_dimensions = max(2, min(dense_dimensions, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=max_dimensions, n_iter=7, random_state=random_seed)
        dense_vectors = self.svd.fit_transform(tfidf_matrix).astype(np.float32)
        self.dense_vectors = normalize(dense_vectors, norm="l2", copy=False).astype(np.float32)

        actual_routes = max(2, min(route_count, len(documents)))
        clusterer = MiniBatchKMeans(
            n_clusters=actual_routes,
            batch_size=2048,
            n_init=5,
            max_iter=200,
            random_state=random_seed,
        )
        labels = clusterer.fit_predict(self.dense_vectors).astype(np.int32)
        feature_names = self.bm25.vectorizer.get_feature_names_out()
        reconstructed = clusterer.cluster_centers_ @ self.svd.components_
        keywords: list[list[str]] = []
        representatives: list[list[int]] = []
        summaries: list[str] = []
        for route_id in range(actual_routes):
            top_terms = np.argsort(reconstructed[route_id])[::-1][:18]
            route_keywords = [str(feature_names[index]) for index in top_terms if reconstructed[route_id, index] > 0]
            keywords.append(route_keywords)
            members = np.flatnonzero(labels == route_id)
            member_scores = self.dense_vectors[members] @ clusterer.cluster_centers_[route_id]
            representative_indices = members[np.argsort(member_scores)[::-1][:5]].astype(int).tolist()
            representatives.append(representative_indices)
            titles = [documents[index].title.replace("\n", " ") for index in representative_indices]
            summaries.append(
                "Keywords: " + ", ".join(route_keywords) + ". Representative titles: " + " | ".join(titles)
            )
        selector = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True)
        selector_matrix = selector.fit_transform(summaries).tocsr()
        self.routing = RoutingLayer(labels, summaries, keywords, representatives, selector, selector_matrix)

        self.hnsw = DenseHNSWIndex(self.dense_vectors)
        self.hnsw.add_route_indices(labels)
        self.graph = StaticDocumentGraph(documents)
        self.build_seconds = time.perf_counter() - started

    def query_dense(self, query: str) -> np.ndarray:
        counts = self.bm25.vectorizer.transform([query])
        tfidf = self.tfidf.transform(counts)
        vector = self.svd.transform(tfidf).astype(np.float32)
        vector = normalize(vector, norm="l2", copy=False)
        return vector[0]
