from __future__ import annotations

from collections import defaultdict

import numpy as np

from .indexes import RetrievalIndex
from .models import MethodConfig, SearchHit


class Retriever:
    def __init__(
        self,
        index: RetrievalIndex,
        retrieval_depth: int = 120,
        top_routes: int = 4,
        rrf_k: int = 60,
        graph_seed_count: int = 24,
        graph_weight: float = 0.65,
    ) -> None:
        self.index = index
        self.retrieval_depth = retrieval_depth
        self.top_routes = top_routes
        self.rrf_k = rrf_k
        self.graph_seed_count = graph_seed_count
        self.graph_weight = graph_weight

    def _rrf(self, rankings: dict[str, list[tuple[int, float]]]) -> tuple[dict[int, float], dict[int, dict[str, float]]]:
        scores: dict[int, float] = defaultdict(float)
        provenance: dict[int, dict[str, float]] = defaultdict(dict)
        for source, ranking in rankings.items():
            for rank, (doc_index, raw_score) in enumerate(ranking, start=1):
                contribution = 1.0 / (self.rrf_k + rank)
                scores[doc_index] += contribution
                provenance[doc_index][source] = contribution
                provenance[doc_index][f"{source}_raw"] = float(raw_score)
        return scores, provenance

    def _expand_graph(
        self,
        scores: dict[int, float],
        provenance: dict[int, dict[str, float]],
        *,
        seed_count: int | None = None,
        neighbor_limit: int | None = None,
    ) -> None:
        applied_seed_count = self.graph_seed_count if seed_count is None else max(0, seed_count)
        seeds = sorted(scores, key=scores.get, reverse=True)[:applied_seed_count]
        bonuses: dict[int, float] = defaultdict(float)
        for rank, seed in enumerate(seeds, start=1):
            neighbors = sorted(
                (int(value) for value in self.index.graph.neighbors[seed]),
                key=lambda value: (-scores.get(value, 0.0), value),
            )
            if neighbor_limit is not None:
                neighbors = neighbors[:max(0, neighbor_limit)]
            if not len(neighbors):
                continue
            rank_discount = 1.0 / np.sqrt(rank)
            degree_discount = 1.0 / np.sqrt(len(neighbors))
            per_neighbor = self.graph_weight * scores[seed] * rank_discount * degree_discount
            for neighbor in neighbors:
                bonuses[int(neighbor)] += float(per_neighbor)
        for doc_index, bonus in bonuses.items():
            scores[doc_index] = scores.get(doc_index, 0.0) + bonus
            provenance[doc_index]["graph"] = bonus

    def search(
        self,
        query: str,
        config: MethodConfig,
        top_k: int = 20,
        *,
        graph_seed_count: int | None = None,
        graph_neighbor_limit: int | None = None,
    ) -> list[SearchHit]:
        selected_routes: list[int] | None = None
        allowed: np.ndarray | None = None
        if config.use_routing:
            selected_routes = self.index.routing.select(query, self.top_routes)
            allowed = self.index.routing.allowed_documents(selected_routes)

        bm25_ranking = self.index.bm25.search(query, self.retrieval_depth, allowed=allowed)
        if config.use_hybrid:
            dense_query = self.index.query_dense(query)
            dense_ranking = self.index.hnsw.search(
                dense_query,
                self.retrieval_depth,
                route_ids=selected_routes,
            )
            scores, provenance = self._rrf({"bm25": bm25_ranking, "hnsw": dense_ranking})
        else:
            scores, provenance = self._rrf({"routing_leaf_bm25": bm25_ranking})

        if config.use_graph:
            self._expand_graph(
                scores,
                provenance,
                seed_count=graph_seed_count,
                neighbor_limit=graph_neighbor_limit,
            )

        ordered = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [
            SearchHit(
                doc_id=self.index.doc_ids[doc_index],
                score=float(scores[doc_index]),
                provenance=provenance[doc_index],
            )
            for doc_index in ordered
        ]
