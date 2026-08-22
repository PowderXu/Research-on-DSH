from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import hnswlib
import numpy as np

from .indexes import BM25Index


METHODS = (
    "bm25",
    "hnsw",
    "bm25_hnsw_rrf",
    "bm25_hnsw_rerank",
    "pure_routing_bm25",
    "hybrid_soft_hierarchy_rerank",
    "hybrid_conditional_graph_rerank",
    "hybrid_hierarchy_conditional_graph_rerank",
    "hybrid_edge_graph_rrf",
    "hybrid_edge_graph_rrf_always",
    "hybrid_hierarchy_edge_graph_rrf",
)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    route: str
    title: str
    text: str


def _paragraph_chunks(text: str, target_chars: int = 1_400, overlap_chars: int = 180) -> list[str]:
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    if not paragraphs:
        return []
    output: list[str] = []
    current = ""
    for paragraph in paragraphs:
        pieces = [paragraph[index : index + target_chars] for index in range(0, len(paragraph), target_chars)]
        for piece in pieces:
            if current and len(current) + len(piece) + 2 > target_chars:
                output.append(current.strip())
                tail = current[-overlap_chars:].strip()
                current = f"{tail}\n\n{piece}" if tail else piece
            else:
                current = f"{current}\n\n{piece}" if current else piece
    if current.strip():
        output.append(current.strip())
    return output


def build_chunks(corpus_rows: list[dict[str, Any]]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for row in corpus_rows:
        title = str(row["title"])
        doc_id = str(row["doc_id"])
        route = str(row["route"])
        text = str(row["rendered_text"])
        parts = _paragraph_chunks(text)
        if not parts:
            parts = [title]
        for index, part in enumerate(parts):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}::c{index:04d}",
                    doc_id=doc_id,
                    route=route,
                    title=title,
                    text=f"{title}\n{doc_id}\n{part}",
                )
            )
    return chunks


def rrf(rankings: Iterable[list[str]], constant: int = 60) -> tuple[list[str], dict[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (constant + rank)
    ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
    return ordered, dict(scores)


def retrieval_metrics(ranked_ids: list[str], relevant_ids: set[str]) -> dict[str, float]:
    output: dict[str, float] = {}
    for cutoff in (1, 5, 10, 20):
        found = sum(doc_id in relevant_ids for doc_id in ranked_ids[:cutoff])
        output[f"recall_at_{cutoff}"] = found / max(1, len(relevant_ids))
        output[f"hit_at_{cutoff}"] = float(found > 0)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_ids[:10], start=1)
        if doc_id in relevant_ids
    )
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(10, len(relevant_ids)) + 1)
    )
    output["ndcg_at_10"] = dcg / ideal if ideal else 0.0
    return output


def graph_trigger(query: str, base_scores: list[float]) -> bool:
    lowered = query.lower()
    connectors = sum(
        lowered.count(value)
        for value in (" and ", " or ", " but ", " versus ", " vs ", " while ", " between ")
    )
    procedural = sum(
        value in lowered
        for value in ("workflow", "depends", "inherit", "from ", " to ", "across", "together")
    )
    lexical_complexity = len(query.split()) >= 12 and connectors >= 1 and procedural >= 2
    uncertain = len(base_scores) >= 2 and abs(base_scores[0] - base_scores[1]) < 0.18
    return lexical_complexity or (len(query.split()) >= 45 and uncertain)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _text_hash(chunks: list[Chunk], model_name: str) -> str:
    digest = hashlib.sha256(model_name.encode())
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode())
        digest.update(chunk.text.encode())
    return digest.hexdigest()


class GitHubDocsRetriever:
    def __init__(
        self,
        corpus_rows: list[dict[str, Any]],
        chunks: list[Chunk],
        embeddings: np.ndarray,
        embedding_model: Any,
        reranker: Any,
        retrieval_depth: int = 350,
        rerank_depth: int = 45,
        graph_seed_depth: int = 8,
        graph_max_neighbors: int = 8,
        graph_weight: float = 1.0,
        hierarchy_weight: float = 0.15,
        edge_trigger_rank: int = 60,
    ) -> None:
        self.corpus_rows = corpus_rows
        self.chunks = chunks
        self.embedding_model = embedding_model
        self.reranker = reranker
        self.retrieval_depth = retrieval_depth
        self.rerank_depth = rerank_depth
        self.graph_seed_depth = graph_seed_depth
        self.graph_max_neighbors = graph_max_neighbors
        self.graph_weight = graph_weight
        self.hierarchy_weight = hierarchy_weight
        self.edge_trigger_rank = edge_trigger_rank
        self.bm25 = BM25Index([chunk.text for chunk in chunks], min_df=1)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.hnsw = hnswlib.Index(space="cosine", dim=int(self.embeddings.shape[1]))
        self.hnsw.init_index(
            max_elements=len(chunks), ef_construction=180, M=24, random_seed=17
        )
        self.hnsw.add_items(self.embeddings, np.arange(len(chunks)), num_threads=1)
        self.hnsw.set_ef(max(160, retrieval_depth))
        self.doc_to_chunks: dict[str, list[int]] = defaultdict(list)
        self.route_to_chunks: dict[str, list[int]] = defaultdict(list)
        self.page_title: dict[str, str] = {}
        self.page_route: dict[str, str] = {}
        self.outgoing: dict[str, set[str]] = defaultdict(set)
        self.incoming: dict[str, set[str]] = defaultdict(set)
        self.edge_records: list[dict[str, str]] = []
        self.doc_to_edge_indices: dict[str, list[int]] = defaultdict(list)
        for index, chunk in enumerate(chunks):
            self.doc_to_chunks[chunk.doc_id].append(index)
            self.route_to_chunks[chunk.route].append(index)
            self.page_title[chunk.doc_id] = chunk.title
            self.page_route[chunk.doc_id] = chunk.route
        for row in corpus_rows:
            source = str(row["doc_id"])
            for target in row.get("outgoing_ids", []):
                self.outgoing[source].add(str(target))
                self.incoming[str(target)].add(source)
            for edge in row.get("link_edges", []):
                target = str(edge.get("target_id", ""))
                if not target or target not in self.page_title:
                    continue
                edge_index = len(self.edge_records)
                record = {
                    "source_id": source,
                    "target_id": target,
                    "text": "\n".join(
                        value
                        for value in (
                            self.page_title.get(source, ""),
                            str(edge.get("source_section", "")).replace("-", " "),
                            str(edge.get("anchor_text", "")),
                            str(edge.get("context", "")),
                            self.page_title.get(target, ""),
                            target,
                            str(edge.get("target_anchor", "")).replace("-", " "),
                        )
                        if value
                    ),
                }
                self.edge_records.append(record)
                self.doc_to_edge_indices[source].append(edge_index)
                self.doc_to_edge_indices[target].append(edge_index)
        self.edge_bm25 = (
            BM25Index([record["text"] for record in self.edge_records], min_df=1)
            if self.edge_records
            else None
        )
        self.routes = sorted(self.route_to_chunks)
        route_texts = []
        for route in self.routes:
            titles = sorted(
                {self.chunks[index].title for index in self.route_to_chunks[route]}
            )
            route_texts.append(f"{route}\n" + "\n".join(titles))
        self.route_bm25 = BM25Index(route_texts, min_df=1, max_features=30_000)

    def _collapse_chunks(
        self, chunk_hits: list[tuple[int, float]], depth: int = 250
    ) -> tuple[list[str], dict[str, float], dict[str, int]]:
        scores: dict[str, float] = {}
        best_chunk: dict[str, int] = {}
        for chunk_index, score in chunk_hits:
            doc_id = self.chunks[chunk_index].doc_id
            if doc_id not in scores or score > scores[doc_id]:
                scores[doc_id] = float(score)
                best_chunk[doc_id] = chunk_index
        ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:depth]
        return ordered, scores, best_chunk

    def _bm25(self, query: str, allowed: np.ndarray | None = None) -> tuple[list[str], dict[str, float], dict[str, int]]:
        hits = self.bm25.search(query, self.retrieval_depth, allowed=allowed)
        return self._collapse_chunks(hits)

    def _dense(self, query_vector: np.ndarray) -> tuple[list[str], dict[str, float], dict[str, int]]:
        labels, distances = self.hnsw.knn_query(
            query_vector.reshape(1, -1),
            k=min(self.retrieval_depth, len(self.chunks)),
            num_threads=1,
        )
        hits = [
            (int(label), float(1.0 - distance))
            for label, distance in zip(labels[0], distances[0])
        ]
        return self._collapse_chunks(hits)

    def _select_routes(self, query: str, count: int = 4) -> list[str]:
        return [self.routes[index] for index, _ in self.route_bm25.search(query, count)]

    def _rerank(
        self,
        query: str,
        candidates: list[str],
        sparse_best: dict[str, int],
        dense_best: dict[str, int],
        depth: int | None = None,
    ) -> tuple[list[str], list[float]]:
        depth = depth or self.rerank_depth
        selected = candidates[:depth]
        missing = [
            doc_id
            for doc_id in selected
            if doc_id not in sparse_best and doc_id not in dense_best
        ]
        fallback_scores = self.bm25.scores(query) if missing else None
        pairs = []
        for doc_id in selected:
            if doc_id in sparse_best:
                chunk_index = sparse_best[doc_id]
            elif doc_id in dense_best:
                chunk_index = dense_best[doc_id]
            else:
                assert fallback_scores is not None
                page_chunks = self.doc_to_chunks[doc_id]
                chunk_index = max(
                    page_chunks,
                    key=lambda index: (float(fallback_scores[index]), -index),
                )
            pairs.append([query, self.chunks[chunk_index].text])
        if not pairs:
            return candidates, []
        scores = np.asarray(
            self.reranker.predict(pairs, batch_size=32, show_progress_bar=False),
            dtype=np.float32,
        ).reshape(-1)
        order = sorted(range(len(selected)), key=lambda index: (-float(scores[index]), selected[index]))
        reranked = [selected[index] for index in order]
        reranked.extend(doc_id for doc_id in candidates if doc_id not in set(selected))
        return reranked, [float(scores[index]) for index in order]

    def _graph_candidates(self, ranked: list[str]) -> tuple[list[str], int]:
        existing = set(ranked)
        additions: list[str] = []
        for doc_id in ranked[: self.graph_seed_depth]:
            neighbors = sorted(
                self.outgoing.get(doc_id, set()) | self.incoming.get(doc_id, set()),
                key=lambda value: (
                    self.page_route.get(value) != self.page_route.get(doc_id),
                    value,
                ),
            )[: self.graph_max_neighbors]
            for neighbor in neighbors:
                if neighbor not in existing:
                    existing.add(neighbor)
                    additions.append(neighbor)
        return ranked[: self.rerank_depth] + additions, len(additions)

    def _edge_graph_rank(
        self,
        query: str,
        fused: list[str],
        fused_scores: dict[str, float],
        *,
        always: bool,
        use_hierarchy: bool,
    ) -> tuple[list[str], bool, int, list[str]]:
        if self.edge_bm25 is None:
            return fused, False, 0, []
        edge_hits = self.edge_bm25.search(
            query, min(1_500, len(self.edge_records))
        )
        edge_rank = {edge_index: rank for rank, (edge_index, _) in enumerate(edge_hits, 1)}
        seeds = fused[: self.graph_seed_depth]
        incident_edges = {
            edge_index
            for seed in seeds
            for edge_index in self.doc_to_edge_indices.get(seed, [])
        }
        best_incident_rank = min(
            (edge_rank.get(edge_index, len(self.edge_records) + 1) for edge_index in incident_edges),
            default=len(self.edge_records) + 1,
        )
        # RRF scores live on a much smaller scale than cross-encoder logits, so
        # passing them to graph_trigger's margin test would label nearly every
        # long query as uncertain. V2 uses lexical complexity plus edge-context
        # evidence instead.
        should_expand = always or (
            len(query.split()) >= 8 and best_incident_rank <= self.edge_trigger_rank
        )
        selected_routes = self._select_routes(query) if use_hierarchy else []
        scores = dict(fused_scores)
        if use_hierarchy:
            for route_rank, route in enumerate(selected_routes, 1):
                boost = self.hierarchy_weight / (60.0 + route_rank)
                for doc_id in fused:
                    if self.page_route.get(doc_id) == route:
                        scores[doc_id] += boost
        if not should_expand:
            return (
                sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id)),
                False,
                0,
                selected_routes,
            )
        additions: set[str] = set()
        graph_boosts: dict[str, float] = defaultdict(float)
        seed_rank = {doc_id: rank for rank, doc_id in enumerate(seeds, 1)}
        for seed in seeds:
            candidates: list[tuple[int, int, str]] = []
            for edge_index in self.doc_to_edge_indices.get(seed, []):
                edge = self.edge_records[edge_index]
                neighbor = (
                    edge["target_id"]
                    if edge["source_id"] == seed
                    else edge["source_id"]
                )
                candidates.append(
                    (
                        edge_rank.get(edge_index, len(self.edge_records) + 1),
                        edge_index,
                        neighbor,
                    )
                )
            candidates.sort(key=lambda value: (value[0], value[2]))
            seen_neighbors: set[str] = set()
            for current_edge_rank, _, neighbor in candidates:
                if neighbor in seen_neighbors:
                    continue
                seen_neighbors.add(neighbor)
                if len(seen_neighbors) > self.graph_max_neighbors:
                    break
                edge_component = (
                    1.0 / (60.0 + current_edge_rank)
                    if current_edge_rank <= len(self.edge_records)
                    else 0.0
                )
                seed_component = 1.0 / (60.0 + seed_rank[seed])
                contribution = self.graph_weight * (seed_component + edge_component)
                graph_boosts[neighbor] = max(graph_boosts[neighbor], contribution)
                if neighbor not in scores:
                    additions.add(neighbor)
                    scores[neighbor] = 0.0
        for doc_id, boost in graph_boosts.items():
            scores[doc_id] += boost
        ranked = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))
        return ranked, True, len(additions), selected_routes

    def search(self, query: str, method: str, top_k: int = 20) -> tuple[list[str], dict[str, Any]]:
        started = time.perf_counter()
        query_vector = None
        if method != "bm25" and method != "pure_routing_bm25":
            query_vector = np.asarray(
                self.embedding_model.encode(
                    [query], normalize_embeddings=True, show_progress_bar=False
                )[0],
                dtype=np.float32,
            )
        graph_applied = False
        graph_added = 0
        selected_routes: list[str] = []

        if method == "pure_routing_bm25":
            selected_routes = self._select_routes(query)
            allowed = np.asarray(
                sorted(
                    {
                        index
                        for route in selected_routes
                        for index in self.route_to_chunks[route]
                    }
                ),
                dtype=np.int64,
            )
            ranked, _, _ = self._bm25(query, allowed=allowed)
        else:
            sparse_ranked, _, sparse_best = self._bm25(query)
            if method == "bm25":
                ranked = sparse_ranked
            else:
                assert query_vector is not None
                dense_ranked, _, dense_best = self._dense(query_vector)
                if method == "hnsw":
                    ranked = dense_ranked
                else:
                    fused, fused_scores = rrf((sparse_ranked, dense_ranked))
                    if method == "bm25_hnsw_rrf":
                        ranked = fused
                    elif method in {
                        "hybrid_edge_graph_rrf",
                        "hybrid_edge_graph_rrf_always",
                        "hybrid_hierarchy_edge_graph_rrf",
                    }:
                        ranked, graph_applied, graph_added, selected_routes = self._edge_graph_rank(
                            query,
                            fused,
                            fused_scores,
                            always=method == "hybrid_edge_graph_rrf_always",
                            use_hierarchy=method == "hybrid_hierarchy_edge_graph_rrf",
                        )
                    else:
                        if "hierarchy" in method:
                            selected_routes = self._select_routes(query)
                            route_set = set(selected_routes)
                            fused = sorted(
                                fused,
                                key=lambda doc_id: (
                                    self.page_route.get(doc_id) not in route_set,
                                    fused.index(doc_id),
                                ),
                            )
                        base_ranked, base_scores = self._rerank(
                            query, fused, sparse_best, dense_best
                        )
                        ranked = base_ranked
                        if "graph" in method and graph_trigger(query, base_scores):
                            graph_applied = True
                            graph_pool, graph_added = self._graph_candidates(base_ranked)
                            ranked, _ = self._rerank(
                                query,
                                graph_pool,
                                sparse_best,
                                dense_best,
                                depth=len(graph_pool),
                            )
                            ranked.extend(
                                doc_id for doc_id in base_ranked if doc_id not in set(ranked)
                            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        return ranked[:top_k], {
            "latency_ms": latency_ms,
            "graph_applied": graph_applied,
            "graph_candidates_added": graph_added,
            "selected_routes": selected_routes,
        }


def build_or_load_embeddings(
    chunks: list[Chunk], model: Any, model_name: str, cache_dir: Path
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _text_hash(chunks, model_name)
    vectors_path = cache_dir / f"chunks_{fingerprint[:16]}.npy"
    if vectors_path.exists():
        return np.load(vectors_path)
    vectors = np.asarray(
        model.encode(
            [chunk.text for chunk in chunks],
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )
    np.save(vectors_path, vectors)
    (cache_dir / f"chunks_{fingerprint[:16]}.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "fingerprint": fingerprint,
                "chunks": len(chunks),
                "dimension": int(vectors.shape[1]),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return vectors


def summarize(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row[key]) for key in group_keys)].append(row)
    output: list[dict[str, Any]] = []
    for group, values in sorted(grouped.items()):
        item: dict[str, Any] = dict(zip(group_keys, group))
        item["questions"] = len(values)
        for metric in (
            "recall_at_5",
            "recall_at_10",
            "recall_at_20",
            "hit_at_1",
            "hit_at_5",
            "hit_at_10",
            "ndcg_at_10",
        ):
            item[metric] = statistics.fmean(float(row[metric]) for row in values)
        item["latency_p50_ms"] = float(
            np.percentile([float(row["latency_ms"]) for row in values], 50)
        )
        item["graph_application_rate"] = statistics.fmean(
            float(row["graph_applied"]) for row in values
        )
        output.append(item)
    return output


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    corpus_rows = _load_jsonl(args.dataset_dir / "corpus.jsonl")
    questions = _load_jsonl(args.dataset_dir / "questions.jsonl")
    if args.split != "all":
        questions = [question for question in questions if question["split"] == args.split]
    if args.limit:
        questions = questions[: args.limit]
    chunks = build_chunks(corpus_rows)
    print(f"Loading embedding model {args.embedding_model}", flush=True)
    embedding_model = SentenceTransformer(
        args.embedding_model, device=args.device, local_files_only=args.local_files_only
    )
    embeddings = build_or_load_embeddings(
        chunks, embedding_model, args.embedding_model, args.cache_dir
    )
    methods = tuple(args.methods) if args.methods else METHODS
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    reranker = None
    if any("rerank" in method for method in methods):
        from sentence_transformers import CrossEncoder

        print(f"Loading reranker {args.reranker_model}", flush=True)
        reranker = CrossEncoder(
            args.reranker_model, device=args.device, local_files_only=args.local_files_only
        )
    retriever = GitHubDocsRetriever(
        corpus_rows,
        chunks,
        embeddings,
        embedding_model,
        reranker,
        retrieval_depth=args.retrieval_depth,
        rerank_depth=args.rerank_depth,
        graph_seed_depth=args.graph_seed_depth,
        graph_max_neighbors=args.graph_max_neighbors,
        graph_weight=args.graph_weight,
        hierarchy_weight=args.hierarchy_weight,
        edge_trigger_rank=args.edge_trigger_rank,
    )
    retriever.search(questions[0]["query"], methods[0], top_k=args.top_k)
    rows: list[dict[str, Any]] = []
    for method in methods:
        print(f"Evaluating {method} on {len(questions)} questions", flush=True)
        for index, question in enumerate(questions, start=1):
            ranked_ids, diagnostics = retriever.search(
                str(question["query"]), method, top_k=args.top_k
            )
            rows.append(
                {
                    "iteration": args.iteration,
                    "schema_version": args.schema_version,
                    "method": method,
                    "question_id": question["question_id"],
                    "split": question["split"],
                    "intent_category": question["intent_category"],
                    "evidence_category": question["evidence_category"],
                    "evidence_structure": question.get(
                        "evidence_structure",
                        "single" if len(set(question["qrel_ids"])) == 1
                        else "linked" if question["evidence_category"] == "multi_page_linked"
                        else "dispersed",
                    ),
                    "qrel_count": int(
                        question.get("qrel_count") or len(set(question["qrel_ids"]))
                    ),
                    "qrel_count_group": "1"
                    if len(set(question["qrel_ids"])) == 1 else "2+",
                    "relevant_ids": question["qrel_ids"],
                    "ranked_ids": ranked_ids,
                    **diagnostics,
                    **retrieval_metrics(ranked_ids, set(question["qrel_ids"])),
                }
            )
            if index % 50 == 0:
                print(f"  {index}/{len(questions)}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_query.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    test_rows = [row for row in rows if row["split"] == "test"]
    report = {
        "iteration": args.iteration,
        "schema_version": args.schema_version,
        "documents": len(corpus_rows),
        "chunks": len(chunks),
        "questions": len(questions),
        "methods": list(methods),
        "split_filter": args.split,
        "graph_weight": args.graph_weight,
        "hierarchy_weight": args.hierarchy_weight,
        "edge_trigger_rank": args.edge_trigger_rank,
        "test_overall": summarize(test_rows, ("method",)),
        "evaluated_overall": summarize(rows, ("method",)),
        "test_by_evidence_category": summarize(
            test_rows, ("evidence_category", "method")
        ),
        "test_by_intent_category": summarize(test_rows, ("intent_category", "method")),
        "test_by_evidence_structure": summarize(
            test_rows, ("evidence_structure", "method")
        ),
        "test_by_qrel_count": summarize(
            test_rows, ("qrel_count_group", "method")
        ),
        "dev_by_evidence_category": summarize(
            [row for row in rows if row["split"] == "dev"],
            ("evidence_category", "method"),
        ),
        "latency_scope": "Warm per-query retrieval including query embedding and reranking; index build excluded.",
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on real GitHub Docs questions")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--iteration", default="v1")
    parser.add_argument("--schema-version", default="v1")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--retrieval-depth", type=int, default=350)
    parser.add_argument("--rerank-depth", type=int, default=45)
    parser.add_argument("--graph-seed-depth", type=int, default=8)
    parser.add_argument("--graph-max-neighbors", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--split", choices=("all", "dev", "test"), default="all")
    parser.add_argument("--graph-weight", type=float, default=1.0)
    parser.add_argument("--hierarchy-weight", type=float, default=0.15)
    parser.add_argument("--edge-trigger-rank", type=int, default=60)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--methods", nargs="*")
    args = parser.parse_args()
    report = evaluate(args)
    print(json.dumps(report["test_overall"], indent=2))


if __name__ == "__main__":
    main()
