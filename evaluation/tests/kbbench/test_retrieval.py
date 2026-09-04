from __future__ import annotations

import numpy as np

from kbbench.retrieval import (
    GitHubDocsRetriever,
    aspect_retrieval_metrics,
    build_chunks,
    retrieval_metrics,
    rrf,
)


def test_rrf_fuses_rankings_without_score_calibration() -> None:
    ranked, scores = rrf((["a", "b"], ["b", "c"]))
    assert ranked[0] == "b"
    assert scores["b"] > scores["a"]


def test_retrieval_metrics_support_multiple_qrels() -> None:
    metrics = retrieval_metrics(["a", "x", "b"], {"a", "b"})
    assert metrics["hit_at_1"] == 1.0
    assert metrics["recall_at_1"] == 0.5
    assert metrics["recall_at_5"] == 1.0
    assert 0.0 < metrics["ndcg_at_10"] <= 1.0


def test_aspect_metrics_reward_complementary_evidence() -> None:
    aspects = [
        {
            "aspect_id": "a1",
            "weight": 0.6,
            "retrieval_doc_ids": ["d1", "d1-alternative"],
        },
        {
            "aspect_id": "a2",
            "weight": 0.4,
            "retrieval_doc_ids": ["d2"],
        },
    ]

    complementary = aspect_retrieval_metrics(["d1", "d2", "d1-alternative"], aspects)
    redundant = aspect_retrieval_metrics(["d1", "d1-alternative"], aspects)

    assert complementary["weighted_aspect_recall_at_10"] == 1.0
    assert complementary["alpha_ndcg_at_10"] == 1.0
    assert redundant["weighted_aspect_recall_at_10"] == 0.6
    assert redundant["alpha_ndcg_at_10"] < complementary["alpha_ndcg_at_10"]


def test_aspect_metrics_exclude_answer_only_aspects() -> None:
    aspects = [
        {
            "aspect_id": "retrievable",
            "weight": 0.5,
            "retrieval_doc_ids": ["local-doc"],
        },
        {
            "aspect_id": "answer-only",
            "weight": 0.5,
            "retrieval_doc_ids": [],
        },
    ]

    metrics = aspect_retrieval_metrics(["local-doc"], aspects)

    assert metrics["retrieval_eligible_aspect_count"] == 1
    assert metrics["weighted_aspect_recall_at_10"] == 1.0
    assert metrics["alpha_ndcg_at_10"] == 1.0


def test_link_expansion_reaches_two_hops_without_a_query_gate() -> None:
    rows = [
        {
            "doc_id": "/a",
            "route": "/a",
            "title": "Document A",
            "rendered_text": "Starting page",
            "outgoing_ids": ["/b"],
            "link_edges": [
                {
                    "target_id": "/b",
                    "anchor_text": "continue",
                    "context": "Continue to the intermediate page",
                }
            ],
        },
        {
            "doc_id": "/b",
            "route": "/b",
            "title": "Document B",
            "rendered_text": "Intermediate page",
            "outgoing_ids": ["/c"],
            "link_edges": [
                {
                    "target_id": "/c",
                    "anchor_text": "target evidence",
                    "context": "The target evidence is on the final page",
                }
            ],
        },
        {
            "doc_id": "/c",
            "route": "/c",
            "title": "Document C",
            "rendered_text": "Target evidence",
            "outgoing_ids": [],
            "link_edges": [],
        },
    ]
    chunks = build_chunks(rows)
    retriever = GitHubDocsRetriever(
        rows,
        chunks,
        np.asarray([[1.0, 0.0], [0.7, 0.3], [0.0, 1.0]], dtype=np.float32),
        embedding_model=None,
        reranker=None,
        graph_seed_depth=1,
        graph_max_neighbors=2,
        graph_max_hops=2,
        graph_max_candidates=4,
    )

    ranked, applied, added, by_hop = retriever._multihop_link_rank(
        "target evidence",
        ["/a"],
        {"/a": 1.0 / 61.0},
    )

    assert applied
    assert ranked.index("/c") < len(ranked)
    assert added == 2
    assert by_hop == [1, 1]


def test_chunking_covers_end_of_long_document() -> None:
    ending = "UNIQUE_ENDING_EVIDENCE"
    rows = [
        {
            "doc_id": "/guide",
            "route": "/",
            "title": "Guide",
            "rendered_text": ("first paragraph\n\n" * 300) + ending,
        }
    ]
    chunks = build_chunks(rows)
    assert len(chunks) > 1
    assert any(ending in chunk.text for chunk in chunks)


def test_hybrid_scope_filters_sparse_and_dense_candidates() -> None:
    rows = [
        {
            "doc_id": "project::/allowed/page",
            "route": "/allowed",
            "title": "Allowed",
            "rendered_text": "permitted fallback evidence",
            "link_edges": [],
        },
        {
            "doc_id": "project::/outside/page",
            "route": "/outside",
            "title": "Outside",
            "rendered_text": "needle exact semantic match",
            "link_edges": [],
        },
    ]
    chunks = build_chunks(rows)

    class Model:
        @staticmethod
        def encode(*_: object, **__: object) -> np.ndarray:
            return np.asarray([[1.0, 0.0]], dtype=np.float32)

    retriever = GitHubDocsRetriever(
        rows,
        chunks,
        np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        embedding_model=Model(),
        reranker=None,
        retrieval_depth=2,
    )

    ranked, diagnostics = retriever.search(
        "needle exact semantic match",
        "bm25_hnsw_rrf",
        top_k=10,
        allowed_doc_ids={"project::/allowed/page"},
    )

    assert ranked == ["project::/allowed/page"]
    assert diagnostics["scope_documents"] == 1
