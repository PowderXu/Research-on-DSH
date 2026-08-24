from __future__ import annotations

from kbbench.retrieval import (
    build_chunks,
    graph_trigger,
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


def test_graph_trigger_is_query_only_plus_score_uncertainty() -> None:
    assert graph_trigger(
        "How do I connect a reusable workflow and inherit secrets across an organization while deploying?",
        [0.5, 0.4],
    )
    assert not graph_trigger("How do I delete a repository?", [0.9, 0.1])


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
