from pathlib import Path

import pytest

from kbbench.scoring import GitHubDocsSourceResolver, score_ranked_sources


DATA = Path(__file__).resolve().parents[1] / "dataset/data"


def test_source_resolver_accepts_doc_ids_urls_and_repo_paths() -> None:
    resolver = GitHubDocsSourceResolver(DATA / "corpus.jsonl")
    doc_id = "/graphql/guides/using-the-graphql-api-for-discussions"
    assert resolver.resolve(doc_id) == doc_id
    assert resolver.resolve(f"https://docs.github.com/en{doc_id}#x") == doc_id
    assert resolver.resolve(f"content{doc_id}.md") == doc_id


def test_score_rewards_rank_and_complete_multi_page_coverage() -> None:
    metrics = score_ranked_sources(["/b", "/a"], ["/a", "/b"])
    assert metrics["hit_at_1"] == 1.0
    assert metrics["recall_at_1"] == 0.5
    assert metrics["recall_at_10"] == 1.0
    assert metrics["all_support_at_10"] == 1.0
    assert metrics["ndcg_at_10"] == pytest.approx(1.0)
