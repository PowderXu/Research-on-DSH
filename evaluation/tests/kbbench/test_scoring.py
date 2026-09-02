import json
from pathlib import Path

import pytest

from kbbench.scoring import GitHubDocsSourceResolver, score_ranked_sources


def test_source_resolver_accepts_doc_ids_urls_and_repo_paths(tmp_path: Path) -> None:
    doc_id = "github-docs::/graphql/guides/using-the-graphql-api-for-discussions"
    source_id = "/graphql/guides/using-the-graphql-api-for-discussions"
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        json.dumps(
            {
                "doc_id": doc_id,
                "project": "github-docs",
                "source_doc_id": source_id,
                "source_path": f"github-docs/content{source_id}.md",
                "repository_source_path": f"content{source_id}.md",
                "route": doc_id,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    resolver = GitHubDocsSourceResolver(corpus_path)
    assert resolver.resolve(doc_id) == doc_id
    assert resolver.resolve(f"https://docs.github.com/en{source_id}#x") == doc_id
    assert resolver.resolve(f"github-docs/content{source_id}.md") == doc_id


def test_source_resolver_prefers_exact_canonical_id_over_route_collision(
    tmp_path: Path,
) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    rows = [
        {
            "doc_id": "github-docs::/rest/releases",
            "project": "github-docs",
            "source_doc_id": "/rest/releases",
            "source_path": "github-docs/content/rest/releases/index.md",
            "repository_source_path": "content/rest/releases/index.md",
            "route": "github-docs::/rest/releases",
        },
        {
            "doc_id": "github-docs::/rest/releases/assets",
            "project": "github-docs",
            "source_doc_id": "/rest/releases/assets",
            "source_path": "github-docs/content/rest/releases/assets.md",
            "repository_source_path": "content/rest/releases/assets.md",
            "route": "github-docs::/rest/releases",
        },
        {
            "doc_id": "github-docs::/rest/releases/releases",
            "project": "github-docs",
            "source_doc_id": "/rest/releases/releases",
            "source_path": "github-docs/content/rest/releases/releases.md",
            "repository_source_path": "content/rest/releases/releases.md",
            "route": "github-docs::/rest/releases",
        },
    ]
    corpus_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    resolver = GitHubDocsSourceResolver(corpus_path)

    assert "github-docs::/rest/releases" not in resolver.aliases
    assert (
        resolver.resolve("github-docs::/rest/releases")
        == "github-docs::/rest/releases"
    )
    assert (
        resolver.resolve("github-docs::/rest/releases/releases")
        == "github-docs::/rest/releases/releases"
    )
    assert (
        resolver.resolve("github-docs/content/rest/releases/assets.md")
        == "github-docs::/rest/releases/assets"
    )


def test_source_resolver_keeps_noncanonical_colliding_alias_unresolved(
    tmp_path: Path,
) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    rows = [
        {
            "doc_id": f"github-docs::/rest/releases/{name}",
            "project": "github-docs",
            "source_doc_id": f"/rest/releases/{name}",
            "source_path": f"github-docs/content/rest/releases/{name}.md",
            "repository_source_path": f"content/rest/releases/{name}.md",
            "route": "github-docs::/rest/shared-route",
        }
        for name in ("assets", "releases")
    ]
    corpus_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    resolver = GitHubDocsSourceResolver(corpus_path)

    assert resolver.resolve("github-docs::/rest/shared-route") is None


def test_score_rewards_rank_and_complete_multi_page_coverage() -> None:
    metrics = score_ranked_sources(["/b", "/a"], ["/a", "/b"])
    assert metrics["hit_at_1"] == 1.0
    assert metrics["recall_at_1"] == 0.5
    assert metrics["recall_at_10"] == 1.0
    assert metrics["all_support_at_10"] == 1.0
    assert metrics["ndcg_at_10"] == pytest.approx(1.0)
