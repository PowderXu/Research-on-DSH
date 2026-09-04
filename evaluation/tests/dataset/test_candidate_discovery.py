from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dataset.scripts.candidate_discovery import (
    DiscussionScope,
    ListingCapRisk,
    audit_frozen_manifest,
    candidate_from_qapage,
    discussion_numbers_from_listing,
    ensure_below_listing_cap,
    max_listing_page,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_listing_parser_is_scope_exact_and_reads_pagination() -> None:
    page = """
    <a href="/orgs/acme/discussions/7">seven</a>
    <a href="/orgs/other/discussions/8">eight</a>
    <a href="/acme/widget/discussions/9">nine</a>
    <a href="/acme/other/discussions/10">ten</a>
    <a href="?discussions_q=is%3Aanswered&amp;page=4">last</a>
    """
    assert discussion_numbers_from_listing(
        page, DiscussionScope(organization="acme")
    ) == {7}
    assert discussion_numbers_from_listing(
        page, DiscussionScope(repository="acme/widget")
    ) == {9}
    assert max_listing_page(page) == 4


def test_listing_cap_guard_fails_closed_near_ceiling() -> None:
    ensure_below_listing_cap(949, "2025-01-01..2025-12-31")
    with pytest.raises(ListingCapRisk, match="split the date range"):
        ensure_below_listing_cap(950, "2025-01-01..2025-12-31")


def test_candidate_requires_accepted_answer_direct_exact_docs_link() -> None:
    page = """
    <script type="application/ld+json">{
      "@type": "QAPage",
      "mainEntity": {
        "@type": "Question",
        "name": "How?",
        "text": "Question",
        "acceptedAnswer": {
          "@type": "Answer",
          "url": "https://github.com/acme/widget/discussions/7#discussioncomment-2",
          "text": "<a href='https://docs.acme.test/docs/guide'>guide</a><a href='https://play.docs.acme.test/docs/demo'>demo</a><a href='https://docs.acme.test/blog/post'>blog</a>"
        }
      }
    }</script>
    """
    row = candidate_from_qapage(
        page,
        dataset="acme",
        discussion_number=7,
        source_url="https://github.com/acme/widget/discussions/7",
        docs_hosts={"docs.acme.test"},
        docs_path_prefixes=("/docs",),
    )
    assert row is not None
    assert row["docs_host_links"] == ["https://docs.acme.test/docs/guide"]


def test_offline_lineage_audit_checks_screen_and_pinned_qrels(
    tmp_path: Path,
) -> None:
    frozen = tmp_path / "discussion_sources.jsonl"
    selected = {
        "dataset": "acme",
        "discussion_number": 7,
        "source_url": "https://github.com/acme/widget/discussions/7",
        "accepted_answer_url": (
            "https://github.com/acme/widget/discussions/7#discussioncomment-2"
        ),
        "docs_host_links": ["https://docs.acme.test/docs/guide"],
        "benchmark_split": "test",
    }
    _write_jsonl(frozen, [selected])
    historical = tmp_path / "history" / "results" / "acme_yield"
    screen_rows = [
        {
            **{key: selected[key] for key in (
                "discussion_number",
                "source_url",
                "accepted_answer_url",
            )},
            "docs_host_links": [
                "https://docs.acme.test/docs/guide",
                "https://play.docs.acme.test/docs/demo",
            ],
        },
        {
            "discussion_number": 8,
            "source_url": "https://github.com/acme/widget/discussions/8",
            "accepted_answer_url": (
                "https://github.com/acme/widget/discussions/8#discussioncomment-3"
            ),
            "docs_host_links": ["https://docs.acme.test/docs/other"],
        },
    ]
    screen_path = historical / "host_link_answers.jsonl"
    _write_jsonl(screen_path, screen_rows)
    screen_sha = hashlib.sha256(screen_path.read_bytes()).hexdigest()
    project_dir = tmp_path / "projects" / "acme-project"
    _write_jsonl(project_dir / "corpus.jsonl", [{"doc_id": "/docs/guide"}])
    _write_jsonl(
        project_dir / "questions.jsonl",
        [{"source_url": selected["source_url"], "qrel_ids": ["/docs/guide"]}],
    )
    lineage = {
        "frozen_manifest": {
            "rows": 1,
            "sha256": hashlib.sha256(frozen.read_bytes()).hexdigest(),
        },
        "datasets": {
            "acme": {
                "project_id": "acme-project",
                "docs_hosts": ["docs.acme.test"],
                "docs_path_prefixes": ["/docs"],
                "frozen_rows": 1,
                "selection_reproducibility": "partial",
                "irrecoverable_step": "historical route state missing",
                "historical_host_screen": {
                    "directory": "results/acme_yield",
                    "rows_sha256": screen_sha,
                },
            }
        }
    }
    report = audit_frozen_manifest(
        frozen_manifest=frozen,
        lineage=lineage,
        project_data_root=tmp_path / "projects",
        historical_root=tmp_path / "history",
    )
    dataset = report["datasets"]["acme"]
    assert dataset["historical_host_screen"] == {
        "status": "available",
        "sha256": screen_sha,
        "expected_sha256": screen_sha,
        "hash_matches": True,
        "base_host_rows": 2,
        "exact_host_and_path_rows": 2,
        "frozen_rows_missing_from_screen": [],
        "frozen_field_mismatches_after_exact_filter": [],
        "exact_screen_rows_not_in_frozen_manifest": [8],
    }
    assert dataset["pinned_corpus_resolution"] == {
        "status": "available",
        "question_rows": 1,
        "frozen_rows_found_by_source_url": 1,
        "frozen_rows_missing_from_project_package": [],
        "project_rows_not_in_frozen_manifest": [],
        "rows_with_at_least_one_resolved_qrel": 1,
        "dangling_qrel_ids": [],
    }


def test_committed_lineage_matches_frozen_798_row_manifest() -> None:
    project_root = Path(__file__).resolve().parents[3]
    frozen = project_root / "evaluation/dataset/templates/discussion_sources.jsonl"
    lineage_path = project_root / "evaluation/dataset/templates/candidate_discovery.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    report = audit_frozen_manifest(
        frozen_manifest=frozen,
        lineage=lineage,
    )
    assert report["frozen_rows"] == 798
    assert report["frozen_manifest_hash_matches"]
    assert report["duplicate_dataset_discussion_keys"] == []
    assert report["frozen_manifest_sha256"] == lineage["frozen_manifest"]["sha256"]
    assert {
        dataset: value["frozen_rows"]
        for dataset, value in report["datasets"].items()
    } == {
        "github_docs": 328,
        "prisma": 213,
        "supabase": 90,
        "tailwind": 167,
    }
