from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import numpy as np

from dsh_plugin.backend.service import (
    GitHubDocsPluginService,
    _corpus_revision,
    _doc_id,
    _doc_uri,
    _scoped_doc_ids,
)


def test_corpus_revision_accepts_source_and_normalized_manifests(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"doc_id":"d1"}\n', encoding="utf-8")

    assert _corpus_revision({"corpus_revision": "commit-1"}, corpus) == "commit-1"
    assert _corpus_revision({}, corpus) == (
        "sha256:" + hashlib.sha256(corpus.read_bytes()).hexdigest()
    )


def test_namespaced_document_uri_round_trip_preserves_project_id() -> None:
    doc_id = "github-docs::/repositories/releases"
    assert _doc_id(_doc_uri(doc_id)) == doc_id


def test_legacy_document_uri_round_trip_preserves_leading_slash() -> None:
    assert _doc_id(_doc_uri("/repositories/releases")) == "/repositories/releases"


def test_scope_resolves_only_exact_or_descendant_documents() -> None:
    rows = {
        "github-docs::/rest/releases": {},
        "github-docs::/rest/releases/releases": {},
        "github-docs::/rest/repos": {},
        "prisma::/rest/releases": {},
    }

    assert _scoped_doc_ids(
        "viking://resources/docsqa/github-docs::/rest/releases", rows
    ) == {
        "github-docs::/rest/releases",
        "github-docs::/rest/releases/releases",
    }
    assert _scoped_doc_ids("viking://resources/docsqa", rows) is None


def test_unknown_scope_is_rejected() -> None:
    try:
        _scoped_doc_ids(
            "viking://resources/docsqa/github-docs::/does-not-exist",
            {"github-docs::/rest/releases": {}},
        )
    except ValueError as error:
        assert "scope does not resolve" in str(error)
    else:
        raise AssertionError("unknown scope was accepted")


def _service_fixture(arm: str = "hybrid") -> GitHubDocsPluginService:
    service = object.__new__(GitHubDocsPluginService)
    service.arm = arm
    service.revision = "corpus-1"
    service.corpus_rows = [
        {
            "doc_id": "github-docs::/rest/releases",
            "title": "Releases",
            "source_path": "content/rest/releases.md",
            "rendered_text": "# Releases\n\nRelease overview.",
        },
        {
            "doc_id": "github-docs::/rest/releases/releases",
            "title": "Release endpoints",
            "source_path": "content/rest/releases/releases.md",
            "rendered_text": "# Release endpoints\n\nCreate a release.",
        },
        {
            "doc_id": "prisma::/rest/releases",
            "title": "Outside project",
            "source_path": "docs/rest/releases.md",
            "rendered_text": "Outside.",
        },
    ]
    service.rows_by_id = {
        row["doc_id"]: row for row in service.corpus_rows
    }
    service.chunks = [object(), object(), object()]
    service.events = []
    service.trace_path = None
    service._trace_lock = threading.Lock()
    service.graph_snapshot = None
    return service


def test_scoped_search_to_fetch_round_trip_is_namespaced() -> None:
    service = _service_fixture()
    observed: dict[str, object] = {}

    class Hybrid:
        @staticmethod
        def search(
            query: str,
            method: str,
            *,
            top_k: int,
            allowed_doc_ids: set[str] | None,
        ) -> tuple[list[str], dict[str, object]]:
            observed.update(
                query=query,
                method=method,
                top_k=top_k,
                allowed=allowed_doc_ids,
            )
            return sorted(allowed_doc_ids or [])[:top_k], {"scope_documents": 2}

    service.hybrid = Hybrid()
    search = service.search(
        {
            "query": "release",
            "scope": "viking://resources/docsqa/github-docs::/rest/releases",
            "result_limit": 10,
        }
    )

    assert observed["allowed"] == {
        "github-docs::/rest/releases",
        "github-docs::/rest/releases/releases",
    }
    assert all(
        row["sourceId"].startswith("github-docs::/rest/releases")
        for row in search["results"]
    )
    fetched = service.fetch({"uris": [search["results"][0]["uri"]]})
    assert fetched["results"][0]["sourceId"] == search["results"][0]["sourceId"]


def test_search_expand_fetch_workflow_preserves_seed_provenance() -> None:
    service = _service_fixture("neo4j")

    class Graph:
        @staticmethod
        def expand_from_seeds(*_: object, **__: object):
            return [
                {
                    "doc_id": "github-docs::/rest/releases/releases",
                    "score": 0.8,
                    "expanded_from": ["github-docs::/rest/releases"],
                    "path_types": ["markdown_link_1hop"],
                    "hops": 1,
                }
            ], {"max_link_hops": 2}

    service.graph = Graph()
    service._query_vector = lambda _: np.asarray([1.0, 0.0], dtype=np.float32)
    expanded = service.expand(
        {
            "query": "create a release",
            "seed_uris": [_doc_uri("github-docs::/rest/releases")],
            "result_limit": 10,
        }
    )

    result = expanded["results"][0]
    assert result["sourceId"] == "github-docs::/rest/releases/releases"
    assert result["expandedFrom"] == [
        _doc_uri("github-docs::/rest/releases")
    ]
    assert service.fetch({"uris": [result["uri"]]})["results"][0][
        "sourceId"
    ] == result["sourceId"]
