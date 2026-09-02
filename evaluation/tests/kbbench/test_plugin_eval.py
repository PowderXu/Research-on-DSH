from __future__ import annotations

import argparse
import subprocess

import numpy as np

from kbbench.plugin_eval import (
    DshFilesystemSearch,
    Neo4jGitHubDocsGraphRAG,
    _stable_evaluation_config,
    _parse_rg_matches,
    canonical_sha256,
    extract_code_entities,
    file_bundle_identity,
    graph_snapshot_verification,
    fuse_exact_hybrid_with_graph,
    markdown_tree_identity,
    sanitize_argv,
    select_fs_terms,
    text_enriched_query,
)
from kbbench.retrieval import Chunk


def test_canonical_sha256_is_order_independent_for_mapping_keys() -> None:
    assert canonical_sha256({"b": 2, "a": [1, 3]}) == canonical_sha256(
        {"a": [1, 3], "b": 2}
    )
    assert canonical_sha256({"a": [1, 3]}) != canonical_sha256({"a": [3, 1]})


def test_runtime_bundle_uses_logical_paths_and_detects_missing_files(tmp_path) -> None:
    source = tmp_path / "host-specific" / "source.py"
    source.parent.mkdir()
    source.write_text("print('stable')\n", encoding="utf-8")

    identity = file_bundle_identity(
        [
            (source, "runtime/source.py"),
            (tmp_path / "absent.lock", "runtime/absent.lock"),
        ]
    )

    assert identity["files"][0]["path"] == "runtime/source.py"
    assert str(tmp_path) not in str(identity)
    assert identity["missing"] == ["runtime/absent.lock"]
    assert len(identity["sha256"]) == 64


def test_markdown_tree_fingerprint_covers_paths_and_bytes_only(tmp_path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "page.md").write_text("first", encoding="utf-8")
    (tmp_path / "ignored.txt").write_text("ignored", encoding="utf-8")
    initial = markdown_tree_identity(tmp_path)

    (tmp_path / "ignored.txt").write_text("changed", encoding="utf-8")
    ignored_change = markdown_tree_identity(tmp_path)
    (tmp_path / "nested" / "page.md").write_text("second", encoding="utf-8")
    markdown_change = markdown_tree_identity(tmp_path)

    assert initial["files"] == 1
    assert initial["sha256"] == ignored_change["sha256"]
    assert initial["sha256"] != markdown_change["sha256"]
    assert str(tmp_path) not in str(initial)


def test_invocation_redacts_both_neo4j_password_forms() -> None:
    assert sanitize_argv(
        [
            "--neo4j-password",
            "secret-one",
            "--neo4j-password=secret-two",
            "--neo4j-uri",
            "bolt://neo4j:uri-secret@127.0.0.1:7688",
            "--split",
            "test",
        ]
    ) == [
        "--neo4j-password",
        "<redacted>",
        "--neo4j-password=<redacted>",
        "--neo4j-uri",
        "bolt://<redacted>@127.0.0.1:7688",
        "--split",
        "test",
    ]


def test_graph_snapshot_verification_requires_exact_snapshot_sha() -> None:
    expected = {"snapshot_sha256": "abc", "kggen_sha256": "kg"}

    assert graph_snapshot_verification(expected, expected)["status"] == "match"
    mismatch = graph_snapshot_verification(
        expected, {"snapshot_sha256": "different", "kggen_sha256": "kg"}
    )
    assert mismatch["status"] == "mismatch"
    assert mismatch["matches_expected"] is False
    assert graph_snapshot_verification(expected, None)["status"] == (
        "missing_observed_snapshot"
    )


def test_stable_config_covers_candidate_policy_without_credentials() -> None:
    args = argparse.Namespace(
        dataset_name="docsqa",
        corpus_revision="revision",
        arms=["dsh_bm25_hnsw", "dsh_neo4j_graphrag"],
        split="test",
        limit=None,
        top_k=10,
        retrieval_depth=50,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
        local_files_only=True,
        fs_max_calls=3,
        fs_max_matches=250,
        graph_seed_count=5,
        max_graph_candidates_per_seed=10,
        max_entity_degree=20,
        max_route_pages=20,
        neo4j_uri="bolt://neo4j:uri-secret@127.0.0.1:7688",
        neo4j_database="neo4j",
        neo4j_password="must-not-appear",
        ingest=False,
        aspects=None,
    )

    config = _stable_evaluation_config(args)

    assert config["retrieval_depth"] == 50
    assert config["graph_seed_count"] == 5
    assert config["max_graph_candidates_per_seed"] == 10
    assert config["hnsw"] == {
        "space": "cosine",
        "m": 24,
        "ef_construction": 180,
        "ef_search": 160,
        "random_seed": 17,
        "build_threads": 1,
        "query_threads": 1,
    }
    assert config["rrf_constant"] == 60
    assert config["neo4j_uri"] == "bolt://<redacted>@127.0.0.1:7688"
    assert "uri-secret" not in str(config)
    assert "password" not in str(config).casefold()


def test_fs_planner_prefers_exact_identifiers_and_is_bounded() -> None:
    terms = select_fs_terms(
        "How can I use `--ignore-revs-file` when automated formatting changes blame?",
        max_terms=3,
    )
    assert terms[0] == "--ignore-revs-file"
    assert len(terms) == 3


def test_parse_rg_matches_keeps_dsh_visible_cap() -> None:
    rows = "\n".join(
        [
            '{"type":"begin","data":{}}',
            '{"type":"match","data":{"path":{"text":"content/a.md"},"lines":{"text":"first\\n"},"line_number":2}}',
            '{"type":"match","data":{"path":{"text":"content/b.md"},"lines":{"text":"second\\n"},"line_number":4}}',
        ]
    )
    matches = _parse_rg_matches(rows, limit=1)
    assert matches == [{"path": "content/a.md", "line": "first", "line_number": 2}]


def test_fs_search_includes_explicitly_scoped_gitignored_docs(
    monkeypatch, tmp_path
) -> None:
    content_root = tmp_path / "content"
    content_root.mkdir()
    (content_root / "page.md").write_text("repository settings", encoding="utf-8")
    rg_path = tmp_path / "rg"
    rg_path.touch()
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 1, "", "")

    monkeypatch.setattr("kbbench.plugin_eval.subprocess.run", fake_run)
    filesystem = DshFilesystemSearch(
        tmp_path,
        [{"source_path": "page.md", "doc_id": "/page"}],
        rg_path,
        content_root=content_root,
    )

    filesystem.search("repository settings", top_k=10)

    assert calls
    assert all("--no-ignore" in command for command in calls)


def test_fs_search_rejects_content_root_that_cannot_map_corpus(tmp_path) -> None:
    content_root = tmp_path / "content"
    content_root.mkdir()
    rg_path = tmp_path / "rg"
    rg_path.touch()

    try:
        DshFilesystemSearch(
            tmp_path,
            [{"source_path": "docs/page.md", "doc_id": "/page"}],
            rg_path,
            content_root=content_root,
        )
    except ValueError as error:
        assert "does not contain any corpus source_path" in str(error)
    else:
        raise AssertionError("invalid content root was accepted")


def test_code_entity_extraction_is_exact_and_code_shaped() -> None:
    entities = extract_code_entities(
        "Set `ACTIONS_STEP_DEBUG`, call `git blame`, and pass --ignore-revs-file. "
        "Do not turn `ordinary` into an entity."
    )
    assert "actions_step_debug" in entities
    assert "--ignore-revs-file" in entities
    assert "ordinary" not in entities


def test_text_enriched_query_adds_local_image_text_without_vectors() -> None:
    query, diagnostics = text_enriched_query(
        {
            "question_id": "q1",
            "query": "Where is this setting?",
            "requires_multimodal_judgment": True,
            "image_text_evidence": [
                {
                    "role": "question",
                    "alt": "Highlighted settings button",
                    "text": "Repository settings. A settings panel has the button highlighted.",
                }
            ],
        }
    )

    assert "Repository settings" in query
    assert "button highlighted" in query
    assert diagnostics["question_has_image"] is True
    assert diagnostics["question_image_text_available"] is True


def test_text_enriched_query_does_not_duplicate_materialized_image_text() -> None:
    image_text = "A settings panel has the button highlighted."
    query, diagnostics = text_enriched_query(
        {
            "question_id": "q1",
            "query": f"Where is this setting?\n\nImage-derived question evidence:\n- [img-1] {image_text}",
            "requires_multimodal_judgment": True,
            "question_images": [{"source_url": "question.png"}],
            "question_image_text_evidence_used": ["img-1"],
            "image_text_evidence": [
                {"evidence_id": "img-1", "role": "question", "text": image_text}
            ],
        }
    )

    assert query.count(image_text) == 1
    assert diagnostics["question_has_image"] is True
    assert diagnostics["question_images_all_resolved"] is True


def test_answer_only_image_is_not_a_question_image() -> None:
    _, diagnostics = text_enriched_query(
        {
            "question_id": "q1",
            "query": "How do I configure it?",
            "requires_multimodal_judgment": True,
            "question_images": [],
            "image_text_evidence": [
                {"evidence_id": "img-1", "role": "accepted_answer", "text": "A dialog"}
            ],
        }
    )

    assert diagnostics["question_has_image"] is False
    assert diagnostics["question_image_text_available"] is False


def test_graph_fusion_starts_from_exact_hybrid_ranking() -> None:
    base = ["/base-a", "/shared", "/base-b"]
    graph_rows = [
        {"doc_id": "/shared", "path_types": ["markdown_link_1hop"]},
        {"doc_id": "/graph-only", "path_types": ["kggen:requires"]},
    ]

    ranked, diagnostics = fuse_exact_hybrid_with_graph(base, graph_rows, top_k=4)

    assert ranked[0] == "/shared"
    assert set(ranked) == {*base, "/graph-only"}
    assert diagnostics["base_top_k_ids"] == base
    assert diagnostics["graph_candidates_added"] == 1
    assert diagnostics["graph_fusion"] == "equal_weight_rrf"


def test_seed_expansion_uses_two_markdown_link_hops() -> None:
    graph = object.__new__(Neo4jGitHubDocsGraphRAG)
    captured: dict[str, object] = {}

    def fake_execute(query: str, **parameters: object) -> list[dict[str, object]]:
        captured["query"] = query
        captured["parameters"] = parameters
        return [
            {
                "doc_id": "/target",
                "title": "Target",
                "score": 0.8,
                "expanded_from": ["/seed"],
                "path_types": ["markdown_link_2hop"],
                "hops": 2,
            }
        ]

    graph._execute = fake_execute  # type: ignore[method-assign]
    rows, diagnostics = graph.expand_from_seeds(
        np.asarray([1.0, 0.0], dtype=np.float32),
        ["/seed"],
        top_k=10,
        max_entity_degree=20,
        max_route_pages=20,
        max_link_hops=2,
        max_graph_candidates_per_seed=10,
        max_seed_count=5,
    )

    assert "[:KB_LINKS_TO*1..2]" in str(captured["query"])
    assert "KB_SUPPORTED_BY" in str(captured["query"])
    assert "kggen:" in str(captured["query"])
    assert "visual_embedding" not in str(captured["query"])
    assert "db.index.vector.queryNodes" not in str(captured["query"])
    assert rows[0]["hops"] == 2
    assert diagnostics["max_link_hops"] == 2
    assert diagnostics["max_seed_count"] == 5
    assert diagnostics["max_graph_candidates_per_seed"] == 10
    assert diagnostics["graph_candidate_upper_bound"] == 50
    assert str(captured["query"]).count("LIMIT $max_graph_candidates_per_seed") >= 6


def test_ingest_materializes_schema_v2_and_kggen_provenance() -> None:
    graph = object.__new__(Neo4jGitHubDocsGraphRAG)
    queries: list[str] = []
    graph.prepare_schema = lambda *_: None  # type: ignore[method-assign]

    def fake_execute(query: str, **_: object) -> list[object]:
        queries.append(query)
        return []

    graph._execute = fake_execute  # type: ignore[method-assign]
    corpus = [
        {
            "doc_id": "docs::/configure",
            "title": "Configure",
            "source_path": "configure.md",
            "route": "/configure",
            "content_type": "article",
            "variant_conditioned": False,
            "rendered_text": "# Configure\n\nThe CLI overrides the config file.",
            "link_edges": [],
            "reusable_ids": [],
        }
    ]
    chunks = [
        Chunk(
            chunk_id="docs::/configure::c0000",
            doc_id="docs::/configure",
            route="/configure",
            title="Configure",
            text="Configure\ndocs::/configure\nThe CLI overrides the config file.",
        )
    ]
    semantic_artifact = {
        "schema_version": 1,
        "entities": [
            {"entity_id": "e1", "project_id": "docs", "canonical_name": "CLI", "aliases": ["CLI"], "claim_count": 1},
            {"entity_id": "e2", "project_id": "docs", "canonical_name": "config file", "aliases": ["config file"], "claim_count": 1},
        ],
        "predicates": [
            {"predicate_id": "p1", "project_id": "docs", "canonical_name": "overrides", "aliases": ["overrides"], "family": "unclassified"}
        ],
        "claims": [
            {
                "claim_id": "c1",
                "project_id": "docs",
                "subject_id": "e1",
                "predicate_id": "p1",
                "object_id": "e2",
                "evidence_unit_id": "docs::/configure::c0000",
                "raw_subject": "CLI",
                "raw_predicate": "overrides",
                "raw_object": "config file",
                "evidence_excerpt": "The CLI overrides the config file.",
                "extraction_method": "kg-gen",
                "confidence": None,
                "polarity": "unspecified",
                "condition": "",
            }
        ],
    }

    stats = graph.ingest(
        corpus,
        chunks,
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        semantic_artifact=semantic_artifact,
    )

    cypher = "\n".join(queries)
    assert "KBDocsQADocument" in cypher
    assert "KB_HAS_SECTION" in cypher
    assert "KB_SUPPORTED_BY" in cypher
    assert stats.projects == 1
    assert stats.sections == 1
    assert stats.entities == 2
    assert stats.predicates == 1
    assert stats.claims == 1
