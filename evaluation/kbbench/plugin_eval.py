from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from dataset.scripts.records import load_records, read_jsonl as _load_jsonl

from dsh_plugin.backend.graph_records import build_graph_snapshot
from dsh_plugin.backend.retrieval_policy import (
    FINAL_RESULT_LIMIT,
    GRAPH_CANDIDATES_PER_SEED,
    GRAPH_ENTITY_DEGREE_CAP,
    GRAPH_MAX_HOPS,
    GRAPH_ROUTE_PAGE_CAP,
    GRAPH_SEED_LIMIT,
    RRF_CANDIDATES_PER_RETRIEVER,
)
from dsh_plugin.backend.semantic_store import load_semantic_artifact

from .backends import (
    DshFilesystemSearch,
    GH_CHUNK_LABEL as GH_CHUNK_LABEL,
    GH_CODE_LABEL as GH_CODE_LABEL,
    GH_FULLTEXT_INDEX as GH_FULLTEXT_INDEX,
    GH_PAGE_LABEL as GH_PAGE_LABEL,
    GH_REUSABLE_LABEL as GH_REUSABLE_LABEL,
    GH_ROUTE_LABEL as GH_ROUTE_LABEL,
    GH_VECTOR_INDEX as GH_VECTOR_INDEX,
    GitHubGraphStats,
    Neo4jGitHubDocsGraphRAG,
    _parse_rg_matches as _parse_rg_matches,
    extract_code_entities as extract_code_entities,
    fuse_exact_hybrid_with_graph,
    graph_snapshot_verification,
    select_fs_terms as select_fs_terms,
)
from .provenance import (
    RETRIEVAL_SOURCE_PATHS,
    _embedding_model_identity,
    _hardware_and_thread_policy,
    canonical_sha256,
    file_bundle_identity,
    file_identity,
    installed_dependency_versions,
    markdown_tree_identity,
    sanitize_argv,
    sanitize_connection_uri,
)
from .retrieval import (
    GitHubDocsRetriever,
    aspect_retrieval_metrics,
    build_chunks,
    build_or_load_embeddings,
    retrieval_metrics,
    summary_tables,
    _text_hash,
)


ARMS = (
    "dsh_fs_search",
    "dsh_bm25_hnsw",
    "dsh_neo4j_graphrag",
)

REPRODUCIBILITY_SCHEMA_VERSION = 1
RUNTIME_SOURCE_PATHS = ("evaluation/kbbench/plugin_eval.py", *RETRIEVAL_SOURCE_PATHS)
DEPENDENCY_POLICY_PATHS = (
    "evaluation/pyproject.toml",
    "evaluation/requirements.txt",
    "evaluation/requirements-graph.txt",
    "dsh_plugin/backend/requirements-kggen.txt",
)


def _stable_evaluation_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return every retrieval-affecting switch without paths or credentials."""

    return {
        "dataset_name": args.dataset_name,
        "corpus_revision": args.corpus_revision,
        "arms": list(args.arms),
        "limit": args.limit,
        "top_k": args.top_k,
        "retrieval_depth": args.retrieval_depth,
        "embedding_model": args.embedding_model,
        "device_requested": args.device,
        "local_files_only": bool(args.local_files_only),
        "reranker": None,
        "bm25": {"min_df": 1},
        "hnsw": {
            "space": "cosine",
            "m": 24,
            "ef_construction": 180,
            "ef_search": max(160, args.retrieval_depth),
            "random_seed": 17,
            "build_threads": 1,
            "query_threads": 1,
        },
        "rrf_constant": 60,
        "fs_max_calls": args.fs_max_calls,
        "fs_max_matches": args.fs_max_matches,
        "graph_seed_count": args.graph_seed_count,
        "graph_max_hops": GRAPH_MAX_HOPS,
        "max_graph_candidates_per_seed": args.max_graph_candidates_per_seed,
        "max_entity_degree": args.max_entity_degree,
        "max_route_pages": args.max_route_pages,
        "graph_fusion": "equal_weight_rrf",
        "neo4j_uri": sanitize_connection_uri(args.neo4j_uri),
        "neo4j_database": args.neo4j_database,
        "neo4j_ingest": bool(args.ingest),
        "aspects_enabled": bool(args.aspects),
    }


def text_enriched_query(
    question: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    query = str(question["query"])
    question_image_rows = [
        row
        for row in question.get("image_text_evidence") or []
        if row.get("role") == "question"
    ]
    image_text = "\n".join(
        str(row.get("text") or row.get("alt") or "").strip()
        for row in question_image_rows
        if str(row.get("text") or row.get("alt") or "").strip()
    )
    # Normalized benchmark rows already materialize question-image text in the
    # query. Retain backwards compatibility for older rows without duplicating
    # that evidence (which would silently overweight image-bearing questions).
    image_text_materialized = bool(
        question.get("question_image_text_evidence_used")
    ) or bool(image_text and image_text in query)
    if image_text and not image_text_materialized:
        query = f"{query}\nQuestion image evidence:\n{image_text}"
    question_image_references = list(question.get("question_images") or [])
    has_question_image = bool(question_image_references or question_image_rows)
    resolved_image_ids = {
        str(row.get("evidence_id") or row.get("sha256") or "")
        for row in question_image_rows
        if str(row.get("text") or row.get("alt") or "").strip()
    }
    return query, {
        "question_has_image": has_question_image,
        "question_image_references": len(question_image_references)
        if question_image_references
        else len(question_image_rows),
        "question_image_text_available": bool(image_text),
        "question_images_resolved": len(resolved_image_ids),
        "question_images_all_resolved": not has_question_image
        or bool(image_text and len(resolved_image_ids) >= len(question_image_references)),
    }


def paired_bootstrap_delta(
    rows: list[dict[str, Any]],
    treatment: str,
    baseline: str,
    metric: str,
    samples: int = 10_000,
    seed: int = 17,
) -> dict[str, float]:
    treatment_rows = {
        str(row["question_id"]): float(row[metric])
        for row in rows
        if row["arm"] == treatment
    }
    baseline_rows = {
        str(row["question_id"]): float(row[metric])
        for row in rows
        if row["arm"] == baseline
    }
    ids = sorted(treatment_rows.keys() & baseline_rows.keys())
    deltas = np.asarray(
        [treatment_rows[key] - baseline_rows[key] for key in ids], dtype=np.float64
    )
    if not len(deltas):
        return {"mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        means[index] = deltas[rng.integers(0, len(deltas), len(deltas))].mean()
    return {
        "mean_delta": float(deltas.mean()),
        "ci_low": float(np.percentile(means, 2.5)),
        "ci_high": float(np.percentile(means, 97.5)),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    overall_started = time.perf_counter()
    construction_timings: dict[str, float | None] = {}
    project_root = Path(__file__).resolve().parents[2]
    corpus_path = args.dataset_dir / "corpus.jsonl"
    questions_path = args.dataset_dir / "questions.jsonl"
    dataset_manifest_path = args.dataset_dir / "manifest.json"

    stage_started = time.perf_counter()
    corpus_rows = _load_jsonl(corpus_path)
    construction_timings["corpus_load"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    all_questions = load_records(args.dataset_dir)
    construction_timings["questions_load"] = time.perf_counter() - stage_started

    aspect_rows: list[dict[str, Any]] = []
    stage_started = time.perf_counter()
    if args.aspects:
        aspect_rows = _load_jsonl(args.aspects)
    construction_timings["aspects_load"] = time.perf_counter() - stage_started
    aspects_by_id = {
        str(row["question_id"]): row
        for row in aspect_rows
        if row.get("status", "accepted") == "accepted"
    }

    stage_started = time.perf_counter()
    semantic_artifact = load_semantic_artifact(args.kggen_artifact)
    construction_timings["kggen_artifact_load_and_validation"] = (
        time.perf_counter() - stage_started
    )

    questions = list(all_questions)
    if aspects_by_id:
        questions = [row for row in questions if str(row["question_id"]) in aspects_by_id]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        raise ValueError("no questions remain after aspect and limit filters")

    stage_started = time.perf_counter()
    text_chunks = build_chunks(corpus_rows)
    construction_timings["chunk_construction"] = time.perf_counter() - stage_started
    # Image pixels are transcribed once and appended to their owning question
    # or document before chunking. ImageOccurrence nodes remain in Neo4j for
    # provenance and adjacency, but are not an additional vector index whose
    # duplicated text could advantage the hybrid/graph arms.
    indexed_chunks = text_chunks
    print(f"Loading embedding model {args.embedding_model}", flush=True)
    stage_started = time.perf_counter()
    embedding_model = SentenceTransformer(
        args.embedding_model,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    construction_timings["embedding_model_load"] = time.perf_counter() - stage_started
    embedding_cache_fingerprint = _text_hash(indexed_chunks, args.embedding_model)
    embedding_vectors_path = (
        args.cache_dir / f"chunks_{embedding_cache_fingerprint[:16]}.npy"
    )
    embedding_metadata_path = (
        args.cache_dir / f"chunks_{embedding_cache_fingerprint[:16]}.json"
    )
    embedding_cache_hit = embedding_vectors_path.is_file()
    stage_started = time.perf_counter()
    embeddings = build_or_load_embeddings(
        indexed_chunks, embedding_model, args.embedding_model, args.cache_dir
    )
    construction_timings["embedding_load_or_construction"] = (
        time.perf_counter() - stage_started
    )
    expected_graph_snapshot = build_graph_snapshot(
        corpus_rows,
        indexed_chunks,
        embeddings,
        embedding_model_name=args.embedding_model,
        semantic_artifact=semantic_artifact,
    )

    stage_started = time.perf_counter()
    hybrid = GitHubDocsRetriever(
        corpus_rows,
        indexed_chunks,
        embeddings,
        embedding_model,
        reranker=None,
        retrieval_depth=args.retrieval_depth,
    )
    construction_timings["hybrid_bm25_hnsw_index_construction"] = (
        time.perf_counter() - stage_started
    )
    stage_started = time.perf_counter()
    filesystem = DshFilesystemSearch(
        args.repo_root,
        corpus_rows,
        args.dsh_rg,
        content_root=args.content_root,
        max_calls=args.fs_max_calls,
        max_matches_per_call=args.fs_max_matches,
    )
    construction_timings["filesystem_adapter_construction"] = (
        time.perf_counter() - stage_started
    )
    graph: Neo4jGitHubDocsGraphRAG | None = None
    graph_stats: GitHubGraphStats | None = None
    observed_graph_snapshot: dict[str, Any] | None = None
    construction_timings["neo4j_connection"] = None
    construction_timings["neo4j_graph_metadata_or_ingest"] = None
    construction_timings["neo4j_graph_ingest"] = None
    if "dsh_neo4j_graphrag" in args.arms:
        stage_started = time.perf_counter()
        graph = Neo4jGitHubDocsGraphRAG(
            args.neo4j_uri,
            args.neo4j_username,
            args.neo4j_password,
            args.neo4j_database,
        )
        construction_timings["neo4j_connection"] = time.perf_counter() - stage_started
        stage_started = time.perf_counter()
        if args.ingest:
            print("Ingesting the namespaced DocsQA graph into Neo4j", flush=True)
            graph_stats = graph.ingest(
                corpus_rows,
                text_chunks,
                embeddings,
                embedding_chunks=indexed_chunks,
                embedding_model_name=args.embedding_model,
                semantic_artifact=semantic_artifact,
            )
            construction_timings["neo4j_graph_ingest"] = graph_stats.ingest_seconds
            print(json.dumps(asdict(graph_stats), indent=2), flush=True)
        else:
            graph_stats = graph.graph_stats()
        observed_graph_snapshot = graph.graph_snapshot()
        construction_timings["neo4j_graph_metadata_or_ingest"] = (
            time.perf_counter() - stage_started
        )

    # Warm every selected arm before latency measurement.
    stage_started = time.perf_counter()
    warm_query, _ = text_enriched_query(questions[0])
    warm_vector = np.asarray(
        embedding_model.encode([warm_query], normalize_embeddings=True, show_progress_bar=False)[0],
        dtype=np.float32,
    )
    if "dsh_fs_search" in args.arms:
        filesystem.search(warm_query, args.top_k)
    if "dsh_bm25_hnsw" in args.arms:
        hybrid.search(warm_query, "bm25_hnsw_rrf", top_k=args.top_k)
    if graph is not None:
        warm_base, _ = hybrid.search(
            warm_query, "bm25_hnsw_rrf", top_k=args.retrieval_depth
        )
        warm_graph_rows, _ = graph.expand_from_seeds(
            warm_vector,
            warm_base[: args.graph_seed_count],
            top_k=args.graph_seed_count * args.max_graph_candidates_per_seed,
            max_entity_degree=args.max_entity_degree,
            max_route_pages=args.max_route_pages,
            max_link_hops=GRAPH_MAX_HOPS,
            max_graph_candidates_per_seed=args.max_graph_candidates_per_seed,
            max_seed_count=args.graph_seed_count,
        )
        fuse_exact_hybrid_with_graph(
            warm_base, warm_graph_rows, top_k=args.top_k
        )
    construction_timings["arm_warmup"] = time.perf_counter() - stage_started

    rows: list[dict[str, Any]] = []
    measured_queries_started = time.perf_counter()
    try:
        for arm in args.arms:
            print(f"Evaluating {arm} on {len(questions)} questions", flush=True)
            for index, question in enumerate(questions, start=1):
                query, image_diagnostics = text_enriched_query(question)
                if arm == "dsh_fs_search":
                    ranked, diagnostics = filesystem.search(query, args.top_k)
                elif arm == "dsh_bm25_hnsw":
                    ranked, diagnostics = hybrid.search(
                        query,
                        "bm25_hnsw_rrf",
                        top_k=args.top_k,
                    )
                elif arm == "dsh_neo4j_graphrag":
                    if graph is None:
                        raise RuntimeError("Neo4j graph arm was not initialized")
                    base_ranked, base_diagnostics = hybrid.search(
                        query,
                        "bm25_hnsw_rrf",
                        top_k=args.retrieval_depth,
                    )
                    embedding_started = time.perf_counter()
                    query_vector = np.asarray(
                        embedding_model.encode(
                            [query], normalize_embeddings=True, show_progress_bar=False
                        )[0],
                        dtype=np.float32,
                    )
                    embedding_ms = (time.perf_counter() - embedding_started) * 1000.0
                    graph_rows, graph_diagnostics = graph.expand_from_seeds(
                        query_vector,
                        base_ranked[: args.graph_seed_count],
                        top_k=(
                            args.graph_seed_count
                            * args.max_graph_candidates_per_seed
                        ),
                        max_entity_degree=args.max_entity_degree,
                        max_route_pages=args.max_route_pages,
                        max_link_hops=GRAPH_MAX_HOPS,
                        max_graph_candidates_per_seed=args.max_graph_candidates_per_seed,
                        max_seed_count=args.graph_seed_count,
                    )
                    ranked, fusion_diagnostics = fuse_exact_hybrid_with_graph(
                        base_ranked, graph_rows, top_k=args.top_k
                    )
                    diagnostics = {
                        **base_diagnostics,
                        **graph_diagnostics,
                        **fusion_diagnostics,
                        "hybrid_latency_ms": base_diagnostics["latency_ms"],
                        "neo4j_latency_ms": graph_diagnostics["latency_ms"],
                        "query_embedding_ms": embedding_ms,
                        "latency_ms": (
                            base_diagnostics["latency_ms"]
                            + embedding_ms
                            + graph_diagnostics["latency_ms"]
                        ),
                        "hybrid_foundation": "exact_local_bm25_hnsw_rrf",
                        "hybrid_seed_ids": base_ranked[: args.graph_seed_count],
                    }
                else:
                    raise ValueError(f"Unknown arm: {arm}")
                rows.append(
                    {
                        "arm": arm,
                        "question_id": question["question_id"],
                        "project": str(
                            question.get("dataset")
                            or str(question["question_id"]).split("::", 1)[0]
                        ),
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
                        "qrel_count_exact": str(
                            int(
                                question.get("qrel_count")
                                or len(set(question["qrel_ids"]))
                            )
                        ),
                        "qrel_count_group": "1"
                        if len(set(question["qrel_ids"])) == 1 else "2+",
                        "question_has_image": image_diagnostics["question_has_image"],
                        "question_image_group": (
                            "image"
                            if image_diagnostics["question_has_image"]
                            else "text_only"
                        ),
                        "question_image_text_group": (
                            "image_text_available"
                            if image_diagnostics["question_image_text_available"]
                            else "image_text_missing"
                            if image_diagnostics["question_has_image"]
                            else "text_only"
                        ),
                        "relevant_ids": question["qrel_ids"],
                        "ranked_ids": ranked,
                        **image_diagnostics,
                        **diagnostics,
                        **retrieval_metrics(ranked, set(question["qrel_ids"])),
                        **(
                            aspect_retrieval_metrics(
                                ranked,
                                aspects_by_id[str(question["question_id"])]["aspects"],
                            )
                            if aspects_by_id
                            else {}
                        ),
                    }
                )
                if index % 50 == 0:
                    print(f"  {index}/{len(questions)}", flush=True)
    finally:
        construction_timings["measured_query_evaluation"] = (
            time.perf_counter() - measured_queries_started
        )
        if graph is not None:
            stage_started = time.perf_counter()
            graph.close()
            construction_timings["neo4j_connection_close"] = (
                time.perf_counter() - stage_started
            )
        else:
            construction_timings["neo4j_connection_close"] = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_query.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    provenance_started = time.perf_counter()
    effective_content_root = (
        args.content_root.resolve()
        if args.content_root is not None
        else (args.repo_root / "content").resolve()
    )
    selected_aspect_rows = [
        aspects_by_id[str(row["question_id"])]
        for row in questions
        if str(row["question_id"]) in aspects_by_id
    ]
    input_identities: dict[str, Any] = {
        "corpus": {
            **file_identity(corpus_path, "dataset/corpus.jsonl"),
            "canonical_sha256": expected_graph_snapshot["corpus_sha256"],
            "rows": len(corpus_rows),
        },
        "questions": {
            **file_identity(questions_path, "dataset/questions.jsonl"),
            "canonical_sha256": canonical_sha256(all_questions),
            "rows": len(all_questions),
        },
        "answers": file_identity(args.dataset_dir / "answers.jsonl", "dataset/answers.jsonl"),
        "dataset_manifest": (
            file_identity(dataset_manifest_path, "dataset/manifest.json")
            if dataset_manifest_path.is_file()
            else None
        ),
        "selected_question_cohort": {
            "limit": args.limit,
            "rows": len(questions),
            "canonical_sha256": canonical_sha256(questions),
            "question_ids_sha256": canonical_sha256(
                [str(row["question_id"]) for row in questions]
            ),
        },
        "aspects": {
            "enabled": bool(args.aspects),
            "file": (
                file_identity(args.aspects, "aspects/aspects.jsonl")
                if args.aspects
                else None
            ),
            "accepted_rows": len(aspects_by_id),
            "selected_rows": len(selected_aspect_rows),
            "selected_canonical_sha256": (
                canonical_sha256(selected_aspect_rows)
                if args.aspects
                else "none"
            ),
        },
        "kggen_artifact": {
            "provided": bool(args.kggen_artifact),
            "file": (
                file_identity(args.kggen_artifact, "graph/kggen_artifact.json")
                if args.kggen_artifact
                else None
            ),
            "canonical_sha256": expected_graph_snapshot["kggen_sha256"],
        },
    }
    runtime_bundle = file_bundle_identity(
        [(project_root / logical_path, logical_path) for logical_path in RUNTIME_SOURCE_PATHS]
    )
    dependency_policy_bundle = file_bundle_identity(
        [
            (project_root / logical_path, logical_path)
            for logical_path in DEPENDENCY_POLICY_PATHS
        ]
    )
    dependency_versions = installed_dependency_versions(
        (
            "docsqa-benchmark",
            "hnswlib",
            "neo4j",
            "numpy",
            "sentence-transformers",
            "torch",
        )
    )
    filesystem_store = markdown_tree_identity(effective_content_root)
    dsh_rg_identity = file_identity(args.dsh_rg, "dsh_fs_search/rg")
    embedding_cache_identity = {
        "fingerprint": embedding_cache_fingerprint,
        "cache_hit_before_run": embedding_cache_hit,
        "vectors": file_identity(
            embedding_vectors_path, "embedding_cache/chunks.npy"
        ),
        "metadata": (
            file_identity(embedding_metadata_path, "embedding_cache/chunks.json")
            if embedding_metadata_path.is_file()
            else None
        ),
    }
    graph_verification = graph_snapshot_verification(
        expected_graph_snapshot, observed_graph_snapshot
    )
    graph_verification["selected"] = "dsh_neo4j_graphrag" in args.arms
    model_identity = {
        **_embedding_model_identity(embedding_model, args.embedding_model),
        "requested_device": args.device,
        "local_files_only": bool(args.local_files_only),
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_dtype": str(embeddings.dtype),
        "corpus_embeddings_sha256": expected_graph_snapshot[
            "embeddings_sha256"
        ],
        "corpus_and_query_normalization": "l2",
        "image_vectors": False,
        "reranker": None,
    }
    execution_environment = {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        **_hardware_and_thread_policy(),
    }
    stable_config = _stable_evaluation_config(args)
    stable_config_sha256 = canonical_sha256(stable_config)
    stores = {
        "filesystem_markdown": filesystem_store,
        "embedding_cache": embedding_cache_identity,
        "neo4j_graph_snapshot": graph_verification,
    }
    fingerprint_stores = {
        **stores,
        "embedding_cache": {
            key: value
            for key, value in embedding_cache_identity.items()
            if key != "cache_hit_before_run"
        },
    }
    deterministic_contract = {
        "schema_version": REPRODUCIBILITY_SCHEMA_VERSION,
        "config_sha256": stable_config_sha256,
        "inputs": input_identities,
        "stores": fingerprint_stores,
        "runtime_sha256": runtime_bundle["sha256"],
        "dependency_policy_sha256": dependency_policy_bundle["sha256"],
        "dependency_versions": dependency_versions,
        "model": model_identity,
    }
    retrieval_artifact_fingerprint = canonical_sha256(deterministic_contract)
    execution_fingerprint = canonical_sha256(
        {
            "retrieval_artifact_fingerprint_sha256": (
                retrieval_artifact_fingerprint
            ),
            "execution_environment": execution_environment,
            "embedding_cache_hit_before_run": embedding_cache_hit,
        }
    )
    construction_timings["provenance_hashing"] = (
        time.perf_counter() - provenance_started
    )
    construction_timings["total_before_report_write"] = (
        time.perf_counter() - overall_started
    )
    reproducibility = {
        "schema_version": REPRODUCIBILITY_SCHEMA_VERSION,
        "retrieval_artifact_fingerprint_sha256": retrieval_artifact_fingerprint,
        "execution_fingerprint_sha256": execution_fingerprint,
        "invocation": {
            "entrypoint": "python -m kbbench.plugin_eval",
            "argv": sanitize_argv(getattr(args, "invocation_argv", [])),
            "credentials_recorded": False,
        },
        "config": stable_config,
        "config_sha256": stable_config_sha256,
        "inputs": input_identities,
        "stores": stores,
        "runtime": runtime_bundle,
        "dependencies": {
            "policy_files": dependency_policy_bundle,
            "installed_versions": dependency_versions,
        },
        "model": model_identity,
        "execution_environment": execution_environment,
        "construction_timings_seconds": construction_timings,
        "timing_scope": {
            "construction_is_excluded_from_per_query_latency": True,
            "embedding_cache_hit_is_reported": True,
            "neo4j_ingest_is_measured_only_when_ingest_was_requested": True,
            "warmup_is_excluded_from_measured_query_latency": True,
        },
        "tools": {"dsh_ripgrep": dsh_rg_identity},
    }
    report = {
        "benchmark": f"{args.dataset_name} DSH three-arm plugin retrieval evaluation",
        "dataset": args.dataset_name,
        "corpus_revision": args.corpus_revision,
        "documents": len(corpus_rows),
        "chunks": len(indexed_chunks),
        "text_chunks": len(text_chunks),
        "image_retrieval_units": len(indexed_chunks) - len(text_chunks),
        "questions": len(questions),
        "aspect_annotation_file": str(args.aspects) if args.aspects else None,
        "aspect_aware_metrics": bool(aspects_by_id),
        "arms": list(args.arms),
        "top_k": args.top_k,
        "retrieval_depth": args.retrieval_depth,
        "graph_parameters": {
            "seed_count": args.graph_seed_count,
            "fusion": "equal_weight_rrf",
            "max_link_hops": GRAPH_MAX_HOPS,
            "max_graph_candidates_per_seed": (
                args.max_graph_candidates_per_seed
            ),
            "graph_candidate_upper_bound": (
                args.graph_seed_count * args.max_graph_candidates_per_seed
            ),
            "max_entity_degree": args.max_entity_degree,
            "max_route_pages": args.max_route_pages,
        },
        "graph_stats": asdict(graph_stats) if graph_stats is not None else None,
        **summary_tables(
            [{**row, "method": row["arm"]} for row in rows],
            extra_slices=(
                ("project", "project"),
                ("qrel_count_exact", "qrel_count_exact"),
                ("question_image", "question_image_group"),
                ("question_image_text_availability", "question_image_text_group"),
            ),
        ),
        "paired_bootstrap_vs_hybrid": {
            metric: paired_bootstrap_delta(
                rows,
                "dsh_neo4j_graphrag",
                "dsh_bm25_hnsw",
                metric,
            )
            for metric in (
                "recall_at_10",
                "hit_at_10",
                "ndcg_at_10",
                "weighted_aspect_recall_at_10",
                "alpha_ndcg_at_10",
            )
            if not metric.startswith(("weighted_aspect", "alpha_")) or aspects_by_id
            if {"dsh_neo4j_graphrag", "dsh_bm25_hnsw"}.issubset(args.arms)
        },
        "latency_scope": (
            "Warm plugin retrieval. Filesystem includes bounded DSH ripgrep subprocess calls; "
            "hybrid includes query embedding, BM25, HNSW, and RRF; Neo4j includes query "
            "embedding, the identical local hybrid foundation, driver/network time, bounded "
            "Cypher expansion, and a second equal-weight RRF. "
            "Index and graph construction are excluded and reported separately."
        ),
        "construct_note": (
            "Plugin-level retrieval comparison. DSH agent model/orchestration and answer "
            "generation are intentionally held out. Structure and image provenance are "
            "deterministic and query-blind. When supplied, the KGGen artifact adds open "
            "entity/predicate extraction and clustering while preserving unit evidence."
        ),
        "image_scope": (
            "All arms receive the same local text derived from reproducible images. "
            "Image vectors are excluded; missing required image text remains explicit."
        ),
        "api_cost_usd": 0.0,
        "reproducibility": reproducibility,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the matched three-arm DSH DocsQA retrieval benchmark"
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--aspects",
        type=Path,
        help="Optional frozen aspects.jsonl; when supplied, evaluate only annotated questions and add aspect-aware metrics.",
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument(
        "--content-root",
        type=Path,
        help="Markdown/MDX root; defaults to <repo-root>/content",
    )
    parser.add_argument("--dataset-name", default="docsqa")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--corpus-revision", required=True)
    parser.add_argument("--dsh-rg", type=Path, required=True)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--top-k", type=int, default=FINAL_RESULT_LIMIT)
    parser.add_argument(
        "--retrieval-depth", type=int, default=RRF_CANDIDATES_PER_RETRIEVER
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--fs-max-calls", type=int, default=3)
    parser.add_argument("--fs-max-matches", type=int, default=250)
    parser.add_argument(
        "--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
    )
    parser.add_argument(
        "--neo4j-username", default=os.environ.get("NEO4J_USERNAME", "neo4j")
    )
    parser.add_argument(
        "--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", "secretgraph")
    )
    parser.add_argument(
        "--neo4j-database", default=os.environ.get("NEO4J_DATABASE", "neo4j")
    )
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument(
        "--kggen-artifact",
        type=Path,
        help="Optional KGGen artifact produced by dsh_plugin.backend.kggen_adapter",
    )
    parser.add_argument("--graph-seed-count", type=int, default=GRAPH_SEED_LIMIT)
    parser.add_argument(
        "--max-entity-degree", type=int, default=GRAPH_ENTITY_DEGREE_CAP
    )
    parser.add_argument(
        "--max-route-pages", type=int, default=GRAPH_ROUTE_PAGE_CAP
    )
    parser.add_argument(
        "--max-graph-candidates-per-seed",
        type=int,
        default=GRAPH_CANDIDATES_PER_SEED,
    )
    args = parser.parse_args()
    args.invocation_argv = list(sys.argv[1:])
    report = evaluate(args)
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
