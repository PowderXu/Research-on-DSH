from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from dsh_plugin.backend.graph_contract import (
    CLAIM_LABEL,
    CODE_LABEL,
    CONSTRAINTS,
    DOCUMENT_LABEL,
    ENTITY_LABEL,
    HAS_OBJECT,
    HAS_DOCUMENT,
    HAS_SECTION,
    HAS_UNIT,
    IMAGE_ASSET_LABEL,
    IMAGE_OCCURRENCE_LABEL,
    INCLUDES,
    IN_ROUTE,
    LINKS_TO,
    LEGACY_CONSTRAINTS,
    MENTIONS,
    NAMESPACED_LABELS,
    NEAR,
    NEXT_UNIT,
    PREDICATE_LABEL,
    PROJECT_LABEL,
    REUSABLE_LABEL,
    ROUTE_LABEL,
    SECTION_LABEL,
    SNAPSHOT_LABEL,
    SUBJECT_OF,
    SUPPORTED_BY,
    TEXT_VECTOR_INDEX,
    UNIT_FULLTEXT_INDEX,
    UNIT_LABEL,
    USES_ASSET,
    USES_PREDICATE,
)
from dsh_plugin.backend.graph_records import build_graph_records, build_graph_snapshot
from dsh_plugin.backend.retrieval_policy import (
    FINAL_RESULT_LIMIT,
    GRAPH_CANDIDATES_PER_SEED,
    GRAPH_ENTITY_DEGREE_CAP,
    GRAPH_MAX_HOPS,
    GRAPH_ROUTE_PAGE_CAP,
    GRAPH_SEED_LIMIT,
    RRF_CANDIDATES_PER_RETRIEVER,
)
from dsh_plugin.backend.semantic_store import (
    ingest_semantic_artifact,
    load_semantic_artifact,
)

from .retrieval import (
    Chunk,
    GitHubDocsRetriever,
    aspect_retrieval_metrics,
    build_chunks,
    build_or_load_embeddings,
    retrieval_metrics,
    rrf,
    summarize,
    _text_hash,
)


ARMS = (
    "dsh_fs_search",
    "dsh_bm25_hnsw",
    "dsh_neo4j_graphrag",
)

REPRODUCIBILITY_SCHEMA_VERSION = 1
RUNTIME_SOURCE_PATHS = (
    "evaluation/kbbench/plugin_eval.py",
    "evaluation/kbbench/retrieval.py",
    "evaluation/kbbench/indexes.py",
    "evaluation/kbbench/scoring.py",
    "dsh_plugin/backend/graph_contract.py",
    "dsh_plugin/backend/graph_records.py",
    "dsh_plugin/backend/retrieval_policy.py",
    "dsh_plugin/backend/semantic_store.py",
)
DEPENDENCY_POLICY_PATHS = (
    "evaluation/pyproject.toml",
    "evaluation/requirements.txt",
    "evaluation/requirements-graph.txt",
    "dsh_plugin/backend/requirements-kggen.txt",
)


def canonical_sha256(value: Any) -> str:
    """Hash JSON-like data with a stable encoding and key order."""

    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def file_identity(path: Path, logical_path: str) -> dict[str, Any]:
    """Describe an input by logical name and bytes, not a host-absolute path."""

    resolved = path.resolve()
    return {
        "path": logical_path,
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "bytes": resolved.stat().st_size,
    }


def file_bundle_identity(
    paths: Sequence[tuple[Path, str]],
) -> dict[str, Any]:
    """Fingerprint a small source/dependency bundle under stable logical paths."""

    existing: list[dict[str, Any]] = []
    missing: list[str] = []
    for path, logical_path in sorted(paths, key=lambda item: item[1]):
        if path.is_file():
            existing.append(file_identity(path, logical_path))
        else:
            missing.append(logical_path)
    return {
        "sha256": canonical_sha256({"files": existing, "missing": missing}),
        "files": existing,
        "missing": missing,
    }


def markdown_tree_identity(root: Path) -> dict[str, Any]:
    """Fingerprint the exact Markdown bytes visible to the filesystem arm."""

    resolved_root = root.resolve()
    files = sorted(
        (
            path
            for path in resolved_root.rglob("*")
            if path.is_file() and path.suffix.casefold() in {".md", ".mdx"}
        ),
        key=lambda path: path.relative_to(resolved_root).as_posix(),
    )
    digest = hashlib.sha256()
    byte_count = 0
    for path in files:
        logical_path = path.relative_to(resolved_root).as_posix()
        content = path.read_bytes()
        digest.update(logical_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
        byte_count += len(content)
    return {
        "path": "filesystem_documents",
        "sha256": digest.hexdigest(),
        "files": len(files),
        "bytes": byte_count,
        "extensions": [".md", ".mdx"],
    }


def installed_dependency_versions(names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def sanitize_connection_uri(value: str) -> str:
    """Remove URI userinfo while preserving the endpoint needed for latency context."""

    return re.sub(r"(?<=://)[^/@]+@", "<redacted>@", value, count=1)


def sanitize_argv(argv: Sequence[str]) -> list[str]:
    """Retain the executed CLI while ensuring Neo4j credentials never enter reports."""

    sanitized: list[str] = []
    redact_next = False
    sanitize_uri_next = False
    for value in argv:
        if redact_next:
            sanitized.append("<redacted>")
            redact_next = False
            continue
        if sanitize_uri_next:
            sanitized.append(sanitize_connection_uri(value))
            sanitize_uri_next = False
            continue
        if value == "--neo4j-password":
            sanitized.append(value)
            redact_next = True
            continue
        if value.startswith("--neo4j-password="):
            sanitized.append("--neo4j-password=<redacted>")
            continue
        if value == "--neo4j-uri":
            sanitized.append(value)
            sanitize_uri_next = True
            continue
        if value.startswith("--neo4j-uri="):
            sanitized.append(
                "--neo4j-uri="
                + sanitize_connection_uri(value.split("=", 1)[1])
            )
            continue
        sanitized.append(value)
    return sanitized


def _embedding_model_identity(model: Any, configured_name: str) -> dict[str, Any]:
    """Record the configured model and the resolved Hugging Face revision when exposed."""

    resolved_revision = ""
    resolved_architecture = ""
    try:
        transformer = model[0]
        auto_model = getattr(transformer, "auto_model", None)
        config = getattr(auto_model, "config", None)
        resolved_revision = str(getattr(config, "_commit_hash", "") or "")
        architectures = list(getattr(config, "architectures", None) or [])
        resolved_architecture = ",".join(map(str, architectures))
    except (IndexError, KeyError, TypeError):
        pass
    actual_device = str(getattr(model, "device", "") or "")
    return {
        "configured_name": configured_name,
        "resolved_revision": resolved_revision,
        "resolved_architecture": resolved_architecture,
        "actual_device": actual_device,
    }


def _hardware_and_thread_policy() -> dict[str, Any]:
    thread_environment = {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    torch_threads: dict[str, int] | None = None
    try:
        import torch

        torch_threads = {
            "intraop": int(torch.get_num_threads()),
            "interop": int(torch.get_num_interop_threads()),
        }
    except (ImportError, RuntimeError):
        pass
    return {
        "hardware": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "threads": {
            "evaluation_query_execution": "serial",
            "arm_execution": "serial",
            "hnsw_build_threads": 1,
            "hnsw_query_threads": 1,
            "embedding_build_batch_size": 64,
            "torch": torch_threads,
            "environment": thread_environment,
        },
    }


def _stable_evaluation_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return every retrieval-affecting switch without paths or credentials."""

    return {
        "dataset_name": args.dataset_name,
        "corpus_revision": args.corpus_revision,
        "arms": list(args.arms),
        "split": args.split,
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


def graph_snapshot_verification(
    expected: Mapping[str, Any], observed: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Compare the locally derived store identity with the queried Neo4j snapshot."""

    observed_row = dict(observed or {})
    expected_sha = str(expected.get("snapshot_sha256") or "")
    observed_sha = str(observed_row.get("snapshot_sha256") or "")
    return {
        "expected": dict(expected),
        "observed": observed_row or None,
        "matches_expected": bool(
            expected_sha and observed_sha and expected_sha == observed_sha
        ),
        "status": (
            "match"
            if expected_sha and expected_sha == observed_sha
            else "missing_observed_snapshot"
            if not observed_sha
            else "mismatch"
        ),
    }


def fuse_exact_hybrid_with_graph(
    base_ranked: list[str],
    graph_rows: list[dict[str, Any]],
    *,
    top_k: int,
) -> tuple[list[str], dict[str, Any]]:
    """Fuse the exact hybrid baseline ranking with bounded graph discoveries.

    Both inputs are rankings, so ordinary equal-weight RRF is appropriate and
    avoids comparing incomparable local-retriever and Neo4j raw scores.
    """

    graph_ranked = list(
        dict.fromkeys(str(row["doc_id"]) for row in graph_rows if row.get("doc_id"))
    )
    fused, _ = rrf((base_ranked, graph_ranked))
    final = fused[:top_k]
    path_types: Counter[str] = Counter(
        str(path_type)
        for row in graph_rows
        for path_type in row.get("path_types") or []
    )
    base_set = set(base_ranked)
    graph_set = set(graph_ranked)
    return final, {
        "graph_applied": True,
        "graph_candidates_added": sum(doc_id not in base_set for doc_id in graph_ranked),
        "graph_boosted_results": sum(doc_id in graph_set for doc_id in final),
        "graph_path_types": dict(path_types),
        "graph_fusion": "equal_weight_rrf",
        "base_top_k_ids": base_ranked[:top_k],
        "graph_ranked_ids": graph_ranked,
    }


# Compatibility aliases retained for existing reports and public imports.
GH_PAGE_LABEL = DOCUMENT_LABEL
GH_CHUNK_LABEL = UNIT_LABEL
GH_ROUTE_LABEL = ROUTE_LABEL
GH_REUSABLE_LABEL = REUSABLE_LABEL
GH_CODE_LABEL = CODE_LABEL
GH_VECTOR_INDEX = TEXT_VECTOR_INDEX
GH_FULLTEXT_INDEX = UNIT_FULLTEXT_INDEX

FS_TOKEN_RE = re.compile(r"(?u)\b[A-Za-z0-9][A-Za-z0-9_.:/-]{2,}\b")
INLINE_CODE_RE = re.compile(r"`([^`\n]{2,100})`")
FLAG_RE = re.compile(r"(?<![\w-])--[A-Za-z0-9][\w-]{1,60}")
ENV_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,60}\b")
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


FS_STOPWORDS = frozenset(
    """
    about after again against all also am an and any are arent as at be because been
    before being between both but by can cannot cant could couldnt did didnt do does
    doesnt doing dont down during each few for from further had hadnt has hasnt have
    havent having he her here hers herself him himself his how i if in into is isnt it
    its itself just may me might more most must my myself no nor not now of off on once
    only or other ought our ours ourselves out over own same should shouldnt so some such
    than that the their theirs them themselves then there these they this those through
    to too under until up very was wasnt we were werent what when where which while who
    whom why will with wont would wouldnt you your yours yourself yourselves github use
    using want need help issue problem question possible way get make like
    """.split()
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _batched(values: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def select_fs_terms(query: str, max_terms: int = 3) -> list[str]:
    """Compile a bounded natural-language query into DSH grep primitives.

    The filesystem arm deliberately has no corpus index. Exact strings inside
    code/quotes are preferred, followed by longer non-stopword tokens. This is
    deterministic query planning for the primitive `dsh-tool-fs-search` grep
    tool, not BM25 hidden behind a filesystem label.
    """

    exact: list[str] = []
    for pattern in (
        re.compile(r"`([^`\n]{2,100})`"),
        re.compile(r"[\"']([^\"'\n]{3,100})[\"']"),
    ):
        exact.extend(match.group(1).strip() for match in pattern.finditer(query))
    tokens = [match.group(0) for match in FS_TOKEN_RE.finditer(query)]
    candidates: list[tuple[int, int, int, str]] = []
    seen: set[str] = set()
    position = 0
    for value in [*exact, *tokens]:
        normalized = value.casefold().strip()
        if not normalized or normalized in seen:
            position += 1
            continue
        seen.add(normalized)
        words = normalized.split()
        if (
            len(normalized) < 4
            or len(words) > 5
            or all(word in FS_STOPWORDS for word in words)
        ):
            position += 1
            continue
        identifier = int(
            value in exact
            or bool(re.search(r"[_.:/-]|[A-Z].*[A-Z]|\d", value))
        )
        informative_length = min(40, len(normalized))
        candidates.append((identifier, informative_length, -position, normalized))
        position += 1
    candidates.sort(reverse=True)
    return [value for _, _, _, value in candidates[:max_terms]]


def _parse_rg_matches(stdout: str, limit: int) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("type") != "match":
            continue
        data = record.get("data") or {}
        path = (data.get("path") or {}).get("text")
        text = (data.get("lines") or {}).get("text")
        line_number = data.get("line_number")
        if not isinstance(path, str) or not isinstance(text, str):
            continue
        matches.append(
            {
                "path": path,
                "line": text.rstrip("\r\n"),
                "line_number": int(line_number or 0),
            }
        )
        if len(matches) >= limit:
            break
    return matches


class DshFilesystemSearch:
    """The packaged DSH grep primitive plus a fixed three-call planner."""

    def __init__(
        self,
        repo_root: Path,
        corpus_rows: list[dict[str, Any]],
        rg_path: Path,
        content_root: Path | None = None,
        max_calls: int = 3,
        max_matches_per_call: int = 250,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.content_root = (
            content_root.resolve()
            if content_root is not None
            else (self.repo_root / "content").resolve()
        )
        self.rg_path = rg_path.resolve()
        self.max_calls = max_calls
        self.max_matches_per_call = max_matches_per_call
        self.source_to_doc = {
            Path(str(row["source_path"])).as_posix(): str(row["doc_id"])
            for row in corpus_rows
        }
        if not self.rg_path.exists():
            raise FileNotFoundError(f"DSH packaged ripgrep not found: {self.rg_path}")
        if self.source_to_doc and not any(
            (self.content_root / source_path).is_file()
            for source_path in self.source_to_doc
        ):
            raise ValueError(
                "Filesystem content root does not contain any corpus source_path: "
                f"{self.content_root}"
            )

    def _doc_id(self, raw_path: str) -> str | None:
        path = Path(raw_path)
        if path.is_absolute():
            try:
                relative = path.resolve().relative_to(self.content_root).as_posix()
            except ValueError:
                return None
        else:
            relative = path.as_posix()
            if relative.startswith("./"):
                relative = relative[2:]
            content_prefix = self.content_root.relative_to(self.repo_root).as_posix() + "/"
            if relative.startswith(content_prefix):
                relative = relative[len(content_prefix) :]
        return self.source_to_doc.get(relative)

    def search(self, query: str, top_k: int) -> tuple[list[str], dict[str, Any]]:
        started = time.perf_counter()
        terms = select_fs_terms(query, self.max_calls)
        scores: dict[str, float] = defaultdict(float)
        best_lines: dict[str, str] = {}
        total_matches = 0
        calls = 0
        for term in terms:
            calls += 1
            pattern = f"(?i){re.escape(term)}"
            completed = subprocess.run(
                [
                    str(self.rg_path),
                    "--json",
                    "--no-ignore",
                    f"--regexp={pattern}",
                    "--glob=*.md",
                    "--glob=*.mdx",
                    "--",
                    str(self.content_root),
                ],
                cwd=self.repo_root,
                text=True,
                capture_output=True,
                check=False,
                timeout=10,
            )
            if completed.returncode not in (0, 1):
                raise RuntimeError(
                    f"DSH filesystem grep failed ({completed.returncode}): "
                    f"{completed.stderr[-1000:]}"
                )
            matches = _parse_rg_matches(completed.stdout, self.max_matches_per_call)
            total_matches += len(matches)
            file_rank = 0
            seen_in_call: set[str] = set()
            query_tokens = {value.casefold() for value in FS_TOKEN_RE.findall(query)}
            for match in matches:
                doc_id = self._doc_id(str(match["path"]))
                if doc_id is None:
                    continue
                if doc_id not in seen_in_call:
                    file_rank += 1
                    seen_in_call.add(doc_id)
                    scores[doc_id] += 1.0 / (60.0 + file_rank)
                line_tokens = {value.casefold() for value in FS_TOKEN_RE.findall(match["line"])}
                coverage = len(query_tokens & line_tokens) / max(1, len(query_tokens))
                scores[doc_id] += 0.003 * coverage
                best_lines.setdefault(doc_id, str(match["line"]))
        ranked = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:top_k]
        return ranked, {
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "fs_terms": terms,
            "fs_calls": calls,
            "fs_matches_visible": total_matches,
            "fs_backend": "@deepseek-ai/dsh-tool-fs-search@0.1.1-rc.2",
            "graph_applied": False,
            "graph_candidates_added": 0,
            "selected_routes": [],
        }


def extract_code_entities(text: str) -> set[str]:
    """Extract exact, code-shaped identifiers without an LLM or query access."""

    values = set(FLAG_RE.findall(text)) | set(ENV_RE.findall(text))
    for match in INLINE_CODE_RE.finditer(text):
        value = match.group(1).strip()
        if len(value.split()) > 3 or value.startswith(("http://", "https://")):
            continue
        if re.search(r"[_.:/-]|\d|[a-z][A-Z]", value):
            values.add(value)
    return {re.sub(r"\s+", " ", value).casefold() for value in values if len(value) <= 100}


@dataclass(frozen=True)
class GitHubGraphStats:
    pages: int
    chunks: int
    markdown_links: int
    routes: int
    reusable_entities: int
    reusable_mentions: int
    code_entities: int
    code_mentions: int
    ingest_seconds: float | None
    projects: int = 0
    sections: int = 0
    image_assets: int = 0
    image_occurrences: int = 0
    entities: int = 0
    predicates: int = 0
    claims: int = 0
    snapshot_sha256: str = ""


class Neo4jGitHubDocsGraphRAG:
    """Neo4j graph store and bounded expansion over exact hybrid seeds."""

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.driver.verify_connectivity()
        self.database = database

    def close(self) -> None:
        self.driver.close()

    def _execute(self, query: str, **parameters: Any) -> list[Any]:
        records, _, _ = self.driver.execute_query(
            query,
            parameters,
            database_=self.database,
        )
        return records

    def prepare_schema(self, dimensions: int) -> None:
        for name in LEGACY_CONSTRAINTS:
            self._execute(f"DROP CONSTRAINT {name} IF EXISTS")
        for name, label, property_name in CONSTRAINTS:
            self._execute(
                f"CREATE CONSTRAINT {name} IF NOT EXISTS "
                f"FOR (node:{label}) REQUIRE node.{property_name} IS UNIQUE"
            )
        self._execute(
            f"CREATE VECTOR INDEX {GH_VECTOR_INDEX} IF NOT EXISTS "
            f"FOR (unit:{UNIT_LABEL}) ON unit.text_embedding "
            "OPTIONS {indexConfig: {"
            f"`vector.dimensions`: {int(dimensions)}, "
            "`vector.similarity_function`: 'cosine'}}"
        )
        self._execute(
            f"CREATE FULLTEXT INDEX {GH_FULLTEXT_INDEX} IF NOT EXISTS "
            f"FOR (unit:{UNIT_LABEL}) ON EACH [unit.search_text]"
        )

    def ingest(
        self,
        corpus_rows: list[dict[str, Any]],
        chunks: list[Any],
        embeddings: np.ndarray,
        *,
        embedding_chunks: list[Any] | None = None,
        embedding_model_name: str = "unspecified",
        semantic_artifact: dict[str, Any] | None = None,
        batch_size: int = 250,
    ) -> GitHubGraphStats:
        started = time.perf_counter()
        records = build_graph_records(corpus_rows, chunks)
        indexed_chunks = embedding_chunks or chunks
        snapshot = build_graph_snapshot(
            corpus_rows,
            indexed_chunks,
            embeddings,
            embedding_model_name=embedding_model_name,
            semantic_artifact=semantic_artifact,
        )
        self.prepare_schema(int(embeddings.shape[1]))
        label_filter = " OR ".join(f"node:{label}" for label in NAMESPACED_LABELS)
        self._execute(f"MATCH (node) WHERE {label_filter} DETACH DELETE node")

        self._execute(
            f"""
            CREATE (snapshot:{SNAPSHOT_LABEL} {{
              snapshot_id: $row.snapshot_id,
              schema_version: $row.schema_version,
              corpus_sha256: $row.corpus_sha256,
              chunks_sha256: $row.chunks_sha256,
              embeddings_sha256: $row.embeddings_sha256,
              embedding_model: $row.embedding_model,
              kggen_sha256: $row.kggen_sha256,
              snapshot_sha256: $row.snapshot_sha256
            }})
            """,
            row=snapshot,
        )

        for batch in _batched(records["projects"], batch_size):
            self._execute(
                f"""
                UNWIND $rows AS row
                CREATE (project:{PROJECT_LABEL} {{project_id: row.project_id}})
                """,
                rows=batch,
            )
        page_rows = records["documents"]
        for batch in _batched(page_rows, batch_size):
            self._execute(
                f"""
                UNWIND $rows AS row
                CREATE (document:{DOCUMENT_LABEL} {{
                  doc_id: row.doc_id,
                  project_id: row.project_id,
                  title: row.title,
                  source_path: row.source_path,
                  route: row.route,
                  content_type: row.content_type,
                  variant_conditioned: row.variant_conditioned
                }})
                WITH document, row
                MATCH (project:{PROJECT_LABEL} {{project_id: row.project_id}})
                CREATE (project)-[:{HAS_DOCUMENT}]->(document)
                """,
                rows=batch,
            )

        for batch in _batched(records["sections"], batch_size):
            self._execute(
                f"""
                UNWIND $rows AS row
                CREATE (section:{SECTION_LABEL} {{
                  section_id: row.section_id,
                  doc_id: row.doc_id,
                  title: row.title,
                  level: row.level,
                  ordinal: row.ordinal
                }})
                WITH section, row
                MATCH (document:{DOCUMENT_LABEL} {{doc_id: row.doc_id}})
                CREATE (document)-[:{HAS_SECTION}]->(section)
                """,
                rows=batch,
            )

        embedding_by_unit = {
            str(chunk.chunk_id): embeddings[index].astype(float).tolist()
            for index, chunk in enumerate(indexed_chunks)
        }
        unit_rows = [
            {
                **row,
                "text_embedding": embedding_by_unit.get(str(row["unit_id"])),
            }
            for row in records["units"]
        ]
        for batch in _batched(unit_rows, batch_size):
            self._execute(
                f"""
                UNWIND $rows AS row
                CREATE (unit:{UNIT_LABEL} {{
                  unit_id: row.unit_id,
                  doc_id: row.doc_id,
                  project_id: row.project_id,
                  section_id: row.section_id,
                  unit_type: row.unit_type,
                  ordinal: row.ordinal,
                  title: row.title,
                  search_text: row.search_text,
                  source_path: row.source_path,
                  source_line: row.source_line,
                  text_embedding: row.text_embedding
                }})
                WITH unit, row
                MATCH (section:{SECTION_LABEL} {{section_id: row.section_id}})
                CREATE (section)-[:{HAS_UNIT}]->(unit)
                """,
                rows=batch,
            )
        for batch in _batched(records["next_edges"], batch_size * 2):
            self._execute(
                f"""
                UNWIND $rows AS row
                MATCH (source:{UNIT_LABEL} {{unit_id: row.source}})
                MATCH (target:{UNIT_LABEL} {{unit_id: row.target}})
                CREATE (source)-[:{NEXT_UNIT}]->(target)
                """,
                rows=batch,
            )

        for batch in _batched(records["image_assets"], batch_size):
            self._execute(
                f"""
                UNWIND $rows AS row
                CREATE (asset:{IMAGE_ASSET_LABEL} {{
                  asset_id: row.asset_id,
                  project_id: row.project_id,
                  source_url: row.source_url,
                  content_hash: row.content_hash,
                  local_path: row.local_path,
                  mime_type: row.mime_type,
                  width: row.width,
                  height: row.height,
                  description: row.description,
                  ocr_text: row.ocr_text
                }})
                """,
                rows=batch,
            )
        for batch in _batched(records["image_occurrences"], batch_size):
            self._execute(
                f"""
                UNWIND $rows AS row
                MATCH (occurrence:{UNIT_LABEL} {{unit_id: row.unit_id}})
                SET occurrence:{IMAGE_OCCURRENCE_LABEL},
                    occurrence.alt_text = row.alt_text,
                    occurrence.image_title = row.image_title,
                    occurrence.source_url = row.source_url,
                    occurrence.source_syntax = row.source_syntax,
                    occurrence.asset_id = row.asset_id
                WITH occurrence, row
                MATCH (asset:{IMAGE_ASSET_LABEL} {{asset_id: row.asset_id}})
                CREATE (occurrence)-[:{USES_ASSET}]->(asset)
                """,
                rows=batch,
            )
        for batch in _batched(records["near_edges"], batch_size * 2):
            self._execute(
                f"""
                UNWIND $rows AS row
                MATCH (occurrence:{IMAGE_OCCURRENCE_LABEL} {{unit_id: row.occurrence_id}})
                MATCH (unit:{UNIT_LABEL} {{unit_id: row.unit_id}})
                CREATE (occurrence)-[:{NEAR}]->(unit)
                """,
                rows=batch,
            )

        link_rows = [
            {
                "source": str(row["doc_id"]),
                "target": str(edge["target_id"]),
                "anchor_text": str(edge.get("anchor_text", "")),
                "source_section": str(edge.get("source_section", "")),
                "target_anchor": str(edge.get("target_anchor", "")),
                "context": str(edge.get("context", "")),
            }
            for row in corpus_rows
            for edge in row.get("link_edges", [])
            if edge.get("target_id")
        ]
        for batch in _batched(link_rows, batch_size * 2):
            self._execute(
                f"""
                UNWIND $rows AS row
                MATCH (source:{DOCUMENT_LABEL} {{doc_id: row.source}})
                MATCH (target:{DOCUMENT_LABEL} {{doc_id: row.target}})
                CREATE (source)-[:{LINKS_TO} {{
                  anchor_text: row.anchor_text,
                  source_section: row.source_section,
                  target_anchor: row.target_anchor,
                  context: row.context
                }}]->(target)
                """,
                rows=batch,
            )

        routes: dict[str, int] = Counter(str(row["route"]) for row in corpus_rows)
        route_rows = [
            {"route_id": route, "document_count": count}
            for route, count in sorted(routes.items())
        ]
        for batch in _batched(route_rows, batch_size):
            self._execute(
                f"""
                UNWIND $rows AS row
                CREATE (route:{ROUTE_LABEL} {{
                  route_id: row.route_id,
                  document_count: row.document_count
                }})
                """,
                rows=batch,
            )
        self._execute(
            f"""
            MATCH (document:{DOCUMENT_LABEL})
            MATCH (route:{ROUTE_LABEL} {{route_id: document.route}})
            CREATE (document)-[:{IN_ROUTE}]->(route)
            """
        )

        reusable_mentions = [
            {"doc_id": str(row["doc_id"]), "reusable_id": str(reusable_id)}
            for row in corpus_rows
            for reusable_id in set(row.get("reusable_ids", []))
        ]
        reusable_counts = Counter(row["reusable_id"] for row in reusable_mentions)
        reusable_rows = [
            {"reusable_id": key, "document_count": count}
            for key, count in sorted(reusable_counts.items())
        ]
        for batch in _batched(reusable_rows, batch_size):
            self._execute(
                f"""
                UNWIND $rows AS row
                CREATE (reusable:{REUSABLE_LABEL} {{
                  reusable_id: row.reusable_id,
                  document_count: row.document_count
                }})
                """,
                rows=batch,
            )
        for batch in _batched(reusable_mentions, batch_size * 2):
            self._execute(
                f"""
                UNWIND $rows AS row
                MATCH (document:{DOCUMENT_LABEL} {{doc_id: row.doc_id}})
                MATCH (reusable:{REUSABLE_LABEL} {{reusable_id: row.reusable_id}})
                CREATE (document)-[:{INCLUDES}]->(reusable)
                """,
                rows=batch,
            )

        raw_code_mentions = [
            (
                str(row["doc_id"]).split("::", 1)[0]
                if "::" in str(row["doc_id"])
                else "default",
                str(row["doc_id"]),
                key,
            )
            for row in corpus_rows
            for key in extract_code_entities(str(row["rendered_text"]))
        ]
        code_counts = Counter(
            (project_id, key) for project_id, _, key in raw_code_mentions
        )
        # Singletons cannot connect documents; very broad identifiers are hubs.
        code_counts = Counter(
            {key: count for key, count in code_counts.items() if 2 <= count <= 30}
        )
        code_mentions = [
            {
                "doc_id": doc_id,
                "code_id": f"{project_id}::{key}",
            }
            for project_id, doc_id, key in raw_code_mentions
            if (project_id, key) in code_counts
        ]
        code_rows = [
            {
                "code_id": f"{project_id}::{key}",
                "project_id": project_id,
                "key": key,
                "document_count": count,
            }
            for (project_id, key), count in sorted(code_counts.items())
        ]
        for batch in _batched(code_rows, batch_size):
            self._execute(
                f"""
                UNWIND $rows AS row
                CREATE (code:{CODE_LABEL} {{
                  code_id: row.code_id,
                  project_id: row.project_id,
                  key: row.key,
                  document_count: row.document_count
                }})
                """,
                rows=batch,
            )
        for batch in _batched(code_mentions, batch_size * 2):
            self._execute(
                f"""
                UNWIND $rows AS row
                MATCH (document:{DOCUMENT_LABEL} {{doc_id: row.doc_id}})
                MATCH (code:{CODE_LABEL} {{code_id: row.code_id}})
                CREATE (document)-[:{MENTIONS}]->(code)
                """,
                rows=batch,
            )

        semantic_counts = {"entities": 0, "predicates": 0, "claims": 0}
        if semantic_artifact is not None:
            semantic_counts = ingest_semantic_artifact(
                self._execute,
                semantic_artifact,
                batch_size=batch_size,
            )

        self._execute("CALL db.awaitIndexes(300)")
        return GitHubGraphStats(
            pages=len(page_rows),
            chunks=len(chunks),
            markdown_links=len(link_rows),
            routes=len(route_rows),
            reusable_entities=len(reusable_rows),
            reusable_mentions=len(reusable_mentions),
            code_entities=len(code_rows),
            code_mentions=len(code_mentions),
            ingest_seconds=time.perf_counter() - started,
            projects=len(records["projects"]),
            sections=len(records["sections"]),
            image_assets=len(records["image_assets"]),
            image_occurrences=len(records["image_occurrences"]),
            snapshot_sha256=str(snapshot["snapshot_sha256"]),
            **semantic_counts,
        )

    def graph_stats(self) -> GitHubGraphStats:
        def count_nodes(label: str) -> int:
            return int(self._execute(f"MATCH (node:{label}) RETURN count(node) AS count")[0]["count"])

        def count_text_units() -> int:
            return int(
                self._execute(
                    f"MATCH (node:{UNIT_LABEL} {{unit_type: 'text_chunk'}}) "
                    "RETURN count(node) AS count"
                )[0]["count"]
            )

        def count_relationships(kind: str) -> int:
            return int(
                self._execute(
                    f"MATCH ()-[relationship:{kind}]->() RETURN count(relationship) AS count"
                )[0]["count"]
            )

        return GitHubGraphStats(
            pages=count_nodes(DOCUMENT_LABEL),
            chunks=count_text_units(),
            markdown_links=count_relationships(LINKS_TO),
            routes=count_nodes(ROUTE_LABEL),
            reusable_entities=count_nodes(REUSABLE_LABEL),
            reusable_mentions=count_relationships(INCLUDES),
            code_entities=count_nodes(CODE_LABEL),
            code_mentions=count_relationships(MENTIONS),
            ingest_seconds=None,
            projects=count_nodes(PROJECT_LABEL),
            sections=count_nodes(SECTION_LABEL),
            image_assets=count_nodes(IMAGE_ASSET_LABEL),
            image_occurrences=count_nodes(IMAGE_OCCURRENCE_LABEL),
            entities=count_nodes(ENTITY_LABEL),
            predicates=count_nodes(PREDICATE_LABEL),
            claims=count_nodes(CLAIM_LABEL),
            snapshot_sha256=str(
                self.graph_snapshot().get("snapshot_sha256") or ""
            ),
        )

    def graph_snapshot(self) -> dict[str, Any]:
        rows = self._execute(
            f"MATCH (snapshot:{SNAPSHOT_LABEL} {{snapshot_id: 'docsqa'}}) "
            "RETURN snapshot{.*} AS snapshot"
        )
        return dict(rows[0]["snapshot"]) if rows else {}

    def expand_from_seeds(
        self,
        query_vector: np.ndarray,
        seed_ids: list[str],
        *,
        top_k: int,
        max_entity_degree: int,
        max_route_pages: int,
        max_link_hops: int = GRAPH_MAX_HOPS,
        max_graph_candidates_per_seed: int = GRAPH_CANDIDATES_PER_SEED,
        max_seed_count: int = GRAPH_SEED_LIMIT,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Run bounded typed traversal from the exact local-hybrid seeds."""

        unique_seeds = list(dict.fromkeys(str(value) for value in seed_ids if value))[
            :max_seed_count
        ]
        if not unique_seeds:
            raise ValueError("at least one Neo4j seed document is required")
        if max_link_hops not in (1, 2):
            raise ValueError("max_link_hops must be 1 or 2")
        if max_graph_candidates_per_seed < 1:
            raise ValueError("max_graph_candidates_per_seed must be positive")
        if max_seed_count < 1:
            raise ValueError("max_seed_count must be positive")
        link_pattern = f"[:{LINKS_TO}*1..{max_link_hops}]"
        started = time.perf_counter()
        rows = self._execute(
            f"""
            UNWIND $seeds AS seed_spec
            MATCH (seed:{DOCUMENT_LABEL} {{doc_id: seed_spec.doc_id}})
            CALL (seed) {{
              CALL (seed) {{
                MATCH path=(seed)-{link_pattern}-(neighbor:{DOCUMENT_LABEL})
                WITH DISTINCT neighbor, length(path) AS hops
                ORDER BY hops, neighbor.doc_id
                LIMIT $max_graph_candidates_per_seed
                RETURN neighbor,
                       CASE hops WHEN 1 THEN 1.0 ELSE 0.5 END AS path_weight,
                       'markdown_link_' + toString(hops) + 'hop' AS path_type,
                       hops
                UNION
                MATCH (seed)-[:{INCLUDES}]->(reusable:{REUSABLE_LABEL})
                      <-[:{INCLUDES}]-(neighbor:{DOCUMENT_LABEL})
                WHERE reusable.document_count <= $max_entity_degree
                WITH DISTINCT neighbor
                ORDER BY neighbor.doc_id
                LIMIT $max_graph_candidates_per_seed
                RETURN neighbor, 0.85 AS path_weight,
                       'shared_reusable' AS path_type, 1 AS hops
                UNION
                MATCH (seed)-[:{MENTIONS}]->(code:{CODE_LABEL})
                      <-[:{MENTIONS}]-(neighbor:{DOCUMENT_LABEL})
                WHERE code.document_count <= $max_entity_degree
                WITH DISTINCT neighbor
                ORDER BY neighbor.doc_id
                LIMIT $max_graph_candidates_per_seed
                RETURN neighbor, 0.70 AS path_weight,
                       'shared_code_entity' AS path_type, 1 AS hops
                UNION
                MATCH (seed)-[:{IN_ROUTE}]->(route:{ROUTE_LABEL})
                      <-[:{IN_ROUTE}]-(neighbor:{DOCUMENT_LABEL})
                WHERE route.document_count <= $max_route_pages
                WITH DISTINCT neighbor
                ORDER BY neighbor.doc_id
                LIMIT $max_graph_candidates_per_seed
                RETURN neighbor, 0.35 AS path_weight,
                       'bounded_route' AS path_type, 1 AS hops
                UNION
                MATCH (seed)-[:{HAS_SECTION}]->(:{SECTION_LABEL})
                      -[:{HAS_UNIT}]->(seed_unit:{UNIT_LABEL})
                MATCH (seed_claim:{CLAIM_LABEL})-[:{SUPPORTED_BY}]->(seed_unit)
                MATCH (entity:{ENTITY_LABEL})-[:{SUBJECT_OF}|{HAS_OBJECT}]-(seed_claim)
                WHERE entity.claim_count <= $max_entity_degree
                MATCH (entity)-[:{SUBJECT_OF}|{HAS_OBJECT}]-(neighbor_claim:{CLAIM_LABEL})
                MATCH (neighbor_claim)-[:{SUPPORTED_BY}]->(neighbor_unit:{UNIT_LABEL})
                MATCH (neighbor:{DOCUMENT_LABEL})-[:{HAS_SECTION}]->(:{SECTION_LABEL})
                      -[:{HAS_UNIT}]->(neighbor_unit)
                MATCH (neighbor_claim)-[:{USES_PREDICATE}]->(predicate:{PREDICATE_LABEL})
                WITH DISTINCT neighbor, predicate
                ORDER BY neighbor.doc_id, predicate.canonical_name
                LIMIT $max_graph_candidates_per_seed
                RETURN neighbor, 0.90 AS path_weight,
                       'kggen:' + predicate.canonical_name AS path_type, 1 AS hops
              }}
              WITH neighbor, path_weight, path_type, hops
              WHERE neighbor <> seed
                AND NOT neighbor.doc_id IN [item IN $seeds | item.doc_id]
              RETURN neighbor, path_weight, path_type, hops
              ORDER BY path_weight DESC, hops, neighbor.doc_id, path_type
              LIMIT $max_graph_candidates_per_seed
            }}
            MATCH (neighbor:{DOCUMENT_LABEL})-[:{HAS_SECTION}]->(:{SECTION_LABEL})
                  -[:{HAS_UNIT}]->(chunk:{UNIT_LABEL})
            WHERE chunk.text_embedding IS NOT NULL
            WITH neighbor, seed_spec, path_weight, path_type, hops,
                 max(vector.similarity.cosine(chunk.text_embedding, $query_vector))
                   AS query_relevance
            WITH neighbor,
                 collect(DISTINCT seed_spec.doc_id) AS expanded_from,
                 collect(DISTINCT path_type) AS path_types,
                 max(path_weight) AS path_weight,
                 max(query_relevance) AS query_relevance,
                 min(seed_spec.rank) AS seed_rank,
                 min(hops) AS hops
            WITH neighbor, expanded_from, path_types, query_relevance,
                 hops,
                 path_weight * CASE
                   WHEN query_relevance > 0.0 THEN query_relevance ELSE 0.0 END
                 + 1.0 / (60.0 + seed_rank) AS score
            RETURN neighbor.doc_id AS doc_id,
                   neighbor.title AS title,
                   score,
                   expanded_from,
                   path_types,
                   hops
            ORDER BY score DESC, doc_id
            LIMIT $top_k
            """,
            seeds=[
                {"doc_id": doc_id, "rank": rank}
                for rank, doc_id in enumerate(unique_seeds, start=1)
            ],
            query_vector=np.asarray(query_vector, dtype=np.float32).astype(float).tolist(),
            max_entity_degree=max_entity_degree,
            max_route_pages=max_route_pages,
            max_graph_candidates_per_seed=max_graph_candidates_per_seed,
            top_k=top_k,
        )
        results = [
            {
                "doc_id": str(row["doc_id"]),
                "title": str(row["title"]),
                "score": float(row["score"]),
                "expanded_from": [str(value) for value in row["expanded_from"]],
                "path_types": [str(value) for value in row["path_types"]],
                "hops": int(row["hops"]),
            }
            for row in rows
        ]
        path_types: Counter[str] = Counter(
            value for row in results for value in row["path_types"]
        )
        return results, {
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "graph_applied": True,
            "graph_candidates_added": len(results),
            "graph_path_types": dict(path_types),
            "max_link_hops": max_link_hops,
            "max_seed_count": max_seed_count,
            "max_graph_candidates_per_seed": max_graph_candidates_per_seed,
            "graph_candidate_upper_bound": (
                max_seed_count * max_graph_candidates_per_seed
            ),
            "neo4j_operation": "bounded typed multi-hop seed expansion",
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
    all_questions = _load_jsonl(questions_path)
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
    questions = [row for row in questions if args.split == "all" or row["split"] == args.split]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        raise ValueError("no questions remain after split and aspect filters")

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
    test_rows = [row for row in rows if row["split"] == "test"]

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
        "dataset_manifest": (
            file_identity(dataset_manifest_path, "dataset/manifest.json")
            if dataset_manifest_path.is_file()
            else None
        ),
        "selected_question_cohort": {
            "split": args.split,
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
        "split_filter": args.split,
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
        "test_overall": summarize(
            [{**row, "method": row["arm"]} for row in test_rows], ("method",)
        ),
        "evaluated_overall": summarize(
            [{**row, "method": row["arm"]} for row in rows], ("method",)
        ),
        "test_by_evidence_category": summarize(
            [{**row, "method": row["arm"]} for row in test_rows],
            ("evidence_category", "method"),
        ),
        "test_by_project": summarize(
            [{**row, "method": row["arm"]} for row in test_rows],
            ("project", "method"),
        ),
        "test_by_intent_category": summarize(
            [{**row, "method": row["arm"]} for row in test_rows],
            ("intent_category", "method"),
        ),
        "test_by_evidence_structure": summarize(
            [{**row, "method": row["arm"]} for row in test_rows],
            ("evidence_structure", "method"),
        ),
        "test_by_qrel_count": summarize(
            [{**row, "method": row["arm"]} for row in test_rows],
            ("qrel_count_group", "method"),
        ),
        "test_by_qrel_count_exact": summarize(
            [{**row, "method": row["arm"]} for row in test_rows],
            ("qrel_count_exact", "method"),
        ),
        "test_by_question_image": summarize(
            [{**row, "method": row["arm"]} for row in test_rows],
            ("question_image_group", "method"),
        ),
        "test_by_question_image_text_availability": summarize(
            [{**row, "method": row["arm"]} for row in test_rows],
            ("question_image_text_group", "method"),
        ),
        "paired_bootstrap_vs_hybrid": {
            metric: paired_bootstrap_delta(
                test_rows,
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
    parser.add_argument(
        "--split",
        choices=("all", "train", "validation", "dev", "test"),
        default="test",
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
    print(json.dumps(report["test_overall"], indent=2))


if __name__ == "__main__":
    main()
