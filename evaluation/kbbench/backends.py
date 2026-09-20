"""Filesystem search and Neo4j retrieval engines shared by evaluation and serving."""

from __future__ import annotations

import json
import re
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

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
    GRAPH_CANDIDATES_PER_SEED,
    GRAPH_MAX_HOPS,
    GRAPH_SEED_LIMIT,
)
from dsh_plugin.backend.semantic_store import ingest_semantic_artifact

from .retrieval import rrf


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
