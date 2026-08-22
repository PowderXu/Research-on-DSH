from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .github_docs_eval import (
    GitHubDocsRetriever,
    build_chunks,
    build_or_load_embeddings,
    retrieval_metrics,
    summarize,
)


ARMS = (
    "dsh_fs_search",
    "dsh_bm25_hnsw",
    "dsh_neo4j_graphrag",
)

GH_PAGE_LABEL = "KBGitHubDocPage"
GH_CHUNK_LABEL = "KBGitHubDocChunk"
GH_ROUTE_LABEL = "KBGitHubDocRoute"
GH_REUSABLE_LABEL = "KBGitHubDocReusable"
GH_CODE_LABEL = "KBGitHubDocCodeEntity"
GH_VECTOR_INDEX = "kb_github_docs_chunk_embedding_v1"
GH_FULLTEXT_INDEX = "kb_github_docs_chunk_fulltext_v1"
GH_PAGE_CONSTRAINT = "kb_github_docs_page_id_v1"
GH_CHUNK_CONSTRAINT = "kb_github_docs_chunk_id_v1"
GH_ROUTE_CONSTRAINT = "kb_github_docs_route_id_v1"
GH_REUSABLE_CONSTRAINT = "kb_github_docs_reusable_id_v1"
GH_CODE_CONSTRAINT = "kb_github_docs_code_key_v1"

FS_TOKEN_RE = re.compile(r"(?u)\b[A-Za-z0-9][A-Za-z0-9_.:/-]{2,}\b")
INLINE_CODE_RE = re.compile(r"`([^`\n]{2,100})`")
FLAG_RE = re.compile(r"(?<![\w-])--[A-Za-z0-9][\w-]{1,60}")
ENV_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,60}\b")
LUCENE_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*")

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
        max_calls: int = 3,
        max_matches_per_call: int = 250,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.content_root = (self.repo_root / "content").resolve()
        self.rg_path = rg_path.resolve()
        self.max_calls = max_calls
        self.max_matches_per_call = max_matches_per_call
        self.source_to_doc = {
            Path(str(row["source_path"])).as_posix(): str(row["doc_id"])
            for row in corpus_rows
        }
        if not self.rg_path.exists():
            raise FileNotFoundError(f"DSH packaged ripgrep not found: {self.rg_path}")

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
                    f"--regexp={pattern}",
                    "--glob=*.md",
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
            "fs_backend": "@deepseek-ai/dsh-tool-fs-search@0.1.0-rc.6",
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


class Neo4jGitHubDocsGraphRAG:
    """Neo4j Community + official neo4j-graphrag hybrid/Cypher retrieval."""

    RETRIEVAL_QUERY = f"""
    MATCH (node)-[:GH_FROM_PAGE]->(page:{GH_PAGE_LABEL})
    WITH page, max(score) AS raw_score
    ORDER BY raw_score DESC, page.doc_id
    LIMIT $base_page_count
    WITH collect({{page: page, raw_score: raw_score}}) AS pages
    CALL (pages) {{
      UNWIND range(0, size(pages) - 1) AS page_index
      WITH pages[page_index].page AS candidate,
           page_index + 1 AS page_rank
      RETURN candidate,
             1.0 / (60.0 + page_rank) AS base_score,
             0.0 AS graph_score,
             0 AS hops,
             'hybrid_seed' AS path_type
      UNION ALL
      UNWIND range(0,
        CASE
          WHEN size(pages) < $graph_seed_count THEN size(pages) - 1
          ELSE $graph_seed_count - 1
        END
      ) AS seed_index
      WITH pages[seed_index].page AS seed_page,
           seed_index + 1 AS seed_rank
      CALL (seed_page) {{
        MATCH (seed_page)-[:GH_LINKS_TO]-(neighbor:{GH_PAGE_LABEL})
        RETURN DISTINCT neighbor, 1.0 AS path_weight, 'markdown_link' AS path_type
        UNION
        MATCH (seed_page)-[:GH_INCLUDES]->(reusable:{GH_REUSABLE_LABEL})
              <-[:GH_INCLUDES]-(neighbor:{GH_PAGE_LABEL})
        WHERE reusable.page_count <= $max_entity_degree
        RETURN DISTINCT neighbor, 0.85 AS path_weight, 'shared_reusable' AS path_type
        UNION
        MATCH (seed_page)-[:GH_MENTIONS]->(code:{GH_CODE_LABEL})
              <-[:GH_MENTIONS]-(neighbor:{GH_PAGE_LABEL})
        WHERE code.page_count <= $max_entity_degree
        RETURN DISTINCT neighbor, 0.70 AS path_weight, 'shared_code_entity' AS path_type
        UNION
        MATCH (seed_page)-[:GH_IN_ROUTE]->(route:{GH_ROUTE_LABEL})
              <-[:GH_IN_ROUTE]-(neighbor:{GH_PAGE_LABEL})
        WHERE route.page_count <= $max_route_pages
        RETURN DISTINCT neighbor, 0.35 AS path_weight, 'bounded_route' AS path_type
      }}
      WITH seed_page, seed_rank, neighbor, path_weight, path_type
      WHERE neighbor <> seed_page
      MATCH (neighbor_chunk:{GH_CHUNK_LABEL})-[:GH_FROM_PAGE]->(neighbor)
      WITH neighbor AS candidate,
           seed_rank,
           path_weight,
           path_type,
           max(vector.similarity.cosine(
             neighbor_chunk.embedding, $query_vector
           )) AS query_relevance
      RETURN candidate,
             0.0 AS base_score,
             $graph_weight * (
               1.0 / (60.0 + seed_rank) +
               path_weight *
               CASE WHEN query_relevance > 0.0 THEN query_relevance ELSE 0.0 END /
               61.0
             ) AS graph_score,
             1 AS hops,
             path_type
    }}
    WITH candidate,
         sum(base_score) AS base_score,
         max(graph_score) AS graph_score,
         min(hops) AS hops,
         collect(DISTINCT path_type) AS path_types
    RETURN candidate.doc_id AS doc_id,
           candidate.title AS title,
           base_score + graph_score AS score,
           hops,
           path_types
    ORDER BY score DESC, doc_id
    LIMIT $return_k
    """

    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        from neo4j import GraphDatabase

        try:
            from neo4j_graphrag.retrievers import HybridCypherRetriever
        except ImportError:
            # This repository intentionally keeps Neo4j GraphRAG's NumPy 2
            # environment separate from the pinned sentence-transformers
            # environment. Appending (not prepending) its pure-Python packages
            # lets this runner reuse the official retriever without replacing
            # the already-loaded NumPy/SciPy stack.
            project_root = Path(__file__).resolve().parents[1]
            candidates = sorted(
                (project_root / ".venv-neo4j" / "lib").glob(
                    "python*/site-packages"
                )
            )
            if candidates:
                sys.path.append(str(candidates[-1]))
            try:
                from neo4j_graphrag.retrievers import HybridCypherRetriever
            except ImportError as exc:
                raise RuntimeError(
                    "Install requirements-neo4j.txt or expose neo4j-graphrag to this Python environment"
                ) from exc

        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.driver.verify_connectivity()
        self.database = database
        self.retriever_class = HybridCypherRetriever
        self.retriever: Any | None = None

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
        constraints = (
            (GH_PAGE_CONSTRAINT, GH_PAGE_LABEL, "doc_id"),
            (GH_CHUNK_CONSTRAINT, GH_CHUNK_LABEL, "chunk_id"),
            (GH_ROUTE_CONSTRAINT, GH_ROUTE_LABEL, "route_id"),
            (GH_REUSABLE_CONSTRAINT, GH_REUSABLE_LABEL, "reusable_id"),
            (GH_CODE_CONSTRAINT, GH_CODE_LABEL, "key"),
        )
        for name, label, property_name in constraints:
            self._execute(
                f"CREATE CONSTRAINT {name} IF NOT EXISTS "
                f"FOR (node:{label}) REQUIRE node.{property_name} IS UNIQUE"
            )
        self._execute(
            f"CREATE VECTOR INDEX {GH_VECTOR_INDEX} IF NOT EXISTS "
            f"FOR (chunk:{GH_CHUNK_LABEL}) ON chunk.embedding "
            "OPTIONS {indexConfig: {"
            f"`vector.dimensions`: {int(dimensions)}, "
            "`vector.similarity_function`: 'cosine'}}"
        )
        self._execute(
            f"CREATE FULLTEXT INDEX {GH_FULLTEXT_INDEX} IF NOT EXISTS "
            f"FOR (chunk:{GH_CHUNK_LABEL}) ON EACH [chunk.search_text]"
        )

    def ingest(
        self,
        corpus_rows: list[dict[str, Any]],
        chunks: list[Any],
        embeddings: np.ndarray,
        batch_size: int = 250,
    ) -> GitHubGraphStats:
        started = time.perf_counter()
        self.prepare_schema(int(embeddings.shape[1]))
        # Delete only this benchmark's namespaced labels. Existing Neo4j data,
        # including the historical TechQA benchmark, is preserved.
        self._execute(
            f"MATCH (node) WHERE node:{GH_PAGE_LABEL} OR node:{GH_CHUNK_LABEL} "
            f"OR node:{GH_ROUTE_LABEL} OR node:{GH_REUSABLE_LABEL} "
            f"OR node:{GH_CODE_LABEL} DETACH DELETE node"
        )

        page_rows = [
            {
                "doc_id": str(row["doc_id"]),
                "title": str(row["title"]),
                "source_path": str(row["source_path"]),
                "route": str(row["route"]),
                "content_type": str(row["content_type"]),
                "variant_conditioned": bool(row["variant_conditioned"]),
            }
            for row in corpus_rows
        ]
        for batch in _batched(page_rows, batch_size):
            self._execute(
                f"""
                UNWIND $rows AS row
                CREATE (page:{GH_PAGE_LABEL} {{
                  doc_id: row.doc_id,
                  title: row.title,
                  source_path: row.source_path,
                  route: row.route,
                  content_type: row.content_type,
                  variant_conditioned: row.variant_conditioned
                }})
                """,
                rows=batch,
            )

        chunk_rows = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "search_text": chunk.text,
                "embedding": embeddings[index].astype(float).tolist(),
            }
            for index, chunk in enumerate(chunks)
        ]
        for batch in _batched(chunk_rows, batch_size):
            self._execute(
                f"""
                UNWIND $rows AS row
                CREATE (chunk:{GH_CHUNK_LABEL} {{
                  chunk_id: row.chunk_id,
                  doc_id: row.doc_id,
                  title: row.title,
                  search_text: row.search_text,
                  embedding: row.embedding
                }})
                WITH chunk, row
                MATCH (page:{GH_PAGE_LABEL} {{doc_id: row.doc_id}})
                CREATE (chunk)-[:GH_FROM_PAGE]->(page)
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
                MATCH (source:{GH_PAGE_LABEL} {{doc_id: row.source}})
                MATCH (target:{GH_PAGE_LABEL} {{doc_id: row.target}})
                CREATE (source)-[:GH_LINKS_TO {{
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
            {"route_id": route, "page_count": count}
            for route, count in sorted(routes.items())
        ]
        for batch in _batched(route_rows, batch_size):
            self._execute(
                f"""
                UNWIND $rows AS row
                CREATE (route:{GH_ROUTE_LABEL} {{
                  route_id: row.route_id,
                  page_count: row.page_count
                }})
                """,
                rows=batch,
            )
        self._execute(
            f"""
            MATCH (page:{GH_PAGE_LABEL})
            MATCH (route:{GH_ROUTE_LABEL} {{route_id: page.route}})
            CREATE (page)-[:GH_IN_ROUTE]->(route)
            """
        )

        reusable_mentions = [
            {"doc_id": str(row["doc_id"]), "reusable_id": str(reusable_id)}
            for row in corpus_rows
            for reusable_id in set(row.get("reusable_ids", []))
        ]
        reusable_counts = Counter(row["reusable_id"] for row in reusable_mentions)
        reusable_rows = [
            {"reusable_id": key, "page_count": count}
            for key, count in sorted(reusable_counts.items())
        ]
        for batch in _batched(reusable_rows, batch_size):
            self._execute(
                f"""
                UNWIND $rows AS row
                CREATE (reusable:{GH_REUSABLE_LABEL} {{
                  reusable_id: row.reusable_id,
                  page_count: row.page_count
                }})
                """,
                rows=batch,
            )
        for batch in _batched(reusable_mentions, batch_size * 2):
            self._execute(
                f"""
                UNWIND $rows AS row
                MATCH (page:{GH_PAGE_LABEL} {{doc_id: row.doc_id}})
                MATCH (reusable:{GH_REUSABLE_LABEL} {{reusable_id: row.reusable_id}})
                CREATE (page)-[:GH_INCLUDES]->(reusable)
                """,
                rows=batch,
            )

        raw_code_mentions = [
            (str(row["doc_id"]), key)
            for row in corpus_rows
            for key in extract_code_entities(str(row["rendered_text"]))
        ]
        code_counts = Counter(key for _, key in raw_code_mentions)
        # Singletons cannot connect documents; very broad identifiers are hubs.
        code_counts = Counter(
            {key: count for key, count in code_counts.items() if 2 <= count <= 30}
        )
        code_mentions = [
            {"doc_id": doc_id, "key": key}
            for doc_id, key in raw_code_mentions
            if key in code_counts
        ]
        code_rows = [
            {"key": key, "page_count": count}
            for key, count in sorted(code_counts.items())
        ]
        for batch in _batched(code_rows, batch_size):
            self._execute(
                f"""
                UNWIND $rows AS row
                CREATE (code:{GH_CODE_LABEL} {{
                  key: row.key,
                  page_count: row.page_count
                }})
                """,
                rows=batch,
            )
        for batch in _batched(code_mentions, batch_size * 2):
            self._execute(
                f"""
                UNWIND $rows AS row
                MATCH (page:{GH_PAGE_LABEL} {{doc_id: row.doc_id}})
                MATCH (code:{GH_CODE_LABEL} {{key: row.key}})
                CREATE (page)-[:GH_MENTIONS]->(code)
                """,
                rows=batch,
            )

        self._execute("CALL db.awaitIndexes(300)")
        self.retriever = None
        return GitHubGraphStats(
            pages=len(page_rows),
            chunks=len(chunk_rows),
            markdown_links=len(link_rows),
            routes=len(route_rows),
            reusable_entities=len(reusable_rows),
            reusable_mentions=len(reusable_mentions),
            code_entities=len(code_rows),
            code_mentions=len(code_mentions),
            ingest_seconds=time.perf_counter() - started,
        )

    def graph_stats(self) -> GitHubGraphStats:
        def count_nodes(label: str) -> int:
            return int(self._execute(f"MATCH (node:{label}) RETURN count(node) AS count")[0]["count"])

        def count_relationships(kind: str) -> int:
            return int(
                self._execute(
                    f"MATCH ()-[relationship:{kind}]->() RETURN count(relationship) AS count"
                )[0]["count"]
            )

        return GitHubGraphStats(
            pages=count_nodes(GH_PAGE_LABEL),
            chunks=count_nodes(GH_CHUNK_LABEL),
            markdown_links=count_relationships("GH_LINKS_TO"),
            routes=count_nodes(GH_ROUTE_LABEL),
            reusable_entities=count_nodes(GH_REUSABLE_LABEL),
            reusable_mentions=count_relationships("GH_INCLUDES"),
            code_entities=count_nodes(GH_CODE_LABEL),
            code_mentions=count_relationships("GH_MENTIONS"),
            ingest_seconds=None,
        )

    @staticmethod
    def _result_formatter(record: Any) -> Any:
        from neo4j_graphrag.types import RetrieverResultItem

        return RetrieverResultItem(
            content={"doc_id": str(record["doc_id"]), "title": str(record["title"])},
            metadata={
                "score": float(record["score"]),
                "hops": int(record["hops"]),
                "path_types": list(record["path_types"]),
            },
        )

    def initialize_retriever(self) -> None:
        self.retriever = self.retriever_class(
            self.driver,
            GH_VECTOR_INDEX,
            GH_FULLTEXT_INDEX,
            self.RETRIEVAL_QUERY,
            result_formatter=self._result_formatter,
            neo4j_database=self.database,
        )

    def search(
        self,
        query: str,
        query_vector: np.ndarray,
        *,
        top_k: int,
        retrieval_depth: int,
        graph_seed_count: int,
        graph_weight: float,
        max_entity_degree: int,
        max_route_pages: int,
    ) -> tuple[list[str], dict[str, Any]]:
        if self.retriever is None:
            self.initialize_retriever()
        started = time.perf_counter()
        result = self.retriever.search(
            query_text=lucene_query_text(query),
            query_vector=np.asarray(query_vector, dtype=np.float32).astype(float).tolist(),
            top_k=retrieval_depth,
            effective_search_ratio=2,
            ranker="linear",
            alpha=0.5,
            query_params={
                "base_page_count": retrieval_depth,
                "graph_seed_count": graph_seed_count,
                "graph_weight": graph_weight,
                "max_entity_degree": max_entity_degree,
                "max_route_pages": max_route_pages,
                "return_k": top_k,
            },
        )
        ranked: list[str] = []
        path_types: Counter[str] = Counter()
        graph_hits = 0
        graph_boosted = 0
        for item in result.items:
            doc_id = str(item.content["doc_id"])
            if doc_id in ranked:
                continue
            ranked.append(doc_id)
            metadata = item.metadata or {}
            paths = [str(value) for value in metadata.get("path_types", [])]
            path_types.update(paths)
            graph_hits += int(int(metadata.get("hops", 0)) > 0)
            graph_boosted += int(any(value != "hybrid_seed" for value in paths))
        return ranked[:top_k], {
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "graph_applied": True,
            "graph_candidates_added": graph_hits,
            "graph_boosted_results": graph_boosted,
            "graph_path_types": dict(path_types),
            "selected_routes": [],
            "neo4j_retriever": "neo4j-graphrag HybridCypherRetriever",
        }

    def expand_from_seeds(
        self,
        query_vector: np.ndarray,
        seed_ids: list[str],
        *,
        top_k: int,
        max_entity_degree: int,
        max_route_pages: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Run a bounded, typed Neo4j traversal from agent-selected seeds.

        Hybrid seed search remains owned by ``HybridCypherRetriever``. This
        explicit expansion operation is plain parameterized Cypher because the
        seed identities come from the prior tool call rather than a second
        independent vector search.
        """

        unique_seeds = list(dict.fromkeys(str(value) for value in seed_ids if value))[:8]
        if not unique_seeds:
            raise ValueError("at least one Neo4j seed document is required")
        started = time.perf_counter()
        rows = self._execute(
            f"""
            UNWIND $seeds AS seed_spec
            MATCH (seed:{GH_PAGE_LABEL} {{doc_id: seed_spec.doc_id}})
            CALL (seed) {{
              MATCH (seed)-[:GH_LINKS_TO]-(neighbor:{GH_PAGE_LABEL})
              RETURN DISTINCT neighbor, 1.0 AS path_weight,
                     'markdown_link' AS path_type
              UNION
              MATCH (seed)-[:GH_INCLUDES]->(reusable:{GH_REUSABLE_LABEL})
                    <-[:GH_INCLUDES]-(neighbor:{GH_PAGE_LABEL})
              WHERE reusable.page_count <= $max_entity_degree
              RETURN DISTINCT neighbor, 0.85 AS path_weight,
                     'shared_reusable' AS path_type
              UNION
              MATCH (seed)-[:GH_MENTIONS]->(code:{GH_CODE_LABEL})
                    <-[:GH_MENTIONS]-(neighbor:{GH_PAGE_LABEL})
              WHERE code.page_count <= $max_entity_degree
              RETURN DISTINCT neighbor, 0.70 AS path_weight,
                     'shared_code_entity' AS path_type
              UNION
              MATCH (seed)-[:GH_IN_ROUTE]->(route:{GH_ROUTE_LABEL})
                    <-[:GH_IN_ROUTE]-(neighbor:{GH_PAGE_LABEL})
              WHERE route.page_count <= $max_route_pages
              RETURN DISTINCT neighbor, 0.35 AS path_weight,
                     'bounded_route' AS path_type
            }}
            WHERE neighbor <> seed
              AND NOT neighbor.doc_id IN [item IN $seeds | item.doc_id]
            MATCH (chunk:{GH_CHUNK_LABEL})-[:GH_FROM_PAGE]->(neighbor)
            WITH neighbor, seed_spec, path_weight, path_type,
                 max(vector.similarity.cosine(chunk.embedding, $query_vector))
                   AS query_relevance
            WITH neighbor,
                 collect(DISTINCT seed_spec.doc_id) AS expanded_from,
                 collect(DISTINCT path_type) AS path_types,
                 max(path_weight) AS path_weight,
                 max(query_relevance) AS query_relevance,
                 min(seed_spec.rank) AS seed_rank
            WITH neighbor, expanded_from, path_types, query_relevance,
                 path_weight * CASE
                   WHEN query_relevance > 0.0 THEN query_relevance ELSE 0.0 END
                 + 1.0 / (60.0 + seed_rank) AS score
            RETURN neighbor.doc_id AS doc_id,
                   neighbor.title AS title,
                   score,
                   expanded_from,
                   path_types
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
            top_k=top_k,
        )
        results = [
            {
                "doc_id": str(row["doc_id"]),
                "title": str(row["title"]),
                "score": float(row["score"]),
                "expanded_from": [str(value) for value in row["expanded_from"]],
                "path_types": [str(value) for value in row["path_types"]],
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
            "neo4j_operation": "bounded typed seed expansion",
        }


def lucene_query_text(query: str) -> str:
    # Keep punctuation out of Lucene's query language. Paths, brackets, and
    # quotes otherwise become operators or unterminated expressions. The cap
    # also prevents long issue bodies from exceeding Lucene's Boolean clause
    # limit; the dense side still embeds the complete original question.
    tokens: list[str] = []
    seen: set[str] = set()
    for raw in LUCENE_TOKEN_RE.findall(query):
        value = raw.casefold()
        if value in seen:
            continue
        seen.add(value)
        tokens.append(value)
        if len(tokens) >= 64:
            break
    return " ".join(tokens) if tokens else "emptyquery"


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

    corpus_rows = _load_jsonl(args.dataset_dir / "corpus.jsonl")
    questions = _load_jsonl(args.dataset_dir / "questions.jsonl")
    questions = [row for row in questions if args.split == "all" or row["split"] == args.split]
    if args.limit:
        questions = questions[: args.limit]
    chunks = build_chunks(corpus_rows)
    print(f"Loading embedding model {args.embedding_model}", flush=True)
    embedding_model = SentenceTransformer(
        args.embedding_model,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    embeddings = build_or_load_embeddings(
        chunks, embedding_model, args.embedding_model, args.cache_dir
    )
    hybrid = GitHubDocsRetriever(
        corpus_rows,
        chunks,
        embeddings,
        embedding_model,
        reranker=None,
        retrieval_depth=args.retrieval_depth,
    )
    filesystem = DshFilesystemSearch(
        args.repo_root,
        corpus_rows,
        args.dsh_rg,
        max_calls=args.fs_max_calls,
        max_matches_per_call=args.fs_max_matches,
    )
    graph: Neo4jGitHubDocsGraphRAG | None = None
    graph_stats: GitHubGraphStats | None = None
    if "dsh_neo4j_graphrag" in args.arms:
        graph = Neo4jGitHubDocsGraphRAG(
            args.neo4j_uri,
            args.neo4j_username,
            args.neo4j_password,
            args.neo4j_database,
        )
        if args.ingest:
            print("Ingesting the namespaced GitHub Docs graph into Neo4j", flush=True)
            graph_stats = graph.ingest(corpus_rows, chunks, embeddings)
            print(json.dumps(asdict(graph_stats), indent=2), flush=True)
        else:
            graph.initialize_retriever()
            graph_stats = graph.graph_stats()

    # Warm every selected arm before latency measurement.
    warm_query = str(questions[0]["query"])
    warm_vector = np.asarray(
        embedding_model.encode([warm_query], normalize_embeddings=True, show_progress_bar=False)[0],
        dtype=np.float32,
    )
    if "dsh_fs_search" in args.arms:
        filesystem.search(warm_query, args.top_k)
    if "dsh_bm25_hnsw" in args.arms:
        hybrid.search(warm_query, "bm25_hnsw_rrf", top_k=args.top_k)
    if graph is not None:
        graph.search(
            warm_query,
            warm_vector,
            top_k=args.top_k,
            retrieval_depth=args.retrieval_depth,
            graph_seed_count=args.graph_seed_count,
            graph_weight=args.graph_weight,
            max_entity_degree=args.max_entity_degree,
            max_route_pages=args.max_route_pages,
        )

    rows: list[dict[str, Any]] = []
    try:
        for arm in args.arms:
            print(f"Evaluating {arm} on {len(questions)} questions", flush=True)
            for index, question in enumerate(questions, start=1):
                query = str(question["query"])
                if arm == "dsh_fs_search":
                    ranked, diagnostics = filesystem.search(query, args.top_k)
                elif arm == "dsh_bm25_hnsw":
                    ranked, diagnostics = hybrid.search(
                        query, "bm25_hnsw_rrf", top_k=args.top_k
                    )
                elif arm == "dsh_neo4j_graphrag":
                    if graph is None:
                        raise RuntimeError("Neo4j graph arm was not initialized")
                    embedding_started = time.perf_counter()
                    query_vector = np.asarray(
                        embedding_model.encode(
                            [query], normalize_embeddings=True, show_progress_bar=False
                        )[0],
                        dtype=np.float32,
                    )
                    embedding_ms = (time.perf_counter() - embedding_started) * 1000.0
                    ranked, diagnostics = graph.search(
                        query,
                        query_vector,
                        top_k=args.top_k,
                        retrieval_depth=args.retrieval_depth,
                        graph_seed_count=args.graph_seed_count,
                        graph_weight=args.graph_weight,
                        max_entity_degree=args.max_entity_degree,
                        max_route_pages=args.max_route_pages,
                    )
                    diagnostics["neo4j_latency_ms"] = diagnostics["latency_ms"]
                    diagnostics["query_embedding_ms"] = embedding_ms
                    diagnostics["latency_ms"] += embedding_ms
                else:
                    raise ValueError(f"Unknown arm: {arm}")
                rows.append(
                    {
                        "arm": arm,
                        "question_id": question["question_id"],
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
                        "qrel_count_group": "1"
                        if len(set(question["qrel_ids"])) == 1 else "2+",
                        "relevant_ids": question["qrel_ids"],
                        "ranked_ids": ranked,
                        **diagnostics,
                        **retrieval_metrics(ranked, set(question["qrel_ids"])),
                    }
                )
                if index % 50 == 0:
                    print(f"  {index}/{len(questions)}", flush=True)
    finally:
        if graph is not None:
            graph.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_query.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    test_rows = [row for row in rows if row["split"] == "test"]
    report = {
        "benchmark": "GitHub Docs DSH three-arm plugin retrieval evaluation",
        "corpus_revision": args.corpus_revision,
        "documents": len(corpus_rows),
        "chunks": len(chunks),
        "questions": len(questions),
        "arms": list(args.arms),
        "split_filter": args.split,
        "top_k": args.top_k,
        "retrieval_depth": args.retrieval_depth,
        "graph_parameters": {
            "seed_count": args.graph_seed_count,
            "weight": args.graph_weight,
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
        "paired_bootstrap_vs_hybrid": {
            metric: paired_bootstrap_delta(
                test_rows,
                "dsh_neo4j_graphrag",
                "dsh_bm25_hnsw",
                metric,
            )
            for metric in ("recall_at_10", "hit_at_10", "ndcg_at_10")
            if {"dsh_neo4j_graphrag", "dsh_bm25_hnsw"}.issubset(args.arms)
        },
        "latency_scope": (
            "Warm plugin retrieval. Filesystem includes bounded DSH ripgrep subprocess calls; "
            "hybrid includes query embedding, BM25, HNSW, and RRF; Neo4j includes query "
            "embedding, driver/network time, official hybrid retrieval, and Cypher expansion. "
            "Index and graph construction are excluded and reported separately."
        ),
        "construct_note": (
            "Plugin-level retrieval comparison. DSH agent model/orchestration and answer "
            "generation are intentionally held out. The Neo4j graph is deterministic and "
            "query-blind; it is richer structural GraphRAG, not paid LLM entity extraction."
        ),
        "api_cost_usd": 0.0,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the matched three-arm DSH GitHub Docs retrieval benchmark"
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--corpus-revision", required=True)
    parser.add_argument("--dsh-rg", type=Path, required=True)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--retrieval-depth", type=int, default=350)
    parser.add_argument("--split", choices=("all", "dev", "test"), default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--fs-max-calls", type=int, default=3)
    parser.add_argument("--fs-max-matches", type=int, default=250)
    parser.add_argument("--neo4j-uri", default="bolt://127.0.0.1:7687")
    parser.add_argument("--neo4j-username", default="neo4j")
    parser.add_argument("--neo4j-password", default="secretgraph")
    parser.add_argument("--neo4j-database", default="neo4j")
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--graph-seed-count", type=int, default=8)
    parser.add_argument("--graph-weight", type=float, default=0.25)
    parser.add_argument("--max-entity-degree", type=int, default=20)
    parser.add_argument("--max-route-pages", type=int, default=20)
    args = parser.parse_args()
    report = evaluate(args)
    print(json.dumps(report["test_overall"], indent=2))


if __name__ == "__main__":
    main()
