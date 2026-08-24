"""GitHub Docs HTTP backend consumed by the real DSH retrieval plugins."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from .github_docs_eval import (
    GitHubDocsRetriever,
    build_chunks,
    build_or_load_embeddings,
)
from .github_docs_three_arm_eval import Neo4jGitHubDocsGraphRAG
from .techdocs_service import (
    RESOURCE_ROOT,
    TechdocsRequestHandler,
    document_id_from_uri,
    document_uri,
    render_evidence_text,
)


TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _bounded(value: Any, minimum: int, maximum: int, fallback: int) -> int:
    try:
        parsed = round(float(value))
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, parsed))


def _doc_uri(doc_id: str) -> str:
    return RESOURCE_ROOT if doc_id == "/" else document_uri(doc_id)


def _doc_id(uri: str) -> str:
    normalized = str(uri).split("#", 1)[0].rstrip("/")
    if normalized == RESOURCE_ROOT:
        return "/"
    return "/" + document_id_from_uri(str(uri)).lstrip("/")


class GitHubDocsPluginService:
    """Shared BM25+HNSW base with optional Neo4j typed expansion."""

    def __init__(
        self,
        *,
        dataset_dir: Path,
        cache_dir: Path,
        arm: str,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        local_files_only: bool = True,
        retrieval_depth: int = 350,
        neo4j_uri: str = "bolt://127.0.0.1:7687",
        neo4j_username: str = "neo4j",
        neo4j_password: str = "secretgraph",
        neo4j_database: str = "neo4j",
        trace_path: Path | None = None,
    ) -> None:
        if arm not in {"hybrid", "neo4j"}:
            raise ValueError("GitHubDocsPluginService arm must be hybrid or neo4j")
        from sentence_transformers import SentenceTransformer

        started = time.perf_counter()
        self.arm = arm
        self.dataset_dir = dataset_dir.resolve()
        self.corpus_rows = _jsonl(self.dataset_dir / "corpus.jsonl")
        manifest_path = self.dataset_dir / "manifest.json"
        if not manifest_path.exists():
            manifest_path = self.dataset_dir / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.revision = str(manifest["corpus_revision"])
        self.rows_by_id = {str(row["doc_id"]): row for row in self.corpus_rows}
        self.chunks = build_chunks(self.corpus_rows)
        self.embedding_model = SentenceTransformer(
            embedding_model_name,
            device=device,
            local_files_only=local_files_only,
        )
        self.embeddings = build_or_load_embeddings(
            self.chunks,
            self.embedding_model,
            embedding_model_name,
            cache_dir.resolve(),
        )
        self.hybrid = GitHubDocsRetriever(
            self.corpus_rows,
            self.chunks,
            self.embeddings,
            self.embedding_model,
            reranker=None,
            retrieval_depth=retrieval_depth,
        )
        self.retrieval_depth = retrieval_depth
        self.graph: Neo4jGitHubDocsGraphRAG | None = None
        if arm == "neo4j":
            self.graph = Neo4jGitHubDocsGraphRAG(
                neo4j_uri, neo4j_username, neo4j_password, neo4j_database
            )
            stats = self.graph.graph_stats()
            if stats.pages != len(self.corpus_rows) or stats.chunks != len(self.chunks):
                self.graph.close()
                self.graph = None
                raise RuntimeError(
                    "Neo4j GitHub Docs graph does not match the frozen corpus; "
                    "run the namespaced ingest before evaluation"
                )
        self.trace_path = trace_path.resolve() if trace_path else None
        self._trace_lock = threading.Lock()
        self.events: list[dict[str, Any]] = []
        self.build_seconds = time.perf_counter() - started

    def close(self) -> None:
        if self.graph is not None:
            self.graph.close()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "arm": self.arm,
            "revision": self.revision,
            "documents": len(self.corpus_rows),
            "chunks": len(self.chunks),
            "method": "BM25+HNSW+RRF"
            if self.arm == "hybrid"
            else "BM25+HNSW seeds + Neo4j typed expansion",
            "buildSeconds": self.build_seconds,
            "paidUsd": 0,
        }

    def _query_vector(self, query: str) -> np.ndarray:
        return np.asarray(
            self.embedding_model.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            )[0],
            dtype=np.float32,
        )

    def _passage(self, row: dict[str, Any], query: str, line_count: int = 18) -> dict[str, Any]:
        lines = str(row.get("rendered_text") or "").splitlines()
        if not lines:
            return {"text": str(row.get("title") or ""), "start": 1, "end": 1, "section": ""}
        terms = {
            token.casefold()
            for token in TOKEN_RE.findall(query)
            if len(token) > 2
        }
        scores = [sum(line.casefold().count(term) for term in terms) for line in lines]
        best = max(range(len(lines)), key=lambda index: (scores[index], -index)) if terms else 0
        start = max(0, best - line_count // 3)
        end = min(len(lines), start + line_count)
        start = max(0, end - line_count)
        heading = ""
        for line in reversed(lines[: best + 1]):
            if line.startswith("#"):
                heading = line.lstrip("#").strip()
                break
        return {
            "text": "\n".join(lines[start:end]).strip(),
            "start": start + 1,
            "end": end,
            "section": heading,
        }

    def _result(
        self,
        doc_id: str,
        query: str,
        rank: int,
        *,
        signals: list[str],
        score: float | None = None,
        expanded_from: list[str] | None = None,
    ) -> dict[str, Any]:
        row = self.rows_by_id[doc_id]
        passage = self._passage(row, query)
        return {
            "sourceId": doc_id,
            "uri": _doc_uri(doc_id),
            "title": str(row.get("title") or doc_id),
            "section": passage["section"],
            "repoPath": str(row.get("source_path") or ""),
            "commit": self.revision,
            "lineStart": passage["start"],
            "lineEnd": passage["end"],
            "targetKind": "section",
            "snippet": passage["text"],
            "score": float(score if score is not None else 1.0 / (60.0 + rank)),
            "signals": signals,
            "expandedFrom": [_doc_uri(value) for value in expanded_from or []],
        }

    def _record(self, event: dict[str, Any]) -> None:
        event = {"time": time.time(), **event}
        with self._trace_lock:
            self.events.append(event)
            if self.trace_path:
                self.trace_path.parent.mkdir(parents=True, exist_ok=True)
                with self.trace_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def search(self, request: dict[str, Any]) -> dict[str, Any]:
        query = str(request.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")
        result_limit = _bounded(request.get("result_limit"), 1, 30, 10)
        token_budget = _bounded(request.get("evidence_token_budget"), 200, 12_000, 2600)
        started = time.perf_counter()
        ranked, diagnostics = self.hybrid.search(
            query, "bm25_hnsw_rrf", top_k=result_limit
        )
        results = [
            self._result(doc_id, query, rank, signals=["bm25", "hnsw", "rrf"])
            for rank, doc_id in enumerate(ranked, start=1)
        ]
        latency_ms = (time.perf_counter() - started) * 1000.0
        query_id = hashlib.sha256(
            f"{self.revision}\0hybrid\0{query}".encode()
        ).hexdigest()[:20]
        response = {
            "queryId": query_id,
            "results": results,
            "trace": {
                "latencyMs": latency_ms,
                "method": "bm25_hnsw_rrf",
                "stageCounts": {
                    "documents": len(self.corpus_rows),
                    "chunks": len(self.chunks),
                    "returned": len(results),
                },
                "graphPolicy": {"requested": False, "applied": False},
                "paidUsd": 0,
                "indexRevision": self.revision,
                **diagnostics,
            },
        }
        response["evidenceText"] = render_evidence_text(query_id, results, token_budget)
        self._record(
            {
                "operation": "search",
                "query_id": query_id,
                "query": query,
                "ranked_ids": ranked,
                "latency_ms": latency_ms,
            }
        )
        return response

    def expand(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.graph is None:
            raise ValueError("graph expansion is unavailable for the hybrid arm")
        query = str(request.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")
        raw_seed_uris = request.get("seed_uris") or []
        if not isinstance(raw_seed_uris, list):
            raise ValueError("seed_uris must be an array")
        seed_ids: list[str] = []
        for uri in raw_seed_uris:
            doc_id = _doc_id(str(uri))
            if doc_id in self.rows_by_id and doc_id not in seed_ids:
                seed_ids.append(doc_id)
        if not seed_ids:
            raise ValueError("at least one valid seed URI is required")
        result_limit = _bounded(request.get("result_limit"), 1, 30, 10)
        token_budget = _bounded(request.get("evidence_token_budget"), 200, 12_000, 2600)
        started = time.perf_counter()
        graph_rows, diagnostics = self.graph.expand_from_seeds(
            self._query_vector(query),
            seed_ids,
            top_k=result_limit,
            max_entity_degree=20,
            max_route_pages=20,
        )
        results = [
            self._result(
                row["doc_id"],
                query,
                rank,
                signals=["neo4j", *row["path_types"]],
                score=row["score"],
                expanded_from=row["expanded_from"],
            )
            for rank, row in enumerate(graph_rows, start=1)
        ]
        latency_ms = (time.perf_counter() - started) * 1000.0
        query_id = hashlib.sha256(
            f"{self.revision}\0neo4j-expand\0{query}\0{'|'.join(seed_ids)}".encode()
        ).hexdigest()[:20]
        response = {
            "queryId": query_id,
            "results": results,
            "trace": {
                "latencyMs": latency_ms,
                "method": "neo4j_typed_seed_expansion",
                "stageCounts": {
                    "seeds": len(seed_ids),
                    "returned": len(results),
                },
                "paidUsd": 0,
                "indexRevision": self.revision,
                **diagnostics,
            },
        }
        response["evidenceText"] = render_evidence_text(query_id, results, token_budget)
        self._record(
            {
                "operation": "expand",
                "query_id": query_id,
                "query": query,
                "seed_ids": seed_ids,
                "ranked_ids": [row["doc_id"] for row in graph_rows],
                "latency_ms": latency_ms,
            }
        )
        return response

    def fetch(self, request: dict[str, Any]) -> dict[str, Any]:
        raw_uris = request.get("uris") or []
        if not isinstance(raw_uris, list):
            raw_uris = [raw_uris]
        token_budget = _bounded(request.get("token_budget"), 200, 12_000, 2600)
        remaining = token_budget * 4
        results: list[dict[str, Any]] = []
        for uri in raw_uris:
            doc_id = _doc_id(str(uri))
            row = self.rows_by_id.get(doc_id)
            if row is None or remaining <= 0:
                continue
            text = str(row.get("rendered_text") or "")[:remaining]
            remaining -= len(text)
            results.append(
                {
                    "sourceId": doc_id,
                    "uri": _doc_uri(doc_id),
                    "repoPath": str(row.get("source_path") or ""),
                    "commit": self.revision,
                    "text": text,
                }
            )
        if not results:
            raise ValueError("no in-scope GitHub Docs URI was found")
        self._record(
            {
                "operation": "fetch",
                "ranked_ids": [row["sourceId"] for row in results],
            }
        )
        return {"results": results, "paidUsd": 0, "indexRevision": self.revision}


class GitHubDocsServiceHost:
    """Daemon-thread HTTP host used by the SkillOpt adapter process."""

    def __init__(self, service: GitHubDocsPluginService, host: str, port: int) -> None:
        handler = type(
            "BoundGitHubDocsRequestHandler",
            (TechdocsRequestHandler,),
            {"service": service},
        )
        self.service = service
        self.server = ThreadingHTTPServer((host, port), handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            name=f"github-docs-{service.arm}-service",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.service.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("evaluation/github_docs_v2"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/evaluation_cache/github_docs_v1"))
    parser.add_argument("--arm", choices=("hybrid", "neo4j"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687"))
    parser.add_argument("--neo4j-username", default=os.environ.get("NEO4J_USERNAME", "neo4j"))
    parser.add_argument("--neo4j-password", default=os.environ.get("NEO4J_PASSWORD", "secretgraph"))
    parser.add_argument("--neo4j-database", default=os.environ.get("NEO4J_DATABASE", "neo4j"))
    parser.add_argument("--trace-path", type=Path)
    args = parser.parse_args()
    port = args.port or (1935 if args.arm == "hybrid" else 1936)
    service = GitHubDocsPluginService(
        dataset_dir=args.dataset_dir,
        cache_dir=args.cache_dir,
        arm=args.arm,
        embedding_model_name=args.embedding_model,
        device=args.device,
        local_files_only=not args.allow_model_download,
        neo4j_uri=args.neo4j_uri,
        neo4j_username=args.neo4j_username,
        neo4j_password=args.neo4j_password,
        neo4j_database=args.neo4j_database,
        trace_path=args.trace_path,
    )
    host = GitHubDocsServiceHost(service, args.host, port)
    host.start()
    print(json.dumps({"listening": f"http://{args.host}:{port}", **service.health()}), flush=True)
    try:
        host.thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        host.close()


if __name__ == "__main__":
    main()
