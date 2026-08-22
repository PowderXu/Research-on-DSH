from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .indexes import BM25Index, RetrievalIndex
from .models import Document, MethodConfig
from .repository_eval import RepositoryLinkGraph, _git_revision, load_repository
from .retriever import Retriever


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_./:-]+")
RESOURCE_ROOT = "viking://resources/techdocs"


@dataclass(frozen=True)
class Passage:
    text: str
    line_start: int
    line_end: int


@dataclass(frozen=True)
class DocumentSection:
    section_id: str
    document_id: str
    heading: str
    slug: str
    depth: int
    text: str
    line_start: int
    line_end: int


class TechdocsIndexService:
    """Small local HTTP adapter used by the DSH plugin and reproducible trials."""

    def __init__(
        self,
        repository_root: Path,
        github_repository: str,
        include_glob: str = "keps/**/*.md",
        dense_dimensions: int = 96,
        route_count: int = 32,
        max_in_degree: int = 64,
        default_method: str = "bm25",
        title_repeats: int = 2,
        bm25_max_df: float = 1.0,
        graph_repository_root: Path | None = None,
    ) -> None:
        started = time.perf_counter()
        self.repository_root = repository_root.resolve()
        self.github_repository = github_repository
        self.revision = _git_revision(self.repository_root)
        self.corpus = load_repository(self.repository_root, github_repository, include_glob)
        self.graph_repository_root = (
            graph_repository_root.resolve() if graph_repository_root is not None else None
        )
        graph_corpus = self.corpus
        if self.graph_repository_root is not None:
            graph_corpus = load_repository(
                self.graph_repository_root,
                github_repository,
                include_glob,
            )
            if [document.doc_id for document in graph_corpus.documents] != [
                document.doc_id for document in self.corpus.documents
            ]:
                raise ValueError("text and graph repositories must contain identical document ids")
        self.documents = {document.doc_id: document for document in self.corpus.documents}
        self.index = RetrievalIndex(
            self.corpus.documents,
            dense_dimensions=dense_dimensions,
            route_count=route_count,
            random_seed=17,
        )
        self.index.bm25 = BM25Index(
            [f"{document.title}\n" * title_repeats + document.text for document in self.corpus.documents],
            max_df=bm25_max_df,
        )
        self.index.graph = RepositoryLinkGraph(
            self.corpus.documents,
            graph_corpus.links,
            max_in_degree=max_in_degree,
        )
        self.retriever = Retriever(self.index)
        self.sections = [
            section
            for document in self.corpus.documents
            for section in parse_document_sections(document)
        ]
        self.section_by_id = {section.section_id: section for section in self.sections}
        self.sections_by_document: dict[str, list[DocumentSection]] = {}
        for section in self.sections:
            self.sections_by_document.setdefault(section.document_id, []).append(section)
        section_documents = [
            Document(
                doc_id=section.section_id,
                title=f"{self.documents[section.document_id].title} — {section.heading}",
                text=section.text,
                path=self.documents[section.document_id].path,
            )
            for section in self.sections
        ]
        self.section_index = RetrievalIndex(
            section_documents,
            dense_dimensions=dense_dimensions,
            route_count=route_count,
            random_seed=17,
        )
        self.section_index.bm25 = BM25Index(
            [f"{document.title}\n" * title_repeats + document.text for document in section_documents],
            max_df=bm25_max_df,
        )
        self.section_retriever = Retriever(self.section_index)
        self.default_method = default_method
        self.title_repeats = title_repeats
        self.bm25_max_df = bm25_max_df
        self.build_seconds = time.perf_counter() - started

    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "repository": self.github_repository,
            "revision": self.revision,
            "documents": len(self.documents),
            "sections": len(self.sections),
            "graphEdges": self.index.graph.edge_count,
            "graphSource": str(self.graph_repository_root or self.repository_root),
            "defaultMethod": self.default_method,
            "titleRepeats": self.title_repeats,
            "bm25MaxDf": self.bm25_max_df,
            "paidUsd": 0,
        }

    def search(self, request: dict[str, Any]) -> dict[str, object]:
        started = time.perf_counter()
        query = str(request.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")
        scope = str(request.get("scope") or RESOURCE_ROOT).rstrip("/")
        if scope != RESOURCE_ROOT and not scope.startswith(f"{RESOURCE_ROOT}/"):
            raise ValueError(f"scope must be below {RESOURCE_ROOT}")
        result_limit = _bounded_integer(request.get("result_limit"), 1, 30, 8)
        evidence_token_budget = _bounded_integer(
            request.get("evidence_token_budget"), 200, 12000, 2200
        )
        passage_mode = str(request.get("passage_mode") or "window")
        if passage_mode not in {"window", "section", "adaptive"}:
            raise ValueError("passage_mode must be window, section, or adaptive")
        candidate_limit = _bounded_integer(request.get("candidate_limit"), result_limit, 200, 50)
        intent = request.get("intent") if isinstance(request.get("intent"), dict) else {}
        retrieval_query = intent_query(query, intent)
        graph_request = request.get("graph") or {}
        requested_edge_types = graph_request.get("edge_types") or []
        supported_edge_types = {"LINKS_TO", "LINKS_TO_SECTION"}
        edge_types = (
            [str(value) for value in requested_edge_types if str(value) in supported_edge_types]
            if requested_edge_types
            else sorted(supported_edge_types)
        )
        graph_requested = bool(graph_request.get("enabled", False)) and bool(edge_types)
        graph_seed_limit = _bounded_integer(graph_request.get("seed_limit"), 0, 50, 8)
        graph_neighbor_limit = _bounded_integer(graph_request.get("neighbor_limit"), 0, 10, 2)
        use_hybrid = self.default_method == "hybrid"
        method = MethodConfig(
            name=f"{self.default_method}_r0_g{int(graph_requested)}",
            use_hybrid=use_hybrid,
            use_routing=False,
            use_graph=graph_requested,
        )
        results: list[dict[str, object]] = []
        applied_method_name = method.name
        if passage_mode == "adaptive":
            section_method = MethodConfig(
                name=f"{self.default_method}_section_r0_g0",
                use_hybrid=use_hybrid,
                use_routing=False,
                use_graph=False,
            )
            applied_method_name = section_method.name
            if graph_requested:
                document_hits = self.retriever.search(
                    retrieval_query,
                    method,
                    top_k=candidate_limit,
                    graph_seed_count=graph_seed_limit,
                    graph_neighbor_limit=graph_neighbor_limit,
                )
                section_hits = [
                    (
                        best_document_section(
                            self.sections_by_document[hit.doc_id], retrieval_query
                        ),
                        hit.score,
                        _signals(hit.provenance, use_hybrid),
                    )
                    for hit in document_hits
                ]
                applied_method_name = f"{method.name}_structural"
            else:
                section_hits = [
                    (
                        self.section_by_id[hit.doc_id],
                        hit.score,
                        _signals(hit.provenance, use_hybrid),
                    )
                    for hit in self.section_retriever.search(
                        retrieval_query,
                        section_method,
                        top_k=candidate_limit,
                    )
                ]
            seen_documents: set[str] = set()
            for section, hit_score, hit_signals in section_hits:
                if section.document_id in seen_documents:
                    continue
                document = self.documents[section.document_id]
                passage = adaptive_section_passage(
                    section,
                    retrieval_query,
                    max_characters=min(4000, max(1200, evidence_token_budget * 4)),
                )
                results.append(
                    {
                        "sourceId": section.section_id,
                        "uri": document_uri(document.doc_id),
                        "title": document.title,
                        "section": section.slug,
                        "repoPath": document.path,
                        "commit": self.revision,
                        "lineStart": passage.line_start,
                        "lineEnd": passage.line_end,
                        "targetKind": "section",
                        "snippet": passage.text,
                        "score": hit_score,
                        "signals": [*hit_signals, "section"],
                        "expandedFrom": [],
                    }
                )
                seen_documents.add(section.document_id)
                if len(results) >= result_limit:
                    break
        else:
            hits = self.retriever.search(
                query,
                method,
                top_k=result_limit,
                graph_seed_count=graph_seed_limit,
                graph_neighbor_limit=graph_neighbor_limit,
            )
            for hit in hits:
                document = self.documents[hit.doc_id]
                passage = best_passage(
                    document,
                    query,
                    window_lines=28 if passage_mode == "section" else 12,
                )
                signals = _signals(hit.provenance, use_hybrid)
                results.append(
                    {
                        "sourceId": document.doc_id,
                        "uri": document_uri(document.doc_id),
                        "title": document.title,
                        "section": "",
                        "repoPath": document.path,
                        "commit": self.revision,
                        "lineStart": passage.line_start,
                        "lineEnd": passage.line_end,
                        "targetKind": "document",
                        "snippet": passage.text,
                        "score": hit.score,
                        "signals": signals,
                        "expandedFrom": [],
                    }
                )
        elapsed_ms = (time.perf_counter() - started) * 1000
        query_id = hashlib.sha256(
            f"{self.revision}\0{method.name}\0{retrieval_query}".encode("utf-8")
        ).hexdigest()[:20]
        response = {
            "queryId": query_id,
            "results": results,
            "trace": {
                "latencyMs": elapsed_ms,
                "stageCounts": {
                    "documents": len(self.documents),
                    "sections": len(self.sections),
                    "returned": len(results),
                    "graphEdges": self.index.graph.edge_count if graph_requested else 0,
                },
                "method": applied_method_name,
                "passageMode": passage_mode,
                "retrievalQuery": retrieval_query,
                "graphPolicy": {
                    "requested": bool(graph_request.get("enabled", False)),
                    "applied": graph_requested,
                    "seedLimit": graph_seed_limit,
                    "neighborLimit": graph_neighbor_limit,
                    "edgeTypes": edge_types,
                    "includeLinkedCode": False,
                },
                "paidUsd": 0,
                "indexRevision": self.revision,
            },
        }
        response["evidenceText"] = render_evidence_text(
            query_id, results, evidence_token_budget
        )
        return response

    def expand(self, request: dict[str, Any]) -> dict[str, object]:
        """Return bounded one-hop neighbors for explicit seed documents.

        The operation is deliberately primitive: the calling harness chooses the
        seeds, while the backend deterministically validates, expands, scores, and
        renders them. Both Codex and DSH call this exact endpoint through the same
        MCP server during the orchestration comparison.
        """

        started = time.perf_counter()
        query = str(request.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")
        raw_uris = request.get("seed_uris") or []
        seed_uris = raw_uris if isinstance(raw_uris, list) else [raw_uris]
        seed_ids: list[str] = []
        for raw_uri in seed_uris:
            doc_id = document_id_from_uri(str(raw_uri))
            if doc_id in self.documents and doc_id not in seed_ids:
                seed_ids.append(doc_id)
        if not seed_ids:
            raise ValueError("at least one in-scope seed URI is required")
        result_limit = _bounded_integer(request.get("result_limit"), 1, 30, 8)
        evidence_token_budget = _bounded_integer(
            request.get("evidence_token_budget"), 200, 12000, 2200
        )
        seed_set = set(seed_ids)
        query_terms = {
            token.lower() for token in TOKEN_PATTERN.findall(query) if len(token) > 2
        }
        candidates: dict[str, set[str]] = {}
        for seed_id in seed_ids:
            seed_index = self.index.id_to_index[seed_id]
            for neighbor_index in self.index.graph.neighbors[seed_index]:
                document = self.corpus.documents[int(neighbor_index)]
                if document.doc_id in seed_set:
                    continue
                candidates.setdefault(document.doc_id, set()).add(seed_id)

        ranked: list[tuple[float, str, Passage]] = []
        for doc_id in candidates:
            document = self.documents[doc_id]
            passage = best_passage(document, query)
            searchable = f"{document.title}\n{passage.text}".lower()
            overlap = sum(searchable.count(term) for term in query_terms)
            score = 1.0 + overlap / max(1, len(query_terms))
            ranked.append((score, doc_id, passage))
        ranked.sort(key=lambda value: (-value[0], value[1]))

        results: list[dict[str, object]] = []
        for score, doc_id, passage in ranked[:result_limit]:
            document = self.documents[doc_id]
            results.append(
                {
                    "sourceId": document.doc_id,
                    "uri": document_uri(document.doc_id),
                    "title": document.title,
                    "section": "",
                    "repoPath": document.path,
                    "commit": self.revision,
                    "lineStart": passage.line_start,
                    "lineEnd": passage.line_end,
                    "targetKind": "document",
                    "snippet": passage.text,
                    "score": score,
                    "signals": ["graph"],
                    "expandedFrom": [document_uri(value) for value in sorted(candidates[doc_id])],
                }
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        query_id = hashlib.sha256(
            f"{self.revision}\0expand\0{query}\0{'|'.join(sorted(seed_ids))}".encode("utf-8")
        ).hexdigest()[:20]
        response: dict[str, object] = {
            "queryId": query_id,
            "results": results,
            "trace": {
                "latencyMs": elapsed_ms,
                "stageCounts": {
                    "seeds": len(seed_ids),
                    "neighbors": len(candidates),
                    "returned": len(results),
                },
                "method": "explicit_link_expand",
                "paidUsd": 0,
                "indexRevision": self.revision,
            },
        }
        response["evidenceText"] = render_evidence_text(
            query_id, results, evidence_token_budget
        )
        return response

    def fetch(self, request: dict[str, Any]) -> dict[str, object]:
        raw_uris = request.get("uris") or []
        uris = raw_uris if isinstance(raw_uris, list) else [raw_uris]
        token_budget = _bounded_integer(request.get("token_budget"), 200, 12000, 2200)
        character_budget = token_budget * 4
        results: list[dict[str, object]] = []
        used = 0
        for raw_uri in uris:
            doc_id = document_id_from_uri(str(raw_uri))
            document = self.documents.get(doc_id)
            if document is None:
                continue
            remaining = max(0, character_budget - used)
            if remaining == 0:
                break
            text = document.text[:remaining]
            used += len(text)
            results.append(
                {
                    "sourceId": document.doc_id,
                    "uri": document_uri(document.doc_id),
                    "repoPath": document.path,
                    "commit": self.revision,
                    "text": text,
                }
            )
        if not results:
            raise ValueError("no in-scope technical-document URI was found")
        return {"results": results, "paidUsd": 0, "indexRevision": self.revision}


class TechdocsRequestHandler(BaseHTTPRequestHandler):
    service: TechdocsIndexService

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if urlsplit(self.path).path != "/health":
            self._write_error(404, "not found")
            return
        self._write_json(200, {"result": self.service.health()})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        try:
            request = self._read_json()
            path = urlsplit(self.path).path
            if path == "/v1/search":
                result = self.service.search(request)
            elif path == "/v1/expand":
                result = self.service.expand(request)
            elif path == "/v1/fetch":
                result = self.service.fetch(request)
            else:
                self._write_error(404, "not found")
                return
            self._write_json(200, {"result": result})
        except (ValueError, json.JSONDecodeError) as error:
            self._write_error(400, str(error))
        except Exception as error:  # pragma: no cover - defensive HTTP boundary
            self._write_error(500, str(error))

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 2_000_000:
            raise ValueError("invalid request body size")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _write_error(self, status: int, message: str) -> None:
        self._write_json(status, {"error": {"message": message}})

    def _write_json(self, status: int, value: object) -> None:
        body = json.dumps(value, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def best_passage(document: Document, query: str, window_lines: int = 12) -> Passage:
    lines = document.text.splitlines()
    if not lines:
        return Passage("", 1, 1)
    terms = {token.lower() for token in TOKEN_PATTERN.findall(query) if len(token) > 2}
    scores = [sum(line.lower().count(term) for term in terms) for line in lines]
    best_index = max(range(len(lines)), key=lambda index: scores[index]) if terms else 0
    start = max(0, best_index - window_lines // 3)
    end = min(len(lines), start + window_lines)
    start = max(0, end - window_lines)
    text = "\n".join(lines[start:end]).strip()
    return Passage(text=text, line_start=start + 1, line_end=end)


ATX_HEADING = re.compile(r"^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*#*\s*$")
HEADING_UNDERLINE = re.compile(r"^(?P<mark>[=\-~^\"'`:+*#<>_])(?P=mark){2,}\s*$")


def parse_document_sections(document: Document) -> list[DocumentSection]:
    """Parse Markdown and common Sphinx/RST headings into hierarchical sections.

    A section extends through its nested subsections and stops at the next heading
    of equal or higher rank. This gives retrieval a stable structural boundary
    without depending on an arbitrary line count.
    """

    lines = document.text.splitlines()
    if not lines:
        return [
            DocumentSection(
                section_id=f"{document.doc_id}::root@1",
                document_id=document.doc_id,
                heading=document.title,
                slug="root",
                depth=1,
                text="",
                line_start=1,
                line_end=1,
            )
        ]
    headings: list[tuple[int, int, str]] = []
    rst_ranks: dict[str, int] = {}
    index = 0
    while index < len(lines):
        atx = ATX_HEADING.match(lines[index].strip())
        if atx:
            headings.append((index, len(atx.group("marks")), atx.group("title").strip()))
            index += 1
            continue
        if index + 1 < len(lines) and lines[index].strip():
            underline = HEADING_UNDERLINE.match(lines[index + 1].strip())
            if underline:
                mark = underline.group("mark")
                if mark not in rst_ranks:
                    rst_ranks[mark] = len(rst_ranks) + 1
                headings.append((index, rst_ranks[mark], lines[index].strip()))
                index += 2
                continue
        index += 1
    if not headings:
        return [
            DocumentSection(
                section_id=f"{document.doc_id}::root@1",
                document_id=document.doc_id,
                heading=document.title,
                slug="root",
                depth=1,
                text=document.text,
                line_start=1,
                line_end=len(lines),
            )
        ]
    sections: list[DocumentSection] = []
    if headings[0][0] > 0 and any(line.strip() for line in lines[: headings[0][0]]):
        preamble_end = headings[0][0]
        sections.append(
            DocumentSection(
                section_id=f"{document.doc_id}::preamble@1",
                document_id=document.doc_id,
                heading=document.title,
                slug="preamble",
                depth=0,
                text="\n".join(lines[:preamble_end]).strip(),
                line_start=1,
                line_end=preamble_end,
            )
        )
    slug_counts: dict[str, int] = {}
    for position, (start, depth, heading) in enumerate(headings):
        end = len(lines)
        for next_start, next_depth, _ in headings[position + 1 :]:
            if next_depth <= depth:
                end = next_start
                break
        base_slug = section_slug(heading)
        slug_counts[base_slug] = slug_counts.get(base_slug, 0) + 1
        slug = base_slug if slug_counts[base_slug] == 1 else f"{base_slug}-{slug_counts[base_slug]}"
        sections.append(
            DocumentSection(
                section_id=f"{document.doc_id}::{slug}@{start + 1}",
                document_id=document.doc_id,
                heading=heading,
                slug=slug,
                depth=depth,
                text="\n".join(lines[start:end]).strip(),
                line_start=start + 1,
                line_end=end,
            )
        )
    return sections


def adaptive_section_passage(
    section: DocumentSection,
    query: str,
    max_characters: int,
) -> Passage:
    """Return the complete section when possible, otherwise select useful blocks.

    Large sections are split at blank-line boundaries. Blocks are ranked by the
    dynamic retrieval intent, and neighbors are added while space remains. The
    returned text stays in source order and marks non-contiguous omissions.
    """

    if len(section.text) <= max_characters:
        return Passage(section.text, section.line_start, section.line_end)
    raw_lines = section.text.splitlines()
    blocks: list[tuple[int, int, str]] = []
    block_start = 0
    in_fence = False
    for index, line in enumerate(raw_lines):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        if not in_fence and not line.strip():
            if index > block_start:
                text = "\n".join(raw_lines[block_start:index]).strip()
                if text:
                    blocks.append((block_start, index, text))
            block_start = index + 1
    if block_start < len(raw_lines):
        text = "\n".join(raw_lines[block_start:]).strip()
        if text:
            blocks.append((block_start, len(raw_lines), text))
    max_block_characters = max(300, max_characters // 3)
    bounded_blocks: list[tuple[int, int, str]] = []
    for start, end, text in blocks:
        if len(text) <= max_block_characters:
            bounded_blocks.append((start, end, text))
            continue
        chunk_start = start
        chunk_lines: list[str] = []
        chunk_characters = 0
        for line_index in range(start, end):
            line = raw_lines[line_index]
            addition = len(line) + (1 if chunk_lines else 0)
            if chunk_lines and chunk_characters + addition > max_block_characters:
                bounded_blocks.append(
                    (chunk_start, line_index, "\n".join(chunk_lines).strip())
                )
                chunk_start = line_index
                chunk_lines = []
                chunk_characters = 0
            chunk_lines.append(line)
            chunk_characters += addition
        if chunk_lines:
            bounded_blocks.append((chunk_start, end, "\n".join(chunk_lines).strip()))
    blocks = [block for block in bounded_blocks if block[2]]
    if not blocks:
        text = section.text[:max_characters].rstrip()
        return Passage(text, section.line_start, section.line_start + text.count("\n"))
    terms = {token.lower() for token in TOKEN_PATTERN.findall(query) if len(token) > 2}
    ranked = sorted(
        range(len(blocks)),
        key=lambda value: (
            -sum(blocks[value][2].lower().count(term) for term in terms),
            value,
        ),
    )
    selected: set[int] = {0}
    used = len(blocks[0][2])
    for best in ranked:
        for candidate in (best, best - 1, best + 1):
            if candidate < 0 or candidate >= len(blocks) or candidate in selected:
                continue
            addition = len(blocks[candidate][2]) + 5
            if used + addition > max_characters:
                continue
            selected.add(candidate)
            used += addition
        if used >= max_characters * 0.8:
            break
    ordered = sorted(selected)
    pieces: list[str] = []
    prior: int | None = None
    for selected_index in ordered:
        if prior is not None and selected_index != prior + 1:
            pieces.append("[…]")
        pieces.append(blocks[selected_index][2])
        prior = selected_index
    start = section.line_start + blocks[ordered[0]][0]
    end = section.line_start + blocks[ordered[-1]][1] - 1
    return Passage("\n\n".join(pieces), start, end)


def best_document_section(
    sections: list[DocumentSection], query: str
) -> DocumentSection:
    terms = {token.lower() for token in TOKEN_PATTERN.findall(query) if len(token) > 2}
    if not sections:
        raise ValueError("document has no structural sections")
    return max(
        sections,
        key=lambda section: (
            sum(
                f"{section.heading}\n{section.text}".lower().count(term)
                for term in terms
            ),
            -section.line_start,
        ),
    )


def intent_query(query: str, intent: dict[str, Any]) -> str:
    parts = [str(intent.get("retrievalQuery") or query).strip()]
    for key in ("identifiers", "documentationClaims", "sourceHints"):
        values = intent.get(key) or []
        if isinstance(values, list):
            parts.extend(str(value).strip() for value in values if str(value).strip())
    return "\n".join(dict.fromkeys(part for part in parts if part))


def section_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:80] or "section"


def render_evidence_text(
    query_id: str,
    results: list[dict[str, object]],
    token_budget: int,
) -> str:
    """Canonical model-facing rendering shared by every harness adapter."""

    character_budget = max(400, token_budget * 4)
    blocks: list[str] = []
    used = 0
    for index, result in enumerate(results, start=1):
        title = str(result.get("title") or result.get("uri") or "")
        uri = str(result.get("uri") or "")
        section = str(result.get("section") or "")
        repo_path = str(result.get("repoPath") or "")
        line_start = result.get("lineStart")
        line_end = result.get("lineEnd")
        commit = str(result.get("commit") or "")
        score = float(result.get("score") or 0.0)
        signals = ", ".join(str(value) for value in result.get("signals") or [])
        source = repo_path
        if source and isinstance(line_start, int):
            source += f":{line_start}"
            if isinstance(line_end, int) and line_end != line_start:
                source += f"-{line_end}"
        if source and commit:
            source += f" @ {commit}"
        expanded_from = [str(value) for value in result.get("expandedFrom") or []]
        lines = [
            f"[{index}] {title}",
            f"URI: {uri}{f'#{section}' if section else ''}",
        ]
        if source:
            lines.append(f"Source: {source}")
        if expanded_from:
            lines.append(f"Expanded from: {', '.join(expanded_from)}")
        lines.extend(
            [
                f"Score: {score:.4f}; signals: {signals or 'unspecified'}",
                str(result.get("snippet") or ""),
            ]
        )
        block = "\n".join(line for line in lines if line)
        if blocks and used + len(block) > character_budget:
            break
        blocks.append(block)
        used += len(block)
    if not blocks:
        return "No technical-document evidence found."
    return "\n\n".join(
        [f'<techdocs-evidence query-id="{query_id}">', *blocks, "</techdocs-evidence>"]
    )


def document_uri(doc_id: str) -> str:
    normalized = doc_id.lstrip("/")
    if not normalized or normalized.startswith("../"):
        raise ValueError("invalid document id")
    return f"{RESOURCE_ROOT}/{normalized}"


def document_id_from_uri(uri: str) -> str:
    value = uri.split("#", 1)[0].rstrip("/")
    prefix = f"{RESOURCE_ROOT}/"
    if not value.startswith(prefix):
        raise ValueError(f"URI must be below {RESOURCE_ROOT}")
    doc_id = unquote(value[len(prefix) :])
    if not doc_id or doc_id.startswith("../") or "/../" in f"/{doc_id}":
        raise ValueError("invalid technical-document URI")
    return doc_id


def _signals(provenance: dict[str, float], use_hybrid: bool) -> list[str]:
    values: list[str] = []
    if any(key in provenance for key in ("bm25", "routing_leaf_bm25")):
        values.append("bm25")
    if use_hybrid and "hnsw" in provenance:
        values.append("dense")
    if "graph" in provenance:
        values.append("graph")
    return values or ["bm25"]


def _bounded_integer(value: object, minimum: int, maximum: int, fallback: int) -> int:
    try:
        parsed = round(float(value))
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, parsed))


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a pinned repository index to the DSH technical-docs plugin.")
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--github-repo", default="kubernetes/enhancements")
    parser.add_argument("--include", default="keps/**/*.md")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1934)
    parser.add_argument("--dense-dimensions", type=int, default=96)
    parser.add_argument("--route-count", type=int, default=32)
    parser.add_argument("--max-in-degree", type=int, default=64)
    parser.add_argument("--default-method", choices=("bm25", "hybrid"), default="bm25")
    parser.add_argument("--title-repeats", type=int, default=2)
    parser.add_argument("--bm25-max-df", type=float, default=1.0)
    parser.add_argument(
        "--graph-repo",
        type=Path,
        help="Optional structurally linked mirror with document ids identical to --repo",
    )
    args = parser.parse_args()

    service = TechdocsIndexService(
        repository_root=args.repo,
        github_repository=args.github_repo,
        include_glob=args.include,
        dense_dimensions=args.dense_dimensions,
        route_count=args.route_count,
        max_in_degree=args.max_in_degree,
        default_method=args.default_method,
        title_repeats=max(1, args.title_repeats),
        bm25_max_df=args.bm25_max_df,
        graph_repository_root=args.graph_repo,
    )
    TechdocsRequestHandler.service = service
    server = ThreadingHTTPServer((args.host, args.port), TechdocsRequestHandler)
    print(
        json.dumps(
            {
                "listening": f"http://{args.host}:{args.port}",
                **service.health(),
                "buildSeconds": service.build_seconds,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
