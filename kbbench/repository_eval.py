from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import re
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable
from urllib.parse import unquote, urlsplit

import numpy as np
from markdown_it import MarkdownIt

from .indexes import RetrievalIndex
from .metrics import paired_bootstrap, per_query_metrics, summarize
from .models import Document, MethodConfig, Query
from .retriever import Retriever


METHODS = (
    MethodConfig("bm25_r0_g0", use_hybrid=False, use_routing=False, use_graph=False),
    MethodConfig("hybrid_r0_g0", use_hybrid=True, use_routing=False, use_graph=False),
    MethodConfig("hybrid_r1_g0", use_hybrid=True, use_routing=True, use_graph=False),
    MethodConfig("hybrid_r0_g1", use_hybrid=True, use_routing=False, use_graph=True),
    MethodConfig("hybrid_r1_g1", use_hybrid=True, use_routing=True, use_graph=True),
)


@dataclass(frozen=True)
class RepositoryLink:
    source_id: str
    target_id: str
    fragment: str
    anchor_text: str
    context: str
    source_line: int | None


@dataclass(frozen=True)
class RepositoryCorpus:
    documents: list[Document]
    links: list[RepositoryLink]
    linked_code_count: int
    external_link_count: int
    broken_internal_count: int


class RepositoryLinkGraph:
    def __init__(
        self,
        documents: list[Document],
        links: Iterable[RepositoryLink],
        max_in_degree: int = 64,
    ) -> None:
        started = time.perf_counter()
        id_to_index = {document.doc_id: index for index, document in enumerate(documents)}
        unique_edges = {(link.source_id, link.target_id) for link in links if link.source_id != link.target_id}
        in_degree = Counter(target for _, target in unique_edges)
        neighbor_sets: list[set[int]] = [set() for _ in documents]
        kept = 0
        retained_edges: set[tuple[str, str]] = set()
        for source_id, target_id in sorted(unique_edges):
            if in_degree[target_id] > max_in_degree:
                continue
            source = id_to_index.get(source_id)
            target = id_to_index.get(target_id)
            if source is None or target is None:
                continue
            neighbor_sets[source].add(target)
            neighbor_sets[target].add(source)
            retained_edges.add((source_id, target_id))
            kept += 1
        self.neighbors = [np.asarray(sorted(values), dtype=np.int64) for values in neighbor_sets]
        self.edge_count = kept
        self.non_isolated = sum(bool(len(values)) for values in self.neighbors)
        self.build_seconds = time.perf_counter() - started
        self.filtered_hub_edges = len(unique_edges) - kept
        self.retained_edges = retained_edges


def load_repository(
    root: Path,
    github_repo: str,
    include_glob: str = "keps/**/*.md",
) -> RepositoryCorpus:
    root = root.resolve()
    markdown = MarkdownIt("commonmark", {"html": False, "linkify": False})
    include_globs = [value.strip() for value in include_glob.split(",") if value.strip()]
    markdown_paths = sorted(
        {
            path
            for pattern in include_globs
            for path in root.glob(pattern)
            if path.is_file()
        }
    )
    document_ids = {path.relative_to(root).as_posix() for path in markdown_paths}
    documents: list[Document] = []
    links: list[RepositoryLink] = []
    linked_code_count = 0
    external_link_count = 0
    broken_internal_count = 0

    for path in markdown_paths:
        doc_id = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        tokens = markdown.parse(text)
        title = _document_title(tokens, path, text)
        documents.append(Document(doc_id=doc_id, title=title, text=text, path=doc_id))
        extracted_links = list(_markdown_links(tokens))
        if path.suffix.lower() in {".rst", ".txt"}:
            extracted_links.extend(_rst_links(text))
        for href, anchor, context, source_line in extracted_links:
            resolved = resolve_repository_link(doc_id, href, github_repo, document_ids)
            if resolved is None:
                if _looks_external(href):
                    if _looks_like_repository_code_link(href, github_repo):
                        linked_code_count += 1
                    else:
                        external_link_count += 1
                elif _looks_internal_path(href):
                    broken_internal_count += 1
                continue
            target_id, fragment = resolved
            links.append(
                RepositoryLink(
                    source_id=doc_id,
                    target_id=target_id,
                    fragment=fragment,
                    anchor_text=anchor,
                    context=_bounded_context(context, anchor),
                    source_line=source_line,
                )
            )

    return RepositoryCorpus(
        documents=documents,
        links=links,
        linked_code_count=linked_code_count,
        external_link_count=external_link_count,
        broken_internal_count=broken_internal_count,
    )


def build_link_queries(
    links: Iterable[RepositoryLink],
    max_queries: int = 400,
) -> list[Query]:
    return [query for query, _ in build_link_query_pairs(links, max_queries=max_queries)]


def build_link_query_pairs(
    links: Iterable[RepositoryLink],
    max_queries: int = 400,
) -> list[tuple[Query, RepositoryLink]]:
    selected: dict[tuple[str, str], RepositoryLink] = {}
    for link in links:
        if link.source_id == link.target_id:
            continue
        if len(link.context.split()) < 6:
            continue
        selected.setdefault((link.source_id, link.target_id), link)
    ordered = sorted(
        selected.values(),
        key=lambda link: hashlib.sha256(
            f"{link.source_id}\0{link.target_id}\0{link.context}".encode("utf-8")
        ).hexdigest(),
    )[:max_queries]
    queries: list[tuple[Query, RepositoryLink]] = []
    for link in ordered:
        digest = hashlib.sha256(
            f"{link.source_id}\0{link.target_id}\0{link.context}".encode("utf-8")
        ).hexdigest()[:16]
        queries.append(
            (
                Query(
                query_id=f"repo-link-{digest}",
                text=link.context,
                relevant_ids=frozenset({link.target_id}),
                ),
                link,
            )
        )
    return queries


def resolve_repository_link(
    source_id: str,
    href: str,
    github_repo: str,
    document_ids: set[str],
) -> tuple[str, str] | None:
    parsed = urlsplit(href.strip())
    fragment = unquote(parsed.fragment or "")
    path: str | None = None
    if parsed.scheme in {"http", "https"}:
        if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
            return None
        parts = [part for part in parsed.path.split("/") if part]
        repo_parts = github_repo.strip("/").split("/")
        if len(parts) < 5 or parts[:2] != repo_parts or parts[2] not in {"blob", "tree"}:
            return None
        path = "/".join(parts[4:])
    elif parsed.scheme or parsed.netloc:
        return None
    else:
        raw_path = unquote(parsed.path)
        if not raw_path:
            return None
        if raw_path.startswith("/"):
            path = raw_path.lstrip("/")
        else:
            path = posixpath.normpath(posixpath.join(str(PurePosixPath(source_id).parent), raw_path))
    if path is None or path.startswith("../"):
        return None
    candidates = [path]
    if path.endswith("/"):
        candidates.extend(
            [f"{path}README.md", f"{path}index.md", f"{path}index.rst", f"{path}index.txt"]
        )
    elif not PurePosixPath(path).suffix:
        candidates.extend(
            [
                f"{path}.md",
                f"{path}.rst",
                f"{path}.txt",
                f"{path}/README.md",
                f"{path}/index.md",
                f"{path}/index.rst",
                f"{path}/index.txt",
            ]
        )
    for candidate in candidates:
        normalized = posixpath.normpath(candidate).lstrip("./")
        if normalized in document_ids:
            return normalized, fragment
    return None


def run_evaluation(
    corpus: RepositoryCorpus,
    queries: list[Query],
    dense_dimensions: int,
    route_count: int,
    max_in_degree: int,
) -> tuple[RetrievalIndex, list[dict[str, object]]]:
    index = RetrievalIndex(
        corpus.documents,
        dense_dimensions=dense_dimensions,
        route_count=route_count,
        random_seed=17,
    )
    index.graph = RepositoryLinkGraph(corpus.documents, corpus.links, max_in_degree=max_in_degree)
    retriever = Retriever(index)
    rows: list[dict[str, object]] = []
    for query in queries:
        for method in METHODS:
            started = time.perf_counter()
            hits = retriever.search(query.text, method, top_k=20)
            latency_ms = (time.perf_counter() - started) * 1000
            ranked_ids = [hit.doc_id for hit in hits]
            metrics = per_query_metrics(ranked_ids, query.relevant_ids)
            rows.append(
                {
                    "query_id": query.query_id,
                    "query": query.text,
                    "relevant_ids": sorted(query.relevant_ids),
                    "method": method.name,
                    "ranked_ids": ranked_ids,
                    "latency_ms": latency_ms,
                    **metrics,
                }
            )
    return index, rows


def _markdown_links(tokens: list[object]) -> Iterable[tuple[str, str, str, int | None]]:
    for token in tokens:
        if getattr(token, "type", "") != "inline":
            continue
        children = list(getattr(token, "children", None) or [])
        context = _inline_plain_text(children)
        source_map = getattr(token, "map", None)
        source_line = int(source_map[0]) + 1 if source_map else None
        index = 0
        while index < len(children):
            child = children[index]
            if child.type != "link_open":
                index += 1
                continue
            href = child.attrGet("href") or ""
            label_parts: list[str] = []
            index += 1
            while index < len(children) and children[index].type != "link_close":
                if children[index].type in {"text", "code_inline", "image"}:
                    label_parts.append(children[index].content)
                index += 1
            yield href, _normalize_space(" ".join(label_parts)), context, source_line
            index += 1


RST_DOC_ROLE = re.compile(r":doc:`(?:(?P<label>[^`<>]+?)\s*<(?P<target>[^`<>]+)>|(?P<plain>[^`<>]+))`")
RST_INLINE_LINK = re.compile(r"`(?P<label>[^`<>]+?)\s*<(?P<target>[^`<>]+)>`_")
RST_TOCTREE = re.compile(r"^\s*\.\.\s+toctree::\s*$")


def _rst_links(text: str) -> Iterable[tuple[str, str, str, int | None]]:
    """Extract repository-resolvable links from common Sphinx/RST forms.

    Django's documentation is primarily ``.txt`` reStructuredText. Treating
    those files as plain Markdown silently discarded its ``:doc:`` and
    ``toctree`` edges, so the benchmark graph could not represent the links
    that users actually navigate.
    """

    lines = text.splitlines()
    in_toctree = False
    for index, line in enumerate(lines):
        source_line = index + 1
        context = _normalize_space(line)
        for match in RST_DOC_ROLE.finditer(line):
            target = (match.group("target") or match.group("plain") or "").strip()
            label = (match.group("label") or match.group("plain") or target).strip()
            if target:
                yield target, label, context, source_line
        for match in RST_INLINE_LINK.finditer(line):
            target = match.group("target").strip()
            if target:
                yield target, match.group("label").strip(), context, source_line

        if RST_TOCTREE.match(line):
            in_toctree = True
            continue
        if not in_toctree:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith(":"):
            continue
        if line == line.lstrip():
            in_toctree = False
            continue
        target_match = re.match(r"(?:(?P<label>.+?)\s*<(?P<target>[^<>]+)>|(?P<plain>\S+))$", stripped)
        if target_match:
            target = (target_match.group("target") or target_match.group("plain") or "").strip()
            label = (target_match.group("label") or target).strip()
            if target and not target.startswith(("http://", "https://")):
                yield target, label, stripped, source_line


def _inline_plain_text(children: list[object]) -> str:
    parts: list[str] = []
    for child in children:
        if child.type in {"text", "code_inline", "image"}:
            parts.append(child.content)
        elif child.type in {"softbreak", "hardbreak"}:
            parts.append(" ")
    return _normalize_space(" ".join(parts))


def _document_title(tokens: list[object], path: Path, text: str = "") -> str:
    for index, token in enumerate(tokens[:-1]):
        if getattr(token, "type", "") == "heading_open" and getattr(token, "tag", "") == "h1":
            inline = tokens[index + 1]
            if getattr(inline, "type", "") == "inline":
                title = _inline_plain_text(list(getattr(inline, "children", None) or []))
                if title:
                    return title
    for line in text.splitlines()[:12]:
        stripped = line.strip()
        if stripped.lower().startswith("title:"):
            title = stripped.split(":", 1)[1].strip()
            if title:
                return title
    if path.name.lower() == "readme.md":
        return path.parent.name.replace("-", " ").replace("_", " ")
    return path.stem.replace("-", " ").replace("_", " ")


def _bounded_context(context: str, anchor: str, limit: int = 600) -> str:
    context = _normalize_space(context)
    if len(context) <= limit:
        return context
    position = context.lower().find(anchor.lower()) if anchor else -1
    if position < 0:
        return context[:limit].rsplit(" ", 1)[0]
    start = max(0, position - limit // 2)
    end = min(len(context), start + limit)
    return context[start:end].strip()


def _normalize_space(value: str) -> str:
    return " ".join(value.split())


def _looks_external(href: str) -> bool:
    return urlsplit(href.strip()).scheme in {"http", "https"}


def _looks_internal_path(href: str) -> bool:
    parsed = urlsplit(href.strip())
    return not parsed.scheme and bool(parsed.path)


def _looks_like_repository_code_link(href: str, github_repo: str) -> bool:
    parsed = urlsplit(href.strip())
    if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        return False
    parts = [part for part in parsed.path.split("/") if part]
    own = github_repo.strip("/").split("/")
    if len(parts) < 5 or parts[2] not in {"blob", "tree"}:
        return False
    if parts[:2] == own:
        return PurePosixPath("/".join(parts[4:])).suffix.lower() not in {".md", ".markdown"}
    return parts[0] == own[0]


def _git_revision(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0 and completed.stdout.strip():
        return completed.stdout.strip()
    digest = hashlib.sha256()
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
        digest.update(b"\0")
    return f"tree-sha256:{digest.hexdigest()}"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate hybrid, routing, and explicit Markdown-link graph retrieval on a pinned docs repository."
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--github-repo", default="kubernetes/enhancements")
    parser.add_argument("--include", default="keps/**/*.md")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=400)
    parser.add_argument("--dense-dimensions", type=int, default=96)
    parser.add_argument("--route-count", type=int, default=32)
    parser.add_argument("--max-in-degree", type=int, default=64)
    args = parser.parse_args()

    started = time.perf_counter()
    corpus = load_repository(args.repo, args.github_repo, args.include)
    query_pairs = build_link_query_pairs(corpus.links, max_queries=args.max_queries)
    queries = [query for query, _ in query_pairs]
    if not queries:
        raise SystemExit("No evaluable internal Markdown links found")
    index, rows = run_evaluation(
        corpus,
        queries,
        dense_dimensions=args.dense_dimensions,
        route_count=args.route_count,
        max_in_degree=args.max_in_degree,
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows)
    bootstrap = paired_bootstrap(rows, metric="ndcg@10")
    query_payload = [
        {
            "query_id": query.query_id,
            "text": query.text,
            "relevant_ids": sorted(query.relevant_ids),
            "source_id": link.source_id,
            "target_id": link.target_id,
            "target_fragment": link.fragment,
            "anchor_text": link.anchor_text,
            "source_line": link.source_line,
            "graph_edge_retained": (link.source_id, link.target_id) in index.graph.retained_edges,
            "role": "structural link-context recovery diagnostic; not a natural-question QA score",
        }
        for query, link in query_pairs
    ]
    _write_json(output / "queries.json", query_payload)
    _write_json(output / "summary.json", summary)
    _write_json(output / "paired_bootstrap_ndcg10.json", bootstrap)
    with (output / "per_query.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "benchmark": "repository-link-context-recovery-v1",
        "claim_boundary": "Mechanism diagnostic only. Queries are corpus-derived link contexts and graph edges contain the linked target by construction.",
        "repository": args.github_repo,
        "revision": _git_revision(args.repo),
        "include": args.include,
        "documents": len(corpus.documents),
        "internal_document_links": len(corpus.links),
        "linked_code_urls": corpus.linked_code_count,
        "external_links": corpus.external_link_count,
        "broken_internal_links": corpus.broken_internal_count,
        "queries": len(queries),
        "queries_with_retained_graph_edge": sum(
            (link.source_id, link.target_id) in index.graph.retained_edges
            for _, link in query_pairs
        ),
        "methods": [asdict(method) for method in METHODS],
        "graph": {
            "edges": index.graph.edge_count,
            "non_isolated_documents": index.graph.non_isolated,
            "filtered_hub_edges": index.graph.filtered_hub_edges,
            "max_in_degree": args.max_in_degree,
        },
        "index_build_seconds": index.build_seconds,
        "total_seconds": time.perf_counter() - started,
        "paid_api_usd": 0,
        "summary_file": "summary.json",
        "rows_file": "per_query.jsonl",
    }
    _write_json(output / "manifest.json", manifest)
    print(json.dumps({"manifest": manifest, "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
