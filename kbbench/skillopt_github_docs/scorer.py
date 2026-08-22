"""Deterministic source normalization and IR scoring for GitHub Docs."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import unquote, urlsplit


VERSION_SEGMENT = re.compile(
    r"^(?:enterprise-cloud|enterprise-server|free-pro-team)(?:@[^/]+)?$",
    re.IGNORECASE,
)


def _normalize_doc_id(value: str) -> str:
    path = unquote(str(value or "").strip()).replace("\\", "/")
    path = path.split("#", 1)[0].split("?", 1)[0]
    path = re.sub(r"/+", "/", path)
    if path.endswith(".md"):
        path = path[:-3]
    if path.endswith("/index"):
        path = path[:-6] or "/"
    return "/" + path.strip("/") if path.strip("/") else "/"


class GitHubDocsSourceResolver:
    """Map agent citations (doc IDs, URLs, or repository paths) to qrel IDs."""

    def __init__(self, corpus_path: Path) -> None:
        self.corpus_path = corpus_path.resolve()
        self.aliases: dict[str, str] = {}
        for line in self.corpus_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = _normalize_doc_id(str(row["doc_id"]))
            source_path = str(row.get("source_path") or "").replace("\\", "/")
            route = str(row.get("route") or "")
            candidates = {
                doc_id,
                doc_id.lstrip("/"),
                source_path,
                source_path.removeprefix("content/"),
                f"content/{source_path.removeprefix('content/')}",
                _normalize_doc_id(source_path),
                _normalize_doc_id(route),
            }
            for candidate in candidates:
                normalized = str(candidate).strip().casefold()
                if normalized:
                    self.aliases[normalized] = doc_id

    def resolve(self, source: str) -> str | None:
        raw = str(source or "").strip()
        if not raw:
            return None
        if raw.startswith("viking://"):
            raw = urlsplit(raw).path
            marker = "/techdocs/"
            raw = raw.split(marker, 1)[-1] if marker in raw else raw
        elif "://" in raw:
            raw = urlsplit(raw).path

        raw = unquote(raw).replace("\\", "/")
        raw = raw.split("#", 1)[0].split("?", 1)[0]
        content_relative = raw.split("/content/", 1)[1] if "/content/" in raw else ""
        segments = [part for part in PurePosixPath(raw).parts if part not in {"/", ""}]
        if segments and segments[0].casefold() == "en":
            segments = segments[1:]
        if segments and VERSION_SEGMENT.match(segments[0]):
            segments = segments[1:]
        raw = "/".join(segments)

        candidates = [
            raw,
            content_relative,
            raw.removeprefix("content/"),
            f"content/{raw.removeprefix('content/')}",
            _normalize_doc_id(raw),
            _normalize_doc_id(raw).lstrip("/"),
        ]
        for candidate in candidates:
            resolved = self.aliases.get(str(candidate).strip().casefold())
            if resolved:
                return resolved
        return None

    def resolve_ranked(self, sources: Iterable[str], limit: int = 20) -> list[str]:
        ranked: list[str] = []
        seen: set[str] = set()
        for source in sources:
            doc_id = self.resolve(str(source))
            if doc_id and doc_id not in seen:
                ranked.append(doc_id)
                seen.add(doc_id)
            if len(ranked) >= limit:
                break
        return ranked


def score_ranked_sources(
    ranked_ids: Iterable[str], relevant_ids: Iterable[str]
) -> dict[str, float]:
    ranked = list(dict.fromkeys(_normalize_doc_id(value) for value in ranked_ids))
    relevant = {_normalize_doc_id(value) for value in relevant_ids}
    metrics: dict[str, float] = {}
    for cutoff in (1, 5, 10, 20):
        found = sum(doc_id in relevant for doc_id in ranked[:cutoff])
        metrics[f"recall_at_{cutoff}"] = found / max(1, len(relevant))
        metrics[f"hit_at_{cutoff}"] = float(found > 0)

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:10], start=1)
        if doc_id in relevant
    )
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(10, len(relevant)) + 1)
    )
    metrics["ndcg_at_10"] = dcg / ideal if ideal else 0.0
    metrics["all_support_at_10"] = float(
        bool(relevant) and relevant.issubset(set(ranked[:10]))
    )
    metrics["hard"] = metrics["hit_at_10"]
    metrics["soft"] = (
        0.50 * metrics["ndcg_at_10"]
        + 0.30 * metrics["recall_at_10"]
        + 0.20 * metrics["hit_at_10"]
    )
    return metrics
