"""Deterministic source normalization and IR scoring for DocsQA corpora."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import unquote, urlsplit

from dataset.scripts.records import read_jsonl


VERSION_SEGMENT = re.compile(
    r"^(?:enterprise-cloud|enterprise-server|free-pro-team)(?:@[^/]+)?$",
    re.IGNORECASE,
)
PROJECT_BY_DOCS_HOST = {
    "docs.github.com": "github-docs",
    "tailwindcss.com": "tailwind-css",
    "www.prisma.io": "prisma",
    "prisma.io": "prisma",
    "supabase.com": "supabase",
}


def _normalize_doc_id(value: str) -> str:
    path = unquote(str(value or "").strip()).replace("\\", "/")
    project = ""
    if "::" in path:
        project, path = path.split("::", 1)
        project = project.strip().casefold()
    path = path.split("#", 1)[0].split("?", 1)[0]
    path = re.sub(r"/+", "/", path)
    if path.endswith(".md"):
        path = path[:-3]
    if path.endswith("/index"):
        path = path[:-6] or "/"
    normalized = "/" + path.strip("/") if path.strip("/") else "/"
    return f"{project}::{normalized}" if project else normalized


class GitHubDocsSourceResolver:
    """Map agent citations (doc IDs, URLs, or repository paths) to qrel IDs."""

    def __init__(self, corpus_path: Path) -> None:
        self.corpus_path = corpus_path.resolve()
        canonical_doc_ids: dict[str, str] = {}
        alias_targets: dict[str, set[str]] = defaultdict(set)
        for row in read_jsonl(self.corpus_path):
            doc_id = _normalize_doc_id(str(row["doc_id"]))
            canonical_doc_ids[doc_id.casefold()] = doc_id
            project = str(row.get("project") or "").casefold()
            source_doc_id = _normalize_doc_id(str(row.get("source_doc_id") or ""))
            source_path = str(row.get("source_path") or "").replace("\\", "/")
            repository_source_path = str(
                row.get("repository_source_path") or source_path
            ).replace("\\", "/")
            route = str(row.get("route") or "")
            candidates = {
                doc_id,
                doc_id.lstrip("/"),
                source_doc_id,
                source_doc_id.lstrip("/"),
                source_path,
                repository_source_path,
                source_path.removeprefix("content/"),
                f"content/{source_path.removeprefix('content/')}",
                _normalize_doc_id(source_path),
                _normalize_doc_id(route),
            }
            if project:
                candidates.update(
                    {
                        _normalize_doc_id(f"{project}::{source_doc_id}"),
                        _normalize_doc_id(f"{project}::{source_path}"),
                        _normalize_doc_id(f"{project}::{repository_source_path}"),
                    }
                )
            for candidate in candidates:
                normalized = str(candidate).strip().casefold()
                if normalized:
                    alias_targets[normalized].add(doc_id)
        self.canonical_doc_ids = canonical_doc_ids
        self.aliases = {
            alias: next(iter(targets))
            for alias, targets in alias_targets.items()
            if len(targets) == 1
        }

    def resolve(self, source: str) -> str | None:
        raw = str(source or "").strip()
        if not raw:
            return None

        # A canonical ID is an identity, not an alias.  Check it before the
        # alias table because a document route may legitimately be shared by
        # several corpus rows (for example, the REST releases landing page and
        # its generated endpoint pages).  Such a collision must not make the
        # canonical landing-page ID unresolvable.
        if "::" in raw:
            canonical = self.canonical_doc_ids.get(
                _normalize_doc_id(raw).casefold()
            )
            if canonical:
                return canonical

        project_hint = ""
        if raw.startswith("viking://"):
            raw = urlsplit(raw).path
            marker = "/docsqa/"
            raw = raw.split(marker, 1)[-1] if marker in raw else raw
        elif "://" in raw:
            parsed = urlsplit(raw)
            project_hint = PROJECT_BY_DOCS_HOST.get(
                (parsed.hostname or "").casefold(), ""
            )
            raw = parsed.path

        raw = unquote(raw).replace("\\", "/")
        raw = raw.split("#", 1)[0].split("?", 1)[0]
        content_relative = raw.split("/content/", 1)[1] if "/content/" in raw else ""
        segments = [part for part in PurePosixPath(raw).parts if part not in {"/", ""}]
        if segments and segments[0].casefold() in set(PROJECT_BY_DOCS_HOST.values()):
            project_hint = segments.pop(0).casefold()
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
        if project_hint:
            candidates.extend(
                _normalize_doc_id(f"{project_hint}::{candidate}")
                for candidate in list(candidates)
            )
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


def retrieval_metrics(ranked_ids: list[str], relevant_ids: set[str]) -> dict[str, float]:
    """Score canonical document rankings; callers supply unique, exact IDs."""
    output: dict[str, float] = {}
    for cutoff in (1, 5, 10, 20):
        found = sum(doc_id in relevant_ids for doc_id in ranked_ids[:cutoff])
        output[f"recall_at_{cutoff}"] = found / max(1, len(relevant_ids))
        output[f"hit_at_{cutoff}"] = float(found > 0)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked_ids[:10], start=1)
        if doc_id in relevant_ids
    )
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(10, len(relevant_ids)) + 1)
    )
    output["ndcg_at_10"] = dcg / ideal if ideal else 0.0
    return output


def aspect_retrieval_metrics(
    ranked_ids: list[str],
    aspects: list[dict[str, Any]],
    *,
    alpha: float = 0.5,
) -> dict[str, float]:
    """Score coverage and novelty over frozen evidence-backed answer aspects.

    Aspects without a retrievable documentation ID are excluded from retrieval
    denominators but remain available to the final-answer judge. A document may
    support multiple aspects and receives the corresponding marginal gain.
    """

    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0, 1)")
    eligible: list[dict[str, Any]] = []
    for row in aspects:
        doc_ids = set(map(str, row.get("retrieval_doc_ids") or []))
        if not doc_ids:
            continue
        eligible.append(
            {
                "aspect_id": str(row["aspect_id"]),
                "weight": float(row.get("weight") or row.get("importance") or 1.0),
                "doc_ids": doc_ids,
            }
        )
    total_weight = sum(row["weight"] for row in eligible)
    unique_ranked = list(dict.fromkeys(map(str, ranked_ids)))
    output: dict[str, float] = {
        "aspect_count": float(len(aspects)),
        "retrieval_eligible_aspect_count": float(len(eligible)),
    }
    for cutoff in (1, 5, 10, 20):
        found = set(unique_ranked[:cutoff])
        covered_weight = sum(
            row["weight"] for row in eligible if row["doc_ids"] & found
        )
        output[f"weighted_aspect_recall_at_{cutoff}"] = (
            covered_weight / total_weight if total_weight else 0.0
        )

    def dcg(order: list[str]) -> float:
        counts = {row["aspect_id"]: 0 for row in eligible}
        score = 0.0
        for rank, doc_id in enumerate(order[:10], start=1):
            gain = 0.0
            for row in eligible:
                if doc_id in row["doc_ids"]:
                    gain += row["weight"] * (1.0 - alpha) ** counts[row["aspect_id"]]
                    counts[row["aspect_id"]] += 1
            score += gain / math.log2(rank + 1)
        return score

    candidates = sorted({doc_id for row in eligible for doc_id in row["doc_ids"]})
    ideal_order: list[str] = []
    ideal_counts = {row["aspect_id"]: 0 for row in eligible}
    remaining = set(candidates)
    for _ in range(min(10, len(remaining))):
        def marginal(doc_id: str) -> float:
            return sum(
                row["weight"] * (1.0 - alpha) ** ideal_counts[row["aspect_id"]]
                for row in eligible
                if doc_id in row["doc_ids"]
            )

        selected = min(remaining, key=lambda doc_id: (-marginal(doc_id), doc_id))
        ideal_order.append(selected)
        remaining.remove(selected)
        for row in eligible:
            if selected in row["doc_ids"]:
                ideal_counts[row["aspect_id"]] += 1
    ideal = dcg(ideal_order)
    output["alpha_ndcg_at_10"] = dcg(unique_ranked) / ideal if ideal else 0.0
    return output


def score_ranked_sources(
    ranked_ids: Iterable[str], relevant_ids: Iterable[str]
) -> dict[str, float]:
    """Normalize and deduplicate citations before applying the shared metrics."""
    ranked = list(dict.fromkeys(_normalize_doc_id(value) for value in ranked_ids))
    relevant = {_normalize_doc_id(value) for value in relevant_ids}
    metrics = retrieval_metrics(ranked, relevant)
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
