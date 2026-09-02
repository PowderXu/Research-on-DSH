"""Build a local-only, answer-normalized DocsQA evaluation dataset.

The pipeline has four deliberately separate stages:

1. resolve every linked URL/fragment to a pinned local document section;
2. transcribe resolved image pixels into local text artifacts (never vectors);
3. synthesize a standalone answer from the accepted answer plus local evidence;
4. independently grade, repair, and re-grade until the conservative score is
   above the configured threshold, or reject the case with an inspectable reason.

External URLs remain provenance metadata only.  Generation, grading, and the
resulting retrieval corpus consume local dataset records and hash-verified local
image bytes.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import re
import shutil
import statistics
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, Field

from .image_evidence import image_data_url
from .llm_runtime import configured_openai_key, percentile, response_usage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "evaluation/dataset"
ANALYSIS_RUNS = PROJECT_ROOT / "results/runs/dataset-analysis"
RUBRIC_PATH = Path(__file__).resolve().parent / "rubrics/normalization_v1.json"


def _load_rubric_config(path: Path) -> tuple[dict[str, Any], str]:
    """Load one domain-neutral rubric and reject hidden override mechanisms."""

    raw = path.read_bytes()
    payload = json.loads(raw)
    required = {
        "schema_version",
        "rubric_version",
        "normalizer_version",
        "requirement_system",
        "requirement_rules",
        "generation_system",
        "generation_rules",
        "grading_system",
        "grading_rules",
        "grading_variants",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"rubric is missing required fields: {missing}")
    forbidden = {"question_overrides", "project_overrides", "dataset_overrides", "case_overrides"}
    present = sorted(forbidden & set(payload))
    if present:
        raise ValueError(f"rubric contains forbidden per-case override fields: {present}")
    if int(payload["schema_version"]) != 1:
        raise ValueError("unsupported rubric schema version")
    return payload, hashlib.sha256(raw).hexdigest()


RUBRIC_CONFIG, RUBRIC_SHA256 = _load_rubric_config(RUBRIC_PATH)
RUBRIC_VERSION = str(RUBRIC_CONFIG["rubric_version"])
NORMALIZER_VERSION = str(RUBRIC_CONFIG["normalizer_version"])
IMAGE_TEXT_VERSION = "literal-image-to-text-v1"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
TOKEN_RE = re.compile(r"[A-Za-z0-9_$.-]+")
URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(https?://[^)]+\)", re.IGNORECASE)
FENCED_CODE_RE = re.compile(r"(```.*?```|~~~.*?~~~)", re.DOTALL)
INLINE_CODE_RE = re.compile(r"(`[^`\n]*`)")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for",
    "from", "how", "i", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "use", "using", "what", "when", "where", "which", "with",
    "you", "your",
}
PROJECT_REDIRECT_NAMES = {
    "prisma": "prisma",
    "supabase": "supabase",
    "tailwind-css": "tailwind",
}


@dataclass(frozen=True)
class MarkdownSection:
    doc_id: str
    section_id: str
    heading: str
    anchor: str
    level: int
    start: int
    end: int
    text: str


class ImageTextArtifact(BaseModel):
    literal_text: str = Field(
        description="Exact legible text/OCR from the image, preserving important labels and values."
    )
    factual_description: str = Field(
        description="Literal description of visible UI, diagram, code, table, or state; no inferred solution."
    )
    visible_elements: list[str]
    unreadable_or_ambiguous: bool
    unreadable_details: list[str]


class DraftClaim(BaseModel):
    claim_id: str
    claim: str
    evidence_ids: list[str] = Field(min_length=1)


class DraftCitation(BaseModel):
    evidence_id: str
    reason: str


class NormalizedAnswerDraft(BaseModel):
    answer_text: str = Field(min_length=1)
    evidence_mode: Literal["direct", "compositional"]
    claims: list[DraftClaim] = Field(min_length=1)
    citations: list[DraftCitation] = Field(min_length=1)
    image_evidence_used: list[str]
    insufficient_evidence: bool
    missing_evidence: list[str]


class RequirementGrade(BaseModel):
    requirement_id: str
    requirement: str
    critical: bool
    coverage: Literal["full", "partial", "none", "contradicted"]
    evidence_ids: list[str]
    explanation: str


class ClaimGrade(BaseModel):
    claim_id: str
    support: Literal["full", "partial", "unsupported", "contradicted"]
    evidence_ids: list[str]
    explanation: str


class AnswerGrade(BaseModel):
    outcome: Literal["solves", "partially_solves", "does_not_solve", "unknown"]
    requirements: list[RequirementGrade] = Field(min_length=1)
    claims: list[ClaimGrade] = Field(min_length=1)
    correctness: int = Field(ge=0, le=4)
    actionability: int = Field(ge=0, le=4)
    directness: int = Field(ge=0, le=4)
    citation_integrity: bool
    unsupported_or_invented_information: list[str]
    repair_instructions: list[str]
    explanation: str
    confidence: Literal["high", "medium", "low"]


class UserRequirement(BaseModel):
    requirement_id: str
    requirement: str
    critical: bool
    source_quote: str = Field(min_length=1)
    requirement_kind: Literal["desired_outcome", "explicit_constraint", "requested_explanation"]


class RequirementPlan(BaseModel):
    requirements: list[UserRequirement] = Field(min_length=1)
    ambiguities: list[str]


def build_local_resolution_index(
    corpus: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Index local titles/headings for deterministic historical-link recovery."""

    documents: dict[str, dict[str, Any]] = {}
    token_to_docs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for doc_id, document in corpus.items():
        project = str(document.get("project") or doc_id.partition("::")[0])
        sections = parse_sections(document)
        headings = [section.heading for section in sections]
        anchors = [section.anchor for section in sections]
        searchable = " ".join(
            [
                str(document.get("title") or ""),
                str(document.get("short_title") or ""),
                str(document.get("source_doc_id") or ""),
                *headings,
            ]
        )
        terms = set(_terms(searchable.replace("/", " ").replace("-", " ")))
        title_terms = set(
            _terms(
                " ".join(
                    (
                        str(document.get("title") or ""),
                        str(document.get("short_title") or ""),
                    )
                )
            )
        )
        route_terms = set(
            _terms(str(document.get("source_doc_id") or "").replace("/", " ").replace("-", " "))
        )
        documents[doc_id] = {
            "project": project,
            "headings": headings,
            "anchors": anchors,
            "terms": terms,
            "title_terms": title_terms,
            "route_terms": route_terms,
            "source_doc_id": str(document.get("source_doc_id") or ""),
            "has_content": bool(sections),
        }
        for term in terms:
            token_to_docs[(project, term)].add(doc_id)
    return {"documents": documents, "token_to_docs": token_to_docs}


def recover_historical_targets(
    question: dict[str, Any],
    corpus: dict[str, dict[str, Any]],
    fragments: dict[str, list[str]],
    url_resolution: list[dict[str, Any]],
    index: dict[str, Any],
) -> list[str]:
    """Add a better local successor when a historical topic moved pages.

    Recovery uses only the URL route/fragment and pinned local titles/headings.
    It never uses accepted-answer text or a live page.  A candidate must improve
    topic-term coverage substantially, or contain the exact historical anchor.
    """

    recovered: list[str] = []
    documents = index["documents"]
    token_to_docs = index["token_to_docs"]
    project = str(question.get("project") or question.get("dataset") or "")
    for record in url_resolution:
        target = str(record.get("local_doc_id") or "")
        anchor = str(record.get("original_fragment") or record.get("route_intent") or "")
        anchor_terms = set(_terms(anchor.replace("-", " ")))
        if not target or not anchor_terms or target not in documents:
            continue
        current = documents[target]
        if anchor in set(current["anchors"]):
            continue
        current_coverage = len(anchor_terms & set(current["terms"])) / len(anchor_terms)
        route_terms = set(
            _terms(str(record.get("normalized_route") or "").replace("/", " ").replace("-", " "))
        ) - {
            "docs",
            "guide",
            "guides",
            "overview",
            "concepts",
            "components",
            "reference",
            "using",
            "customizing",
        }
        current_route_coverage = len(route_terms & set(current["terms"])) / max(1, len(route_terms))
        candidates: set[str] = set()
        for term in anchor_terms:
            candidates.update(token_to_docs.get((project, term), set()))
        scored: list[tuple[float, float, bool, str]] = []
        for doc_id in candidates:
            if doc_id == target:
                continue
            candidate = documents[doc_id]
            coverage = len(anchor_terms & set(candidate["terms"])) / len(anchor_terms)
            exact = anchor in set(candidate["anchors"])
            route_overlap = len(route_terms & set(candidate["terms"])) / max(1, len(route_terms))
            score = 1000.0 * float(exact) + 100.0 * coverage + 10.0 * route_overlap
            scored.append((score, coverage, exact, doc_id))
        scored.sort(reverse=True)
        if not scored:
            continue
        best_score, best_coverage, exact, best_doc_id = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else -math.inf
        strong_improvement = best_coverage >= 0.75 and best_coverage >= current_coverage + 0.20
        unambiguous = exact or not math.isfinite(second_score) or best_score - second_score >= 8.0
        best_route_coverage = (
            len(route_terms & set(documents[best_doc_id]["terms"])) / max(1, len(route_terms))
        )
        route_scope_preserved = not route_terms or best_route_coverage >= current_route_coverage
        exact_improves_scope = exact and route_scope_preserved and (
            current_coverage < 0.75 or best_coverage >= current_coverage + 0.20
        )
        if not (
            unambiguous
            and (exact_improves_scope or (strong_improvement and route_scope_preserved))
        ):
            continue
        fragments.setdefault(best_doc_id, []).append(anchor)
        fragments[best_doc_id] = list(dict.fromkeys(fragments[best_doc_id]))
        recovered.append(best_doc_id)
        record.update(
            {
                "semantic_local_doc_id": best_doc_id,
                "semantic_recovery_method": "exact_historical_anchor" if exact else "local_heading_topic_recovery",
                "semantic_recovery_score": round(best_score, 6),
                "semantic_recovery_margin": (
                    round(best_score - second_score, 6) if math.isfinite(second_score) else None
                ),
                "original_target_topic_coverage": round(current_coverage, 6),
                "recovered_target_topic_coverage": round(best_coverage, 6),
            }
        )
    return list(dict.fromkeys(recovered))


def expand_empty_routing_pages(
    question: dict[str, Any],
    document_ids: list[str],
    fragments: dict[str, list[str]],
    index: dict[str, Any],
    *,
    max_children: int = 3,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Replace contentless documentation indexes with query-ranked descendants.

    GitHub Docs uses contentless index Markdown files as navigation nodes.  They
    are valid link targets but cannot support an answer by themselves.  Expansion
    is restricted to pinned descendants of that exact route and ranked from the
    user question only; the accepted answer is deliberately unavailable here.
    """

    documents = index["documents"]
    token_to_docs = index["token_to_docs"]
    question_terms = set(_terms(str(question.get("query") or "")))
    expanded: list[str] = []
    records: list[dict[str, Any]] = []
    for doc_id in document_ids:
        current = documents.get(doc_id)
        if not current or current.get("has_content"):
            expanded.append(doc_id)
            continue
        project = str(current.get("project") or "")
        prefix = str(current.get("source_doc_id") or "").rstrip("/") + "/"
        project_size = max(
            1,
            sum(1 for row in documents.values() if str(row.get("project") or "") == project),
        )
        ranked: list[tuple[float, int, str]] = []
        for candidate_id, candidate in documents.items():
            route = str(candidate.get("source_doc_id") or "")
            if (
                str(candidate.get("project") or "") != project
                or not candidate.get("has_content")
                or not prefix.strip("/")
                or not route.startswith(prefix)
            ):
                continue
            title_terms = set(candidate.get("title_terms") or set())
            route_terms = set(candidate.get("route_terms") or set())
            all_terms = set(candidate.get("terms") or set())
            overlap = question_terms & all_terms
            if not overlap:
                continue
            score = 0.0
            for term in overlap:
                document_frequency = len(token_to_docs.get((project, term), set()))
                inverse_frequency = math.log1p(project_size / max(1, document_frequency))
                score += inverse_frequency
                if term in title_terms:
                    score += 3.0 * inverse_frequency
                if term in route_terms:
                    score += 1.5 * inverse_frequency
            # Prefer a nearer child when lexical evidence is otherwise equal.
            depth = max(1, route[len(prefix) :].count("/") + 1)
            score += 0.25 / depth
            ranked.append((score, len(overlap), candidate_id))
        ranked.sort(key=lambda row: (-row[0], -row[1], row[2]))
        selected = [candidate_id for score, _, candidate_id in ranked[:max_children] if score > 0]
        expanded.extend(selected)
        records.append(
            {
                "routing_doc_id": doc_id,
                "routing_source_doc_id": current.get("source_doc_id"),
                "method": "question_only_pinned_descendant_ranking",
                "selected_doc_ids": selected,
                "ranked_candidates": [
                    {
                        "doc_id": candidate_id,
                        "score": round(score, 6),
                        "matched_question_terms": overlap_count,
                    }
                    for score, overlap_count, candidate_id in ranked[:max_children]
                ],
            }
        )
        for candidate_id in selected:
            fragments.setdefault(candidate_id, [])
    return list(dict.fromkeys(expanded)), records


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(value: str) -> str:
    value = re.sub(r"<[^>]+>|\{.*?\}", " ", unquote(value))
    value = re.sub(r"[`*_~]", "", value).casefold()
    value = re.sub(r"[^\w\s-]", "", value)
    return re.sub(r"[-\s]+", "-", value).strip("-")


def _terms(value: str) -> list[str]:
    return [
        token.casefold().strip(".-")
        for token in TOKEN_RE.findall(value)
        if token.casefold().strip(".-") not in STOPWORDS
        and len(token.casefold().strip(".-")) > 1
    ]


def parse_sections(document: dict[str, Any]) -> list[MarkdownSection]:
    """Parse nested Markdown sections and keep a local document fallback."""

    doc_id = str(document["doc_id"])
    text = str(document.get("rendered_text") or document.get("raw_text") or "").strip()
    if not text:
        return []
    matches = list(HEADING_RE.finditer(text))
    sections: list[MarkdownSection] = []
    if not matches:
        return [
            MarkdownSection(
                doc_id=doc_id,
                section_id=f"{doc_id}#document",
                heading=str(document.get("title") or "Document"),
                anchor="document",
                level=1,
                start=0,
                end=len(text),
                text=text,
            )
        ]
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(
            MarkdownSection(
                doc_id=doc_id,
                section_id=f"{doc_id}#document-overview",
                heading=str(document.get("title") or "Document overview"),
                anchor="document-overview",
                level=1,
                start=0,
                end=matches[0].start(),
                text=preamble,
            )
        )
    for index, match in enumerate(matches):
        level = len(match.group(1))
        end = len(text)
        for later in matches[index + 1 :]:
            if len(later.group(1)) <= level:
                end = later.start()
                break
        heading = re.sub(r"\s+#+\s*$", "", match.group(2)).strip()
        anchor = _slug(heading) or f"section-{index + 1}"
        sections.append(
            MarkdownSection(
                doc_id=doc_id,
                section_id=f"{doc_id}#{anchor}",
                heading=heading,
                anchor=anchor,
                level=level,
                start=match.start(),
                end=end,
                text=text[match.start() : end].strip(),
            )
        )
    return sections


def _rank_section(
    section: MarkdownSection,
    fragment: str,
    question: str,
) -> float:
    anchor = _slug(fragment)
    if anchor and section.anchor == anchor:
        return 10_000.0
    fragment_terms = set(_terms(anchor.replace("-", " ")))
    heading_terms = set(_terms(section.heading))
    body_terms = set(_terms(section.text[:8_000]))
    # Section selection is question-only.  Using the accepted answer here leaks
    # the reference solution into evidence selection and can make a broad or
    # redirected page appear more specific than the user's actual request.
    query_terms = set(_terms(question))
    score = 0.0
    if fragment_terms:
        score += 80.0 * len(fragment_terms & heading_terms) / len(fragment_terms)
        score += 30.0 * len(fragment_terms & body_terms) / len(fragment_terms)
        score += 25.0 * len(fragment_terms & heading_terms) / max(1, len(heading_terms))
    if query_terms:
        score += 12.0 * len(query_terms & heading_terms) / max(1, len(heading_terms))
        score += 8.0 * len(query_terms & body_terms) / len(query_terms)
    score += 1.0 / (1.0 + section.start / 10_000)
    return score


def select_sections(
    document: dict[str, Any],
    fragments: Iterable[str],
    question: str,
    _answer: str = "",
    *,
    max_sections: int = 4,
    max_chars: int = 18_000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return evidence sections plus an inspectable fragment-resolution map."""

    sections = parse_sections(document)
    if not sections:
        return [], []
    fragments = [value for value in dict.fromkeys(_slug(str(item)) for item in fragments) if value]
    selected: dict[str, tuple[MarkdownSection, float, str]] = {}
    mappings: list[dict[str, Any]] = []
    search_fragments = fragments or [""]
    for fragment in search_fragments:
        ranked = sorted(
            ((section, _rank_section(section, fragment, question)) for section in sections),
            key=lambda item: (-item[1], item[0].start),
        )
        best, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -math.inf
        exact = bool(fragment and best.anchor == fragment)
        margin = best_score - second_score
        # A stale fragment can be accepted deterministically when lexical evidence
        # clearly favors one local heading.  A low-margin mapping remains recorded
        # as ambiguous and cannot be the sole evidence for a passing answer.
        ambiguous = bool(fragment and not exact and (best_score < 34.0 or margin < 4.0))
        mappings.append(
            {
                "original_fragment": fragment,
                "canonical_heading": best.heading,
                "canonical_anchor": best.anchor,
                "section_id": best.section_id,
                "method": "exact_anchor" if exact else "ranked_local_heading" if fragment else "question_guided_local_section",
                "score": round(best_score, 6),
                "margin": round(margin, 6) if math.isfinite(margin) else None,
                "ambiguous": ambiguous,
            }
        )
        selected[best.section_id] = (best, best_score, "anchor" if fragment else "query")

    # Add question-ranked sections. This repairs stale anchor vocabulary
    # without using a live page or parametric facts.
    ranked_all = sorted(
        ((section, _rank_section(section, "", question)) for section in sections),
        key=lambda item: (-item[1], item[0].start),
    )
    for section, score in ranked_all:
        selected.setdefault(section.section_id, (section, score, "query"))
        if len(selected) >= max_sections:
            break

    output: list[dict[str, Any]] = []
    remaining = max_chars
    ordered = sorted(selected.values(), key=lambda item: (-item[1], item[0].start))
    for section, score, reason in ordered[:max_sections]:
        text = section.text
        if len(text) > remaining:
            text = text[:remaining]
        if not text:
            continue
        output.append(
            {
                "evidence_id": section.section_id,
                "doc_id": section.doc_id,
                "heading": section.heading,
                "anchor": section.anchor,
                "selection_reason": reason,
                "selection_score": round(score, 6),
                "text": text,
            }
        )
        remaining -= len(text)
        if remaining <= 0:
            break
    return output, mappings


def _normalized_route(value: str) -> str:
    path = unquote(urlsplit(value).path or "/").casefold()
    parts = [part for part in re.sub(r"/+", "/", path).split("/") if part]
    if parts and re.fullmatch(r"[a-z]{2}(?:-[a-z]{2})?", parts[0]):
        parts.pop(0)
    if parts and re.fullmatch(
        r"(?:enterprise-(?:cloud|server)|free-pro-team)@[^/]+",
        parts[0],
    ):
        parts.pop(0)
    return "/" + "/".join(parts).rstrip("/")


def _local_frontmatter_redirects(document: dict[str, Any]) -> list[str]:
    """Read redirect aliases from the pinned local Markdown when needed.

    New corpus builds persist ``redirects`` directly.  This fallback keeps old
    frozen packages verifiable without consulting a live documentation site.
    """

    persisted = [str(value) for value in document.get("redirects") or []]
    if persisted:
        return persisted
    path = DATASET_ROOT / local_doc_path(document)
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.startswith("---\n"):
        return []
    end = text.find("\n---", 4)
    if end < 0:
        return []
    frontmatter = text[4:end]
    match = re.search(
        r"(?ms)^redirect_from:\s*\n(?P<items>(?:\s+-\s+[^\n]+\n?)+)",
        frontmatter,
    )
    if not match:
        return []
    return [
        value.strip().strip("'\"")
        for value in re.findall(r"(?m)^\s+-\s+(.+?)\s*$", match.group("items"))
    ]


def _redirect_map(project: str, cache_root: Path) -> dict[str, str]:
    name = PROJECT_REDIRECT_NAMES.get(project)
    if not name:
        return {}
    path = cache_root / "discussions" / name / "redirects.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in payload.items()}


def resolve_urls_to_qrels(
    question: dict[str, Any],
    corpus: dict[str, dict[str, Any]],
    cache_root: Path,
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    """Resolve source URLs to qrels only through inspectable local evidence.

    Dataset builders may provide an exact ``url_resolutions`` record.  Older
    frozen packages retain ``resolution_kinds``; for those, a single qrel is
    accepted only when the builder recorded a verified canonical or redirect
    resolution.  Merely having one qrel is never proof that an arbitrary URL
    maps to it.
    """

    qrels = [str(value) for value in question.get("qrel_ids") or []]
    output: dict[str, list[str]] = {doc_id: [] for doc_id in qrels}
    for doc_id, anchors in (question.get("qrel_anchors") or {}).items():
        if str(doc_id) in output:
            output[str(doc_id)].extend(_slug(str(value)) for value in anchors if value)

    project = str(question.get("project") or question.get("dataset") or "")
    redirects = _redirect_map(project, cache_root)
    by_source_id = {
        _normalized_route(str(corpus[doc_id].get("source_doc_id") or "")): doc_id
        for doc_id in qrels
        if corpus[doc_id].get("source_doc_id")
    }
    by_redirect: dict[str, str] = {}
    for doc_id in qrels:
        for redirect in _local_frontmatter_redirects(corpus[doc_id]):
            by_redirect[_normalized_route(str(redirect))] = doc_id

    explicit = question.get("url_resolutions") or {}
    recorded_kinds = question.get("resolution_kinds") or {}
    verified_kinds = {
        "canonical",
        "canonical_path",
        "frontmatter_redirect",
        "redirect",
        "redirect_cache",
        "exact_local_route",
        "live_http_redirect",
        "unique_slug",
    }
    records: list[dict[str, Any]] = []
    urls = question.get("docs_urls") or question.get("reference_answer_links") or []
    for url in urls:
        raw = str(url)
        normalized_route = _normalized_route(raw)
        fragment = _slug(unquote(urlsplit(raw).fragment))
        route_intent = _slug(normalized_route.rsplit("/", 1)[-1])
        target: str | None = None
        method = "unresolved"
        resolution_kind = str(recorded_kinds.get(raw) or "")

        explicit_row = explicit.get(raw) or {}
        explicit_doc_id = str(explicit_row.get("doc_id") or "")
        if explicit_doc_id in output:
            target = explicit_doc_id
            resolution_kind = str(explicit_row.get("resolution_kind") or resolution_kind)
            explicit_anchor = _slug(str(explicit_row.get("anchor") or ""))
            if explicit_anchor:
                fragment = explicit_anchor
            method = "dataset_url_resolution"
        if target is None and normalized_route in by_redirect:
            target = by_redirect[normalized_route]
            resolution_kind = resolution_kind or "frontmatter_redirect"
            method = "corpus_frontmatter_redirect"
        if target is None:
            redirected = redirects.get(raw)
            if redirected:
                target = by_source_id.get(_normalized_route(redirected))
                if target:
                    resolution_kind = resolution_kind or "redirect_cache"
                    method = "frozen_redirect_cache"
        if target is None:
            target = by_source_id.get(normalized_route)
            if target:
                resolution_kind = resolution_kind or "canonical"
                method = "exact_local_route"
        if target is None and resolution_kind == "unique_slug":
            slug = normalized_route.rsplit("/", 1)[-1]
            matches = [
                doc_id
                for source_route, doc_id in by_source_id.items()
                if source_route.rsplit("/", 1)[-1] == slug
            ]
            if len(matches) == 1:
                target = matches[0]
                method = "verified_unique_slug"
        if (
            target is None
            and len(qrels) == 1
            and resolution_kind in verified_kinds
        ):
            target = qrels[0]
            method = "verified_builder_single_qrel"

        selection_anchor = fragment
        if target and not selection_anchor and resolution_kind in {
            "frontmatter_redirect",
            "live_http_redirect",
            "unique_slug",
        }:
            # Preserve the semantic intent of a historical page slug even when
            # it had no fragment.  This helps constrain a broader successor page.
            selection_anchor = route_intent
        if target and selection_anchor:
            output[target].append(selection_anchor)
        records.append(
            {
                "original_url": raw,
                "normalized_route": normalized_route,
                "original_fragment": fragment,
                "route_intent": route_intent,
                "local_doc_id": target,
                "resolution_kind": resolution_kind or "unresolved",
                "mapping_method": method,
                "verified": bool(target and method != "unresolved"),
            }
        )
    return (
        {doc_id: list(dict.fromkeys(values)) for doc_id, values in output.items()},
        records,
    )


def map_urls_to_qrels(
    question: dict[str, Any],
    corpus: dict[str, dict[str, Any]],
    cache_root: Path,
) -> dict[str, list[str]]:
    """Backward-compatible fragment map backed by verified URL resolution."""

    return resolve_urls_to_qrels(question, corpus, cache_root)[0]


def local_doc_path(document: dict[str, Any]) -> str:
    return "docs/" + str(document.get("source_path") or "").lstrip("/")


def build_evidence_package(
    question: dict[str, Any],
    corpus: dict[str, dict[str, Any]],
    image_row: dict[str, Any],
    image_text_by_sha: dict[str, dict[str, Any]],
    cache_root: Path,
    resolution_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fragments, url_resolution = resolve_urls_to_qrels(question, corpus, cache_root)
    resolution_index = resolution_index or build_local_resolution_index(corpus)
    recovered_doc_ids = recover_historical_targets(
        question,
        corpus,
        fragments,
        url_resolution,
        resolution_index,
    )
    candidate_doc_ids = list(
        dict.fromkeys(
            [str(value) for value in question.get("qrel_ids") or []] + recovered_doc_ids
        )
    )
    evidence_doc_ids, routing_expansions = expand_empty_routing_pages(
        question,
        candidate_doc_ids,
        fragments,
        resolution_index,
    )
    documents: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = [
        {
            "evidence_id": "accepted_answer",
            "kind": "accepted_answer",
            "text": str(question.get("reference_answer") or "").strip(),
        },
        {
            "evidence_id": "question_context",
            "kind": "question_context",
            "text": str(question.get("query") or "").strip(),
            "constraint": "May support only facts about the user's supplied context or scaffold.",
        },
    ]
    anchor_map: list[dict[str, Any]] = []
    total_qrels = max(1, len(evidence_doc_ids))
    for raw_doc_id in evidence_doc_ids:
        doc_id = str(raw_doc_id)
        document = corpus[doc_id]
        path = local_doc_path(document)
        selected, mappings = select_sections(
            document,
            fragments.get(doc_id) or [],
            str(question.get("query") or ""),
            str(question.get("reference_answer") or ""),
            max_chars=max(5_000, 24_000 // total_qrels),
        )
        for mapping in mappings:
            anchor_map.append({"doc_id": doc_id, "local_path": path, **mapping})
            if mapping["original_fragment"]:
                evidence.append(
                    {
                        "evidence_id": f"historical-link:{doc_id}#{mapping['original_fragment']}",
                        "kind": "historical_link_context",
                        "doc_id": doc_id,
                        "local_path": path,
                        "heading": mapping["canonical_heading"],
                        "text": (
                            "The accepted answer explicitly linked the historical documentation fragment "
                            f"'{mapping['original_fragment']}', resolved locally to the heading "
                            f"'{mapping['canonical_heading']}'."
                        ),
                        "constraint": (
                            "This establishes the answer author's intended documentation topic and location, "
                            "not a product fact that is absent from the local section."
                        ),
                    }
                )
        document_record = {
            "doc_id": doc_id,
            "title": document.get("title") or document.get("short_title") or "",
            "local_path": path,
            "fragments": fragments.get(doc_id) or [],
            "sections": selected,
        }
        documents.append(document_record)
        for section in selected:
            evidence.append(
                {
                    "evidence_id": section["evidence_id"],
                    "kind": "local_document_section",
                    "doc_id": doc_id,
                    "local_path": path,
                    "heading": section["heading"],
                    "text": section["text"],
                }
            )

    images: list[dict[str, Any]] = []
    for image in image_row.get("images") or []:
        if image.get("status") == "text_fallback":
            record = {
                "evidence_id": str(image["image_id"]),
                "kind": "pinned_document_alt_text",
                "role": image.get("role"),
                "doc_id": image.get("doc_id"),
                "alt": image.get("alt") or "",
                "text": str(image.get("fallback_text") or image.get("alt") or "").strip(),
                "pixel_verified": False,
                "constraint": (
                    "Author-written alt text from the pinned local document. It may support only its explicit text; "
                    "do not infer additional visual details."
                ),
            }
            images.append(record)
            evidence.append(record)
            continue
        if image.get("status") != "resolved":
            continue
        artifact = image_text_by_sha.get(str(image.get("sha256") or ""))
        if not artifact:
            continue
        text = "\n".join(
            value
            for value in (
                str(artifact.get("literal_text") or "").strip(),
                str(artifact.get("factual_description") or "").strip(),
            )
            if value
        )
        record = {
            "evidence_id": str(image["image_id"]),
            "kind": "image_derived_text",
            "role": image.get("role"),
            "doc_id": image.get("doc_id"),
            "sha256": image.get("sha256"),
            "alt": image.get("alt") or "",
            "text": text,
            "unreadable_or_ambiguous": bool(artifact.get("unreadable_or_ambiguous")),
        }
        images.append(record)
        evidence.append(record)
    return {
        "question_id": question["question_id"],
        "question": str(question.get("query") or ""),
        "accepted_answer": str(question.get("reference_answer") or ""),
        "documents": documents,
        "original_qrel_ids": [str(value) for value in question.get("qrel_ids") or []],
        "recovered_doc_ids": recovered_doc_ids,
        "routing_expansions": routing_expansions,
        "evidence_doc_ids": evidence_doc_ids,
        "images": images,
        "evidence": evidence,
        "anchor_map": anchor_map,
        "url_resolution": url_resolution,
        "ambiguous_anchor_count": sum(bool(row["ambiguous"]) for row in anchor_map),
    }


def _call_parse(
    client: Any,
    model: str,
    instructions: str,
    content: list[dict[str, Any]],
    text_format: type[BaseModel],
    reasoning_effort: str,
    attempts: int,
) -> tuple[BaseModel, dict[str, int], float]:
    last_error: Exception | None = None
    for attempt in range(attempts):
        started = time.perf_counter()
        try:
            response = client.responses.parse(
                model=model,
                instructions=instructions,
                input=[{"role": "user", "content": content}],
                text_format=text_format,
                reasoning={"effort": reasoning_effort},
                store=False,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise ValueError("model returned no structured output")
            return parsed, response_usage(response), time.perf_counter() - started
        except Exception as error:
            last_error = error
            if "credit_balance_exhausted" in str(error) or "insufficient_quota" in str(error):
                break
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def derive_user_requirements(
    client: Any,
    model: str,
    question: str,
    reasoning_effort: str,
    attempts: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a fixed, method-neutral requirement plan from the question only."""

    payload = {
        "rubric_version": RUBRIC_VERSION,
        "rubric_sha256": RUBRIC_SHA256,
        "question": question,
        "rules": RUBRIC_CONFIG["requirement_rules"],
    }
    parsed, usage, latency = _call_parse(
        client,
        model,
        str(RUBRIC_CONFIG["requirement_system"]),
        [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}],
        RequirementPlan,
        reasoning_effort,
        attempts,
    )
    plan = parsed.model_dump()
    return plan, {
        "usage": usage,
        "latency_seconds": round(latency, 6),
        "prompt_sha256": hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
    }


def transcribe_image(
    client: Any,
    model: str,
    image: dict[str, Any],
    reasoning_effort: str,
    attempts: int,
) -> dict[str, Any]:
    prompt = {
        "task": "Create a literal text artifact for local document retrieval.",
        "rules": [
            "Transcribe all legible text, code, labels, values, and error messages.",
            "Describe only visible structure and state; do not infer a solution or use outside knowledge.",
            "Keep exact product names, commands, identifiers, and ordering when visible.",
            "Mark unreadable regions explicitly rather than guessing.",
        ],
        "image": {
            "sha256": image["sha256"],
            "mime_type": image["mime_type"],
            "width": image["width"],
            "height": image["height"],
            "alt": image.get("alt") or "",
            "title": image.get("title") or "",
        },
    }
    parsed, usage, latency = _call_parse(
        client,
        model,
        "You are a conservative OCR and document-image transcriber. Never answer the underlying question.",
        [
            {"type": "input_text", "text": json.dumps(prompt, ensure_ascii=False)},
            {"type": "input_image", "image_url": image_data_url(image), "detail": "high"},
        ],
        ImageTextArtifact,
        reasoning_effort,
        attempts,
    )
    return {
        "sha256": image["sha256"],
        "mime_type": image["mime_type"],
        "width": image["width"],
        "height": image["height"],
        **parsed.model_dump(),
        "model": model,
        "artifact_version": IMAGE_TEXT_VERSION,
        "usage": usage,
        "latency_seconds": round(latency, 6),
    }


def _evidence_for_prompt(package: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            key: value
            for key, value in row.items()
            if key in {"evidence_id", "kind", "doc_id", "local_path", "heading", "role", "text", "constraint"}
        }
        for row in package["evidence"]
    ]


def canonicalize_draft_evidence_ids(
    draft: dict[str, Any], valid_evidence_ids: set[str]
) -> list[dict[str, Any]]:
    """Repair only unambiguous near-copies of supplied local section IDs.

    Models occasionally rewrite a route segment while copying a long local
    evidence ID.  We accept a correction only when the heading anchor is exact,
    the document-route similarity is high, and no second candidate is close.
    This never resolves a made-up heading or an arbitrary citation.
    """

    cited = {
        str(value)
        for claim in draft.get("claims") or []
        for value in claim.get("evidence_ids") or []
    } | {str(row.get("evidence_id")) for row in draft.get("citations") or []}
    corrections: dict[str, str] = {}
    for evidence_id in cited - valid_evidence_ids:
        if "#" not in evidence_id:
            continue
        document_id, anchor = evidence_id.rsplit("#", 1)
        candidates = [
            candidate
            for candidate in valid_evidence_ids
            if "#" in candidate and candidate.rsplit("#", 1)[1] == anchor
        ]
        ranked = sorted(
            (
                difflib.SequenceMatcher(
                    None,
                    document_id.casefold(),
                    candidate.rsplit("#", 1)[0].casefold(),
                ).ratio(),
                candidate,
            )
            for candidate in candidates
        )
        ranked.reverse()
        if ranked and ranked[0][0] >= 0.88:
            second = ranked[1][0] if len(ranked) > 1 else 0.0
            if ranked[0][0] - second >= 0.06:
                corrections[evidence_id] = ranked[0][1]
                continue
        same_document = [
            candidate
            for candidate in valid_evidence_ids
            if "#" in candidate and candidate.rsplit("#", 1)[0] == document_id
        ]
        ranked_anchors = sorted(
            (
                difflib.SequenceMatcher(
                    None,
                    anchor.casefold(),
                    candidate.rsplit("#", 1)[1].casefold(),
                ).ratio(),
                candidate,
            )
            for candidate in same_document
        )
        ranked_anchors.reverse()
        if not ranked_anchors or ranked_anchors[0][0] < 0.86:
            continue
        second_anchor = ranked_anchors[1][0] if len(ranked_anchors) > 1 else 0.0
        if ranked_anchors[0][0] - second_anchor < 0.12:
            continue
        corrections[evidence_id] = ranked_anchors[0][1]
    if not corrections:
        return []
    for claim in draft.get("claims") or []:
        claim["evidence_ids"] = [
            corrections.get(str(value), str(value)) for value in claim.get("evidence_ids") or []
        ]
    for citation in draft.get("citations") or []:
        original = str(citation.get("evidence_id"))
        citation["evidence_id"] = corrections.get(original, original)
    rows = [
        {
            "original_evidence_id": original,
            "canonical_evidence_id": canonical,
            "method": (
                "exact_anchor_unique_near_route"
                if original.rsplit("#", 1)[1] == canonical.rsplit("#", 1)[1]
                else "exact_document_unique_near_anchor"
            ),
        }
        for original, canonical in sorted(corrections.items())
    ]
    draft["evidence_id_corrections"] = rows
    return rows


def deduplicate_exact_grade_requirements(grade: AnswerGrade) -> int:
    """Remove exact duplicate grader rows without repairing altered requirements."""

    unique: list[RequirementGrade] = []
    seen: set[tuple[str, str, bool]] = set()
    removed = 0
    for row in grade.requirements:
        key = (str(row.requirement_id), str(row.requirement), bool(row.critical))
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        unique.append(row)
    if removed:
        grade.requirements = unique
    return removed


def generate_answer(
    client: Any,
    model: str,
    package: dict[str, Any],
    reasoning_effort: str,
    attempts: int,
    previous: dict[str, Any] | None = None,
    feedback: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = {
        "normalizer_version": NORMALIZER_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "rubric_sha256": RUBRIC_SHA256,
        "question": package["question"],
        "user_requirements": package["user_requirements"],
        "evidence": _evidence_for_prompt(package),
        "rules": RUBRIC_CONFIG["generation_rules"],
        "previous_draft": previous,
        "grader_feedback": feedback or [],
    }
    parsed, usage, latency = _call_parse(
        client,
        model,
        str(RUBRIC_CONFIG["generation_system"]),
        [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}],
        NormalizedAnswerDraft,
        reasoning_effort,
        attempts,
    )
    draft = parsed.model_dump()
    return draft, {
        "usage": usage,
        "latency_seconds": round(latency, 6),
        "prompt_sha256": hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
    }


def _support_value(value: str) -> float:
    return {"full": 1.0, "partial": 0.5, "none": 0.0, "unsupported": 0.0, "contradicted": 0.0}[value]


def compute_grade(
    grade: AnswerGrade,
    draft: dict[str, Any],
    valid_evidence_ids: set[str],
    expected_requirements: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    critical = [row for row in grade.requirements if row.critical] or list(grade.requirements)
    coverage = statistics.fmean(_support_value(row.coverage) for row in critical)
    grounding = statistics.fmean(_support_value(row.support) for row in grade.claims)
    cited = {
        str(value)
        for claim in draft.get("claims") or []
        for value in claim.get("evidence_ids") or []
    } | {str(row.get("evidence_id")) for row in draft.get("citations") or []}
    provenance = bool(cited) and cited <= valid_evidence_ids and grade.citation_integrity
    score = (
        0.35 * coverage
        + 0.25 * (grade.correctness / 4.0)
        + 0.20 * grounding
        + 0.10 * (grade.actionability / 4.0)
        + 0.05 * (grade.directness / 4.0)
        + 0.05 * float(provenance)
    )
    contradicted = any(row.coverage == "contradicted" for row in critical) or any(
        row.support == "contradicted" for row in grade.claims
    )
    critical_full = all(row.coverage == "full" for row in critical)
    claims_full = all(row.support == "full" for row in grade.claims)
    # Partial entailment is already penalized in the weighted score.  Keep a
    # hard veto only for unsupported or contradicted claims; otherwise the old
    # all-perfect binary gate could reject a 0.98 answer whose critical user
    # requirements were fully covered because of one ancillary qualification.
    claims_safe = all(row.support in {"full", "partial"} for row in grade.claims)
    expected = [
        (str(row["requirement_id"]), str(row["requirement"]), bool(row["critical"]))
        for row in (expected_requirements or [])
    ]
    actual = [
        (str(row.requirement_id), str(row.requirement), bool(row.critical))
        for row in grade.requirements
    ]
    requirements_match = not expected or actual == expected
    return {
        "score": round(score, 6),
        "critical_requirement_coverage": round(coverage, 6),
        "claim_grounding": round(grounding, 6),
        "provenance_valid": provenance,
        "critical_requirements_full": critical_full,
        "claims_fully_supported": claims_full,
        "claims_safely_supported": claims_safe,
        "requirements_match": requirements_match,
        "contradiction": contradicted,
    }


def grade_answer(
    client: Any,
    model: str,
    package: dict[str, Any],
    draft: dict[str, Any],
    reasoning_effort: str,
    attempts: int,
    variant: Literal["coverage", "falsification"],
) -> tuple[dict[str, Any], dict[str, Any]]:
    common_rules = RUBRIC_CONFIG["grading_rules"]
    variant_rule = RUBRIC_CONFIG["grading_variants"][variant]
    payload = {
        "rubric_version": RUBRIC_VERSION,
        "rubric_sha256": RUBRIC_SHA256,
        "variant": variant,
        "question": package["question"],
        "user_requirements": package["user_requirements"],
        "draft": draft,
        "local_evidence": _evidence_for_prompt(package),
        "rules": common_rules,
        "variant_rule": variant_rule,
    }
    parsed, usage, latency = _call_parse(
        client,
        model,
        str(RUBRIC_CONFIG["grading_system"]),
        [{"type": "input_text", "text": json.dumps(payload, ensure_ascii=False)}],
        AnswerGrade,
        reasoning_effort,
        attempts,
    )
    duplicate_requirements_removed = deduplicate_exact_grade_requirements(parsed)
    grade = parsed.model_dump()
    metrics = compute_grade(
        parsed,
        draft,
        {str(row["evidence_id"]) for row in package["evidence"]},
        package["user_requirements"]["requirements"],
    )
    return {**grade, **metrics}, {
        "usage": usage,
        "latency_seconds": round(latency, 6),
        "prompt_sha256": hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        "variant": variant,
        "duplicate_requirements_removed": duplicate_requirements_removed,
    }


def passes(grade: dict[str, Any], threshold: float) -> bool:
    return bool(
        float(grade["score"]) > threshold
        and grade["outcome"] == "solves"
        and grade["critical_requirements_full"]
        and grade.get("claims_safely_supported", grade["claims_fully_supported"])
        and grade.get("requirements_match", True)
        and grade["provenance_valid"]
        and not grade["contradiction"]
    )


def _failure_base(question: dict[str, Any], reason: str, details: Any) -> dict[str, Any]:
    return {
        "question_id": question["question_id"],
        "project": question.get("project") or question.get("dataset"),
        "split": question.get("split"),
        "status": "rejected",
        "rejection_reason": reason,
        "details": details,
    }


def normalize_one(
    client: Any,
    model: str,
    question: dict[str, Any],
    corpus: dict[str, dict[str, Any]],
    image_row: dict[str, Any],
    image_text_by_sha: dict[str, dict[str, Any]],
    cache_root: Path,
    threshold: float,
    max_repairs: int,
    reasoning_effort: str,
    attempts: int,
    resolution_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failed_images = [row for row in image_row.get("images") or [] if row.get("status") == "failed"]
    if failed_images:
        return _failure_base(
            question,
            "unreproducible_image_evidence",
            [{key: image.get(key) for key in ("image_id", "role", "doc_id", "error_code")} for image in failed_images],
        )
    missing_transcripts = [
        str(row.get("image_id"))
        for row in image_row.get("images") or []
        if row.get("status") == "resolved" and str(row.get("sha256")) not in image_text_by_sha
    ]
    if missing_transcripts:
        return _failure_base(question, "missing_image_text_artifact", missing_transcripts)
    missing_qrels = [str(value) for value in question.get("qrel_ids") or [] if str(value) not in corpus]
    if missing_qrels:
        return _failure_base(question, "missing_local_qrel", missing_qrels)
    package = build_evidence_package(
        question,
        corpus,
        image_row,
        image_text_by_sha,
        cache_root,
        resolution_index,
    )
    unresolved_urls = [
        row for row in package["url_resolution"] if not row.get("verified")
    ]
    if unresolved_urls:
        return _failure_base(question, "unverified_url_to_local_mapping", unresolved_urls)
    local_paths = [doc["local_path"] for doc in package["documents"]]
    absent_paths = [path for path in local_paths if not (DATASET_ROOT / path).is_file()]
    if absent_paths:
        return _failure_base(question, "missing_local_document_file", absent_paths)
    if any(not document["sections"] for document in package["documents"]):
        return _failure_base(question, "empty_local_document_evidence", [doc["doc_id"] for doc in package["documents"] if not doc["sections"]])

    requirement_plan, requirement_meta = derive_user_requirements(
        client,
        model,
        str(question.get("query") or ""),
        reasoning_effort,
        attempts,
    )
    package["user_requirements"] = requirement_plan
    package["requirement_derivation"] = requirement_meta

    rounds: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    feedback: list[str] = []
    accepted_draft: dict[str, Any] | None = None
    accepted_grade: dict[str, Any] | None = None
    accepted_final_grade: dict[str, Any] | None = None
    accepted_final_meta: dict[str, Any] | None = None
    for round_index in range(max_repairs + 1):
        draft, generation_meta = generate_answer(
            client, model, package, reasoning_effort, attempts, previous, feedback
        )
        citation_corrections = canonicalize_draft_evidence_ids(
            draft,
            {str(row["evidence_id"]) for row in package["evidence"]},
        )
        generation_meta["evidence_id_corrections"] = citation_corrections
        grade, grading_meta = grade_answer(
            client, model, package, draft, reasoning_effort, attempts, "coverage"
        )
        rounds.append(
            {
                "round": round_index,
                "draft": draft,
                "normalizer_marked_insufficient": draft["insufficient_evidence"],
                "development_grade": grade,
                "generation": generation_meta,
                "grading": grading_meta,
            }
        )
        if passes(grade, threshold):
            final_grade, final_meta = grade_answer(
                client, model, package, draft, reasoning_effort, attempts, "falsification"
            )
            rounds[-1]["final_grade"] = final_grade
            rounds[-1]["final_grading"] = final_meta
            conservative_score = min(float(grade["score"]), float(final_grade["score"]))
            if passes(final_grade, threshold) and conservative_score > threshold:
                accepted_draft = draft
                accepted_grade = grade
                accepted_final_grade = final_grade
                accepted_final_meta = final_meta
                break
            feedback = list(final_grade.get("repair_instructions") or []) + list(
                final_grade.get("unsupported_or_invented_information") or []
            )
            feedback.append(
                "The independent falsification grade did not pass. Remove or qualify the specific unsupported "
                "claim it identified while still answering the user's underlying problem."
            )
        else:
            feedback = list(grade.get("repair_instructions") or []) + list(
                grade.get("unsupported_or_invented_information") or []
            )
        previous = draft
        if draft["insufficient_evidence"]:
            feedback.append(
                "If the evidence supports a truthful negative, qualified boundary, or operator-in-scaffold answer, "
                "return that complete answer and clear insufficient_evidence."
            )
    if (
        accepted_draft is None
        or accepted_grade is None
        or accepted_final_grade is None
        or accepted_final_meta is None
    ):
        final = rounds[-1].get("development_grade") or {}
        independent = rounds[-1].get("final_grade") or {}
        return {
            **_failure_base(
                question,
                "normalization_threshold_not_met",
                {
                    "development_score": final.get("score"),
                    "independent_score": independent.get("score"),
                    "feedback": (independent or final).get("repair_instructions"),
                },
            ),
            "evidence_package": package,
            "rounds": rounds,
        }
    final_grade = accepted_final_grade
    final_meta = accepted_final_meta
    conservative_score = min(float(accepted_grade["score"]), float(final_grade["score"]))

    used_evidence = {
        str(value)
        for claim in accepted_draft["claims"]
        for value in claim.get("evidence_ids") or []
    }
    used_documents = [
        doc for doc in package["documents"]
        if any(section["evidence_id"] in used_evidence for section in doc["sections"])
    ] or package["documents"]
    citations = [
        {
            "doc_id": doc["doc_id"],
            "local_path": doc["local_path"],
            "headings": [
                section["heading"] for section in doc["sections"] if section["evidence_id"] in used_evidence
            ] or [section["heading"] for section in doc["sections"][:1]],
        }
        for doc in used_documents
    ]
    used_image_rows = [
        row for row in package["images"] if row["evidence_id"] in used_evidence
    ]
    image_block = ""
    if used_image_rows:
        image_block = "\n\nImage-derived text evidence:\n" + "\n".join(
            f"- [{row['evidence_id']}] {row['text']}" for row in used_image_rows
        )
    source_block = "\n\nLocal documentation:\n" + "\n".join(
        f"- {row['local_path']}" + (f" — {', '.join(row['headings'])}" if row["headings"] else "")
        for row in citations
    )
    normalized = {
        **{key: value for key, value in question.items() if key not in {"reference_answer"}},
        "original_reference_answer": question.get("reference_answer"),
        "original_qrel_ids": package["original_qrel_ids"],
        "qrel_ids": package["evidence_doc_ids"],
        "qrel_count": len(package["evidence_doc_ids"]),
        "normalized_answer": accepted_draft["answer_text"].strip(),
        "normalized_answer_with_local_sources": (
            accepted_draft["answer_text"].strip() + image_block + source_block
        ),
        "local_document_citations": citations,
        "image_text_evidence": package["images"],
        "image_text_evidence_used": [row["evidence_id"] for row in used_image_rows],
        "evidence_mode": accepted_draft["evidence_mode"],
        "normalized_claims": accepted_draft["claims"],
        "user_requirements": requirement_plan,
        "url_resolution": package["url_resolution"],
        "anchor_resolution": package["anchor_map"],
        "normalization_status": "accepted",
        "answerability_score": round(conservative_score, 6),
        "development_score": accepted_grade["score"],
        "independent_final_score": final_grade["score"],
        "grader": {
            "model": model,
            "rubric_version": RUBRIC_VERSION,
            "threshold": threshold,
            "development_grade": accepted_grade,
            "final_grade": final_grade,
            "final_grading": final_meta,
        },
        "normalization_rounds": len(rounds),
        "retrieval_modalities": ["text", "image_derived_text"] if package["images"] else ["text"],
    }
    return {
        "question_id": question["question_id"],
        "project": question.get("project") or question.get("dataset"),
        "split": question.get("split"),
        "status": "accepted",
        "normalized_question": normalized,
        "evidence_package": package,
        "rounds": rounds,
    }


def _load_completed(path: Path) -> dict[str, dict[str, Any]]:
    """Load every verdict after the caller verifies the frozen run contract."""

    return {str(row["question_id"]): row for row in _load_jsonl(path)}


def _run_contract(args: argparse.Namespace) -> dict[str, Any]:
    contract = {
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "normalizer_version": NORMALIZER_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "rubric_sha256": RUBRIC_SHA256,
        "rubric_path": str(RUBRIC_PATH.relative_to(PROJECT_ROOT)),
        "image_text_version": IMAGE_TEXT_VERSION,
        "threshold": args.threshold,
        "max_repairs": args.max_repairs,
        "source_corpus_sha256": _sha256(args.dataset / "corpus.jsonl"),
        "source_questions_sha256": _sha256(args.dataset / "questions.jsonl"),
    }
    if args.image_text_cache and args.image_text_cache.is_file():
        contract["seed_image_text_sha256"] = _sha256(args.image_text_cache)
    return contract


def _verify_or_create_run_contract(args: argparse.Namespace) -> dict[str, Any]:
    contract = _run_contract(args)
    path = args.work_dir / "run_contract.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError(
                "--resume work directory uses a different dataset/model/rubric contract; "
                "choose a new work directory"
            )
    else:
        path.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return contract


def transcribe_all_images(
    args: argparse.Namespace,
    client: Any,
    question_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    image_rows = _load_jsonl(args.image_evidence / "per_image.jsonl")
    if question_ids is not None:
        image_rows = [row for row in image_rows if str(row.get("question_id")) in question_ids]
    resolved: dict[str, dict[str, Any]] = {}
    for row in image_rows:
        if row.get("status") == "resolved":
            resolved.setdefault(str(row["sha256"]), row)
    path = args.work_dir / "image_text.jsonl"
    completed = {
        str(row["sha256"]): row
        for row in _load_jsonl(path)
        if row.get("model") == args.model and row.get("artifact_version") == IMAGE_TEXT_VERSION
    }
    if args.image_text_cache and args.image_text_cache.is_file():
        for row in _load_jsonl(args.image_text_cache):
            if (
                row.get("model") == args.model
                and row.get("artifact_version") == IMAGE_TEXT_VERSION
            ):
                reused = dict(row)
                reused["reused_from_cache"] = True
                reused["source_usage"] = reused.pop("usage", None)
                reused["source_latency_seconds"] = reused.get("latency_seconds")
                reused["latency_seconds"] = 0.0
                completed.setdefault(str(row["sha256"]), reused)
    pending = [row for sha, row in resolved.items() if sha not in completed]
    lock = threading.Lock()
    failures: list[dict[str, str]] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                transcribe_image,
                client,
                args.model,
                row,
                args.reasoning_effort,
                args.attempts,
            ): str(row["sha256"])
            for row in pending
        }
        with path.open("a", encoding="utf-8") as handle:
            for future in as_completed(futures):
                sha = futures[future]
                try:
                    artifact = future.result()
                    completed[sha] = artifact
                    with lock:
                        handle.write(json.dumps(artifact, ensure_ascii=False, sort_keys=True) + "\n")
                        handle.flush()
                        if len(completed) == len(resolved) or len(completed) % 25 == 0:
                            print(f"image text progress: {len(completed)}/{len(resolved)}", flush=True)
                except Exception as error:
                    failures.append({"sha256": sha, "error": f"{type(error).__name__}: {error}"})
    _write_jsonl(path, [completed[key] for key in sorted(completed)])
    if failures:
        _write_jsonl(args.work_dir / "image_text_failures.jsonl", failures)
    return completed


def _enrich_corpus(
    corpus: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    image_texts: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    used_qrels = {
        str(doc_id)
        for row in accepted
        for doc_id in row["normalized_question"].get("qrel_ids") or []
    }
    for row in accepted:
        for image in row["normalized_question"].get("image_text_evidence") or []:
            doc_id = image.get("doc_id")
            if doc_id:
                image_key = str(image.get("sha256") or image.get("evidence_id") or "")
                if image_key:
                    image_texts[str(doc_id)][image_key] = image
    substantive_ids = {
        str(row["doc_id"])
        for row in corpus
        if str(row.get("rendered_text") or row.get("raw_text") or "").strip()
    } | used_qrels
    output: list[dict[str, Any]] = []
    for source in corpus:
        doc_id = str(source["doc_id"])
        text = str(source.get("rendered_text") or source.get("raw_text") or "").strip()
        if doc_id not in substantive_ids or not text:
            continue
        images = list(image_texts.get(doc_id, {}).values())
        image_appendix = ""
        if images:
            image_appendix = "\n\n## Image-derived text evidence\n\n" + "\n\n".join(
                f"[Image {image.get('sha256') or image.get('evidence_id')}]\n{image['text']}"
                for image in images
            )
        row = dict(source)
        row.update(
            {
                "rendered_text": text + image_appendix,
                "local_path": local_doc_path(source),
                "image_texts": images,
                "retrieval_modalities": ["text", "image_derived_text"] if images else ["text"],
                "image_vector_enabled": False,
                "outgoing_ids": [str(value) for value in source.get("outgoing_ids") or [] if str(value) in substantive_ids],
                "outgoing_paths": [str(value) for value in source.get("outgoing_paths") or [] if str(value) in substantive_ids],
                "link_edges": [edge for edge in source.get("link_edges") or [] if str(edge.get("target_id")) in substantive_ids],
            }
        )
        output.append(row)
    return sorted(output, key=lambda row: str(row["doc_id"]))


def _upgrade_normalized_question(row: dict[str, Any]) -> None:
    """Apply deterministic output-shape upgrades to resumed v4 verdicts."""

    question = row["normalized_question"]
    package = row.get("evidence_package") or {}
    question["normalized_answer"] = _sanitize_normalized_answer(
        str(question["normalized_answer"])
    )
    all_images = list(package.get("images") or [])
    question_images = [
        image
        for image in all_images
        if image.get("role") == "question" and str(image.get("text") or "").strip()
    ]
    base_query = str(package.get("question") or question.get("query") or "").strip()
    question_image_block = ""
    if question_images:
        question_image_block = "\n\nImage-derived question evidence:\n" + "\n".join(
            f"- [{image['evidence_id']}] "
            f"{_sanitize_normalized_answer(str(image['text']))}"
            for image in question_images
        )
    question["query"] = base_query + question_image_block
    question["question_image_text_evidence_used"] = [
        str(image["evidence_id"]) for image in question_images
    ]
    question["question_modalities"] = (
        ["text", "image_derived_text"] if question_images else ["text"]
    )
    claim_evidence = {
        str(evidence_id)
        for claim in question.get("normalized_claims") or []
        for evidence_id in claim.get("evidence_ids") or []
    }
    used_ids = [
        str(image["evidence_id"])
        for image in all_images
        if str(image["evidence_id"]) in claim_evidence
    ]
    used = [image for image in all_images if str(image["evidence_id"]) in set(used_ids)]
    image_block = ""
    if used:
        image_block = "\n\nImage-derived text evidence:\n" + "\n".join(
            f"- [{image['evidence_id']}] {_sanitize_normalized_answer(str(image['text']))}"
            for image in used
        )
    citations = list(question.get("local_document_citations") or [])
    source_block = "\n\nLocal documentation:\n" + "\n".join(
        f"- {citation['local_path']}"
        + (
            f" — {', '.join(citation.get('headings') or [])}"
            if citation.get("headings")
            else ""
        )
        for citation in citations
    )
    question["image_text_evidence"] = all_images
    question["image_text_evidence_used"] = used_ids
    question["normalized_answer_with_local_sources"] = (
        str(question["normalized_answer"]).strip() + image_block + source_block
    )
    question["retrieval_modalities"] = (
        ["text", "image_derived_text"] if all_images else ["text"]
    )


def _map_markdown_prose(text: str, transform: Any) -> str:
    """Transform prose while preserving fenced and inline code verbatim."""

    fenced_parts = FENCED_CODE_RE.split(text)
    for fenced_index in range(0, len(fenced_parts), 2):
        inline_parts = INLINE_CODE_RE.split(fenced_parts[fenced_index])
        for inline_index in range(0, len(inline_parts), 2):
            inline_parts[inline_index] = transform(inline_parts[inline_index])
        fenced_parts[fenced_index] = "".join(inline_parts)
    return "".join(fenced_parts)


def _sanitize_normalized_answer(text: str) -> str:
    """Keep technical URL literals but prevent live/clickable answer links.

    URLs inside code are part of technical examples and remain unchanged.
    Markdown links become their visible labels, while bare prose URLs become
    inline-code literals so Markdown renderers do not turn them into live
    dependencies.
    """

    def sanitize_prose(prose: str) -> str:
        prose = MARKDOWN_LINK_RE.sub(lambda match: match.group(1), prose)

        def protect_url(match: re.Match[str]) -> str:
            raw = match.group(0)
            core = raw.rstrip(".,;:!?")
            return f"`{core}`{raw[len(core):]}"

        return URL_RE.sub(protect_url, prose)

    return _map_markdown_prose(text, sanitize_prose)


def _contains_live_prose_url(text: str) -> bool:
    """Return whether Markdown prose still exposes a live URL."""

    prose_only = FENCED_CODE_RE.sub("", text)
    prose_only = INLINE_CODE_RE.sub("", prose_only)
    return bool(URL_RE.search(prose_only))


def verify_normalized_dataset(dataset: Path, threshold: float) -> dict[str, Any]:
    corpus = _load_jsonl(dataset / "corpus.jsonl")
    questions = _load_jsonl(dataset / "questions.jsonl")
    rejected = _load_jsonl(dataset / "rejected.jsonl")
    errors: list[str] = []
    corpus_by_id = {str(row["doc_id"]): row for row in corpus}
    if len(corpus_by_id) != len(corpus):
        errors.append("duplicate corpus document IDs")
    if len({str(row["question_id"]) for row in questions}) != len(questions):
        errors.append("duplicate question IDs")
    for document in corpus:
        if not str(document.get("rendered_text") or "").strip():
            errors.append(f"empty corpus text: {document['doc_id']}")
        if document.get("image_vector_enabled") is not False:
            errors.append(f"image vector not explicitly disabled: {document['doc_id']}")
        path = DATASET_ROOT / str(document.get("local_path") or "")
        if not path.is_file():
            errors.append(f"missing local corpus file: {document.get('local_path')}")
    for question in questions:
        score = float(question.get("answerability_score") or 0)
        if score <= threshold:
            errors.append(f"score not above threshold: {question['question_id']}={score}")
        if not str(question.get("normalized_answer") or "").strip():
            errors.append(f"empty normalized answer: {question['question_id']}")
        question_image_rows = [
            image
            for image in question.get("image_text_evidence") or []
            if image.get("role") == "question"
        ]
        used_question_images = set(
            map(str, question.get("question_image_text_evidence_used") or [])
        )
        expected_question_images = {
            str(image.get("evidence_id"))
            for image in question_image_rows
            if str(image.get("text") or "").strip()
        }
        if used_question_images != expected_question_images:
            errors.append(
                f"question image text is not fully materialized: {question['question_id']}"
            )
        if expected_question_images and "Image-derived question evidence:" not in str(
            question.get("query") or ""
        ):
            errors.append(
                f"question query omits image-derived text: {question['question_id']}"
            )
        for answer_field in ("normalized_answer", "normalized_answer_with_local_sources"):
            if _contains_live_prose_url(str(question.get(answer_field) or "")):
                errors.append(
                    f"{answer_field} contains live prose URL: {question['question_id']}"
                )
        for doc_id in question.get("qrel_ids") or []:
            if str(doc_id) not in corpus_by_id:
                errors.append(f"missing normalized qrel: {question['question_id']}->{doc_id}")
        for citation in question.get("local_document_citations") or []:
            if not (DATASET_ROOT / str(citation.get("local_path") or "")).is_file():
                errors.append(f"missing citation path: {question['question_id']}->{citation.get('local_path')}")
        if "image_vector" in json.dumps(question, ensure_ascii=False).casefold():
            # retrieval_modalities is allowed to state the policy; actual vector
            # payload keys such as image_embedding are not.
            for key in ("image_embedding", "image_vector_values", "image_vector_id"):
                if key in json.dumps(question, ensure_ascii=False).casefold():
                    errors.append(f"forbidden image-vector payload: {question['question_id']}")
    split_rows: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        rows = json.loads((dataset / "splits" / f"{split}.json").read_text(encoding="utf-8"))
        expected = [row for row in questions if row.get("split") == split]
        split_rows[split] = len(rows)
        if {row["question_id"] for row in rows} != {row["question_id"] for row in expected}:
            errors.append(f"split mismatch: {split}")
    report = {
        "valid": not errors,
        "errors": errors,
        "documents": len(corpus),
        "accepted_questions": len(questions),
        "rejected_questions": len(rejected),
        "minimum_answerability_score": min((float(row["answerability_score"]) for row in questions), default=None),
        "median_answerability_score": percentile((float(row["answerability_score"]) for row in questions), 0.5),
        "questions_with_image_derived_text": sum("image_derived_text" in (row.get("retrieval_modalities") or []) for row in questions),
        "corpus_documents_with_image_derived_text": sum(bool(row.get("image_texts")) for row in corpus),
        "image_vector_enabled": False,
        "splits": split_rows,
    }
    return report


def _build_output(
    args: argparse.Namespace,
    corpus: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    image_texts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    accepted = sorted((row for row in rows if row.get("status") == "accepted"), key=lambda row: str(row["question_id"]))
    rejected = sorted((row for row in rows if row.get("status") != "accepted"), key=lambda row: str(row["question_id"]))
    for row in accepted:
        _upgrade_normalized_question(row)
    questions = [row["normalized_question"] for row in accepted]
    normalized_corpus = _enrich_corpus(corpus, accepted)
    output = args.output_dir
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    complete = False
    try:
        _write_jsonl(staging / "corpus.jsonl", normalized_corpus)
        _write_jsonl(staging / "questions.jsonl", questions)
        _write_jsonl(staging / "rejected.jsonl", rejected)
        _write_jsonl(staging / "image_text.jsonl", [image_texts[key] for key in sorted(image_texts)])
        anchor_rows = [
            {"question_id": row["question_id"], "anchor_resolution": row["normalized_question"].get("anchor_resolution") or []}
            for row in accepted
        ]
        _write_jsonl(staging / "anchor_resolution.jsonl", anchor_rows)
        split_dir = staging / "splits"
        split_dir.mkdir()
        split_counts: dict[str, int] = {}
        for split in ("train", "validation", "test"):
            split_questions = [row for row in questions if row.get("split") == split]
            split_counts[split] = len(split_questions)
            (split_dir / f"{split}.json").write_text(
                json.dumps(split_questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        normalizer_version = getattr(
            args, "materialized_normalizer_version", NORMALIZER_VERSION
        )
        rubric_version = getattr(args, "materialized_rubric_version", RUBRIC_VERSION)
        image_text_version = getattr(
            args, "materialized_image_text_version", IMAGE_TEXT_VERSION
        )
        manifest = {
            "schema_version": 2,
            "name": "docsqa-unified-local-normalized",
            "description": "Standalone locally grounded answers with local document paths and image-derived text.",
            "source_dataset": str(args.dataset),
            "model": args.model,
            "normalizer_version": normalizer_version,
            "rubric_version": rubric_version,
            "image_text_version": image_text_version,
            "answerability_threshold_exclusive": args.threshold,
            "documents": len(normalized_corpus),
            "accepted_questions": len(questions),
            "rejected_questions": len(rejected),
            "splits": split_counts,
            "retrieval_evidence": "modified local dataset only",
            "image_policy": "pixels are converted once to local text; image vectors are disabled",
            "source_corpus_sha256": _sha256(args.dataset / "corpus.jsonl"),
            "source_questions_sha256": _sha256(args.dataset / "questions.jsonl"),
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        verification = verify_normalized_dataset(staging, args.threshold)
        (staging / "verification.json").write_text(
            json.dumps(verification, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if not verification["valid"]:
            raise ValueError(f"normalized dataset verification failed: {verification['errors'][:10]}")
        if output.exists():
            shutil.rmtree(output)
        staging.replace(output)
        complete = True
    finally:
        if not complete and staging.exists():
            shutil.rmtree(staging)
    return {**manifest, "verification": verification}


def _aggregate_model_usage(
    rows: list[dict[str, Any]], image_texts: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    normalization_calls: list[dict[str, Any]] = []
    for row in rows:
        requirement_call = (row.get("evidence_package") or {}).get("requirement_derivation")
        if requirement_call and requirement_call.get("usage"):
            normalization_calls.append(requirement_call)
        for round_row in row.get("rounds") or []:
            for key in ("generation", "grading", "final_grading"):
                call = round_row.get(key)
                if call and call.get("usage"):
                    normalization_calls.append(call)
    image_calls = [row for row in image_texts.values() if row.get("usage")]

    def total(calls: list[dict[str, Any]]) -> dict[str, Any]:
        usage_keys = (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
        )
        result: dict[str, Any] = {
            "calls": len(calls),
            "latency_seconds_sum": sum(float(call.get("latency_seconds") or 0) for call in calls),
        }
        for key in usage_keys:
            result[key] = sum(int((call.get("usage") or {}).get(key) or 0) for call in calls)
        return result

    normalization = total(normalization_calls)
    image = total(image_calls)
    combined = {
        key: normalization[key] + image[key]
        for key in normalization
    }
    return {
        "normalization_and_grading": normalization,
        "image_transcription": image,
        "combined": combined,
    }


def render_report(report: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    accepted = [row for row in rows if row.get("status") == "accepted"]
    rejected = [row for row in rows if row.get("status") != "accepted"]
    reasons = Counter(str(row.get("rejection_reason")) for row in rejected)
    by_project = Counter(str(row.get("project")) for row in accepted)
    rounds = Counter(int((row.get("normalized_question") or {}).get("normalization_rounds") or 0) for row in accepted)
    lines = [
        "# Local Answer-Normalization Result",
        "",
        f"- Source questions: {len(rows)}",
        f"- Accepted: {len(accepted)}",
        f"- Rejected: {len(rejected)}",
        f"- Minimum accepted score: {report['verification']['minimum_answerability_score']}",
        f"- Median accepted score: {report['verification']['median_answerability_score']}",
        f"- Accepted cases with image-derived text available: {report['accepted_with_image_text_available']}",
        f"- Accepted answers that actually use image-derived text: {report['accepted_using_image_text']}",
        f"- Corpus documents enriched with image-derived text: {report['verification']['corpus_documents_with_image_derived_text']}",
        f"- Image-vector retrieval: disabled",
        f"- Final verifier: {'passed' if report['verification']['valid'] else 'failed'}",
        "",
        "Every accepted answer is standalone text plus verified local documentation paths. Resolved image pixels were transcribed into local text artifacts before normalization; unresolved required images reject a case. Technical URL literals in code/configuration are preserved, but live Markdown links are removed and prose URLs are rendered as non-clickable code. The conservative score is the lower of a coverage-oriented grade and an independent falsification-oriented grade.",
        "",
        "## Accepted by project",
        "",
        "| Project | Accepted |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in sorted(by_project.items()))
    lines.extend(["", "## Rejections", "", "| Reason | Count |", "|---|---:|"])
    lines.extend(f"| `{key}` | {value} |" for key, value in sorted(reasons.items()))
    lines.extend(["", "## Normalization iterations", "", "| Generation rounds | Accepted cases |", "|---:|---:|"])
    lines.extend(f"| {key} | {value} |" for key, value in sorted(rounds.items()))
    usage = report["model_usage"]
    lines.extend(
        [
            "",
            "## Recorded model usage",
            "",
            "| Stage | Calls | Input tokens | Cached input | Output tokens | Reasoning tokens | Total tokens | Sum of call latency (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, key in (
        ("Normalization and grading", "normalization_and_grading"),
        ("Image transcription", "image_transcription"),
        ("Combined", "combined"),
    ):
        values = usage[key]
        lines.append(
            f"| {label} | {values['calls']} | {values['input_tokens']} | "
            f"{values['cached_input_tokens']} | {values['output_tokens']} | "
            f"{values['reasoning_tokens']} | {values['total_tokens']} | "
            f"{values['latency_seconds_sum']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Method basis",
            "",
            "- MM-BizRAG (ACL Industry 2026) motivates explicit structure and artifact transformation instead of relying on opaque page/image embeddings.",
            "- SAJA (ACL Industry 2026) motivates a fixed multidimensional rubric rather than a single opaque judge number.",
            "- Bhat and Varma (Findings of ACL 2026) motivate factually verifiable attributes and prompt-variation checks.",
            "- Anthropic's 2026 evaluation guidance motivates multiple graders/trials, outcome grounding, and an explicit unknown/reject path.",
            "- OpenAI's official Responses/Graders documentation provides the structured-output and vision primitives used for one-time image transcription and grading.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from openai import OpenAI

    questions = _load_jsonl(args.dataset / "questions.jsonl")
    if args.question_id:
        wanted = set(args.question_id)
        questions = [row for row in questions if str(row["question_id"]) in wanted]
    if args.limit:
        questions = questions[: args.limit]
    corpus_rows = _load_jsonl(args.dataset / "corpus.jsonl")
    corpus = {str(row["doc_id"]): row for row in corpus_rows}
    resolution_index = build_local_resolution_index(corpus)
    image_rows = {
        str(row["question_id"]): row for row in _load_jsonl(args.image_evidence / "per_question.jsonl")
    }
    args.work_dir.mkdir(parents=True, exist_ok=True)
    contract = _verify_or_create_run_contract(args)
    # The SDK's default retry/timeout window can leave a whole worker pool
    # blocked for many minutes after a rate burst.  This pipeline owns retries
    # in _call_parse, so keep individual failures bounded and observable.
    client = OpenAI(timeout=args.request_timeout, max_retries=0)
    image_texts = transcribe_all_images(
        args,
        client,
        {str(row["question_id"]) for row in questions},
    )

    detail_path = args.work_dir / "per_question.jsonl"
    completed = _load_completed(detail_path) if args.resume else {}
    if not args.resume:
        detail_path.write_text("", encoding="utf-8")
    pending = [row for row in questions if str(row["question_id"]) not in completed]
    failures: list[dict[str, Any]] = []
    quota_exhausted = False
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                normalize_one,
                client,
                args.model,
                question,
                corpus,
                image_rows.get(str(question["question_id"]), {"images": []}),
                image_texts,
                args.cache_root,
                args.threshold,
                args.max_repairs,
                args.reasoning_effort,
                args.attempts,
                resolution_index,
            ): str(question["question_id"])
            for question in pending
        }
        with detail_path.open("a", encoding="utf-8") as handle:
            for future in as_completed(futures):
                question_id = futures[future]
                try:
                    row = future.result()
                except Exception as error:
                    failure = {"question_id": question_id, "error": f"{type(error).__name__}: {error}"}
                    failures.append(failure)
                    print(
                        f"execution failure {question_id}: {failure['error'][:500]}",
                        flush=True,
                    )
                    if "credit_balance_exhausted" in failure["error"] or "insufficient_quota" in failure["error"]:
                        quota_exhausted = True
                        for pending_future in futures:
                            if pending_future is not future:
                                pending_future.cancel()
                        break
                    continue
                row["pipeline_contract"] = contract
                completed[question_id] = row
                with lock:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    if len(completed) == len(questions) or len(completed) % 25 == 0:
                        accepted = sum(value.get("status") == "accepted" for value in completed.values())
                        print(
                            f"normalization progress: {len(completed)}/{len(questions)}; accepted={accepted}; execution_failures={len(failures)}",
                            flush=True,
                        )
    requested = {str(row["question_id"]) for row in questions}
    rows = sorted((row for key, row in completed.items() if key in requested), key=lambda row: str(row["question_id"]))
    for row in rows:
        row["pipeline_contract"] = contract
        if row.get("status") == "accepted":
            _upgrade_normalized_question(row)
    _write_jsonl(detail_path, rows)
    if failures:
        _write_jsonl(args.work_dir / "execution_failures.jsonl", failures)
        if quota_exhausted:
            raise RuntimeError(
                "OpenAI API credit balance is exhausted; completed rows are preserved. "
                "Add credits and rerun the exact command with --resume."
            )
        raise RuntimeError(f"normalization has {len(failures)} execution failures; rerun with --resume")
    report = _build_output(args, corpus_rows, rows, image_texts)
    report["source_questions"] = len(rows)
    report["accepted_questions"] = sum(row.get("status") == "accepted" for row in rows)
    report["rejected_questions"] = sum(row.get("status") != "accepted" for row in rows)
    report["accepted_with_image_text_available"] = sum(
        row.get("status") == "accepted" and bool((row.get("evidence_package") or {}).get("images"))
        for row in rows
    )
    report["accepted_using_image_text"] = sum(
        row.get("status") == "accepted"
        and bool((row.get("normalized_question") or {}).get("image_text_evidence_used"))
        for row in rows
    )
    report["rejection_reasons"] = dict(
        sorted(Counter(str(row.get("rejection_reason")) for row in rows if row.get("status") != "accepted").items())
    )
    report["model_usage"] = _aggregate_model_usage(rows, image_texts)
    (args.work_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.work_dir / "REPORT.md").write_text(render_report(report, rows), encoding="utf-8")
    return report


def materialize_cached_output(args: argparse.Namespace) -> dict[str, Any]:
    """Rebuild the generated dataset from a complete frozen work directory."""

    source_questions = _load_jsonl(args.dataset / "questions.jsonl")
    source_ids = {str(row["question_id"]) for row in source_questions}
    rows = _load_jsonl(args.work_dir / "per_question.jsonl")
    row_ids = {str(row["question_id"]) for row in rows}
    if len(rows) != len(row_ids) or row_ids != source_ids:
        missing = sorted(source_ids - row_ids)
        extra = sorted(row_ids - source_ids)
        raise ValueError(
            "cached normalization rows do not exactly cover the source dataset; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    frozen_contract = rows[0].get("pipeline_contract") or {}
    frozen_versions = {
        key: frozen_contract.get(key)
        for key in (
            "normalizer_version",
            "rubric_version",
            "image_text_version",
        )
    }
    if not all(frozen_versions.values()):
        raise ValueError("cached normalization rows lack frozen version metadata")
    expected = {
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "threshold": args.threshold,
        "max_repairs": args.max_repairs,
        **frozen_versions,
    }
    for row in rows:
        contract = row.get("pipeline_contract") or {}
        mismatched = {
            key: (contract.get(key), value)
            for key, value in expected.items()
            if contract.get(key) != value
        }
        if mismatched:
            raise ValueError(
                f"cached normalization contract mismatch for {row['question_id']}: {mismatched}"
            )
        if row.get("status") == "accepted":
            _upgrade_normalized_question(row)
    args.materialized_normalizer_version = frozen_versions["normalizer_version"]
    args.materialized_rubric_version = frozen_versions["rubric_version"]
    args.materialized_image_text_version = frozen_versions["image_text_version"]
    image_texts = {
        str(row["sha256"]): row
        for row in _load_jsonl(args.work_dir / "image_text.jsonl")
    }
    return _build_output(
        args,
        _load_jsonl(args.dataset / "corpus.jsonl"),
        sorted(rows, key=lambda row: str(row["question_id"])),
        image_texts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "evaluation/dataset/evaluation_data/combined")
    parser.add_argument(
        "--image-evidence",
        type=Path,
        default=ANALYSIS_RUNS / "image-validation-v4",
    )
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "results/cache")
    parser.add_argument(
        "--image-text-cache",
        type=Path,
        default=PROJECT_ROOT / "evaluation/dataset/evaluation_data/normalized/image_text.jsonl",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=ANALYSIS_RUNS / "normalization"
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "evaluation/dataset/evaluation_data/normalized")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--materialize-only",
        action="store_true",
        help="Rebuild output from a complete frozen work directory without model calls.",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.attempts < 1 or args.max_repairs < 0 or args.request_timeout <= 0:
        raise SystemExit("workers/attempts must be positive and max-repairs non-negative")
    if not 0 <= args.threshold < 1:
        raise SystemExit("threshold must be in [0, 1)")
    if args.materialize_only:
        if args.limit or args.question_id:
            raise SystemExit("--materialize-only requires the complete source dataset")
        print(json.dumps(materialize_cached_output(args), ensure_ascii=False, indent=2))
        return
    if not configured_openai_key():
        raise SystemExit("OPENAI_API_KEY is unset; set it or configure KBBENCH_OPENAI_ENV_FILE")
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
