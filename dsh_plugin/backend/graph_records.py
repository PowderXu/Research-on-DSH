"""Build query-blind structural, image, and KGGen evidence records."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import Any, Iterable

from .graph_contract import SCHEMA_VERSION


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
MARKDOWN_IMAGE_RE = re.compile(
    r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+[\"']([^\"']*)[\"'])?\)",
    re.MULTILINE,
)
HTML_IMAGE_RE = re.compile(
    r"<(?:img|image)\b([^>]*?)\bsrc=[\"']([^\"']+)[\"']([^>]*)>",
    re.IGNORECASE | re.DOTALL,
)
HTML_ALT_RE = re.compile(r"\balt=[\"']([^\"']*)[\"']", re.IGNORECASE)
HTML_TITLE_RE = re.compile(r"\btitle=[\"']([^\"']*)[\"']", re.IGNORECASE)


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}-{digest}"


def _project(row: dict[str, Any]) -> str:
    explicit = str(row.get("project") or "").strip()
    if explicit:
        return explicit
    doc_id = str(row.get("doc_id") or "")
    return doc_id.split("::", 1)[0] if "::" in doc_id else "default"


def build_graph_snapshot(
    corpus_rows: list[dict[str, Any]],
    chunks: Iterable[Any],
    embeddings: Any,
    *,
    embedding_model_name: str,
    semantic_artifact: dict[str, Any] | None,
) -> dict[str, Any]:
    """Fingerprint every local input that determines the materialized graph."""

    chunk_rows = [
        {
            "chunk_id": str(chunk.chunk_id),
            "doc_id": str(chunk.doc_id),
            "route": str(getattr(chunk, "route", "")),
            "title": str(chunk.title),
            "text": str(chunk.text),
        }
        for chunk in chunks
    ]

    def canonical_hash(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    embedding_digest = hashlib.sha256()
    embedding_digest.update(embedding_model_name.encode("utf-8"))
    embedding_digest.update(str(getattr(embeddings, "shape", "")).encode("utf-8"))
    embedding_digest.update(str(getattr(embeddings, "dtype", "")).encode("utf-8"))
    embedding_digest.update(embeddings.tobytes(order="C"))
    snapshot = {
        "snapshot_id": "docsqa",
        "schema_version": SCHEMA_VERSION,
        "corpus_sha256": canonical_hash(corpus_rows),
        "chunks_sha256": canonical_hash(chunk_rows),
        "embeddings_sha256": embedding_digest.hexdigest(),
        "embedding_model": embedding_model_name,
        "kggen_sha256": (
            canonical_hash(semantic_artifact)
            if semantic_artifact is not None
            else "none"
        ),
    }
    snapshot["snapshot_sha256"] = canonical_hash(snapshot)
    return snapshot


def _heading_at(text: str, offset: int, fallback: str) -> tuple[str, int]:
    title = fallback
    level = 1
    for match in HEADING_RE.finditer(text, 0, max(0, offset) + 1):
        title = match.group(2).strip()
        level = len(match.group(1))
    return title, level


def _section_id(doc_id: str, title: str, level: int) -> str:
    return _stable_id("section", doc_id, str(level), title.casefold())


def _image_references(text: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for match in MARKDOWN_IMAGE_RE.finditer(text):
        values.append(
            {
                "offset": match.start(),
                "url": match.group(2).strip(),
                "alt": match.group(1).strip(),
                "title": (match.group(3) or "").strip(),
                "source_syntax": "markdown",
            }
        )
    for match in HTML_IMAGE_RE.finditer(text):
        attributes = f"{match.group(1)} {match.group(3)}"
        alt = HTML_ALT_RE.search(attributes)
        title = HTML_TITLE_RE.search(attributes)
        values.append(
            {
                "offset": match.start(),
                "url": match.group(2).strip(),
                "alt": alt.group(1).strip() if alt else "",
                "title": title.group(1).strip() if title else "",
                "source_syntax": "html_or_mdx",
            }
        )
    return sorted(values, key=lambda item: (int(item["offset"]), str(item["url"])))


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, max(0, offset)) + 1


def _chunk_source_offset(source: str, chunk_text: str, fallback: int) -> int:
    """Recover a chunk's approximate source position without fixed line cutoffs."""

    body = chunk_text.split("\n", 2)[-1].strip()
    if not body:
        return fallback
    probes = (body, body[:500], body[:200], body[:80])
    for probe in probes:
        if probe:
            found = source.find(probe)
            if found >= 0:
                return found
    return fallback


def build_graph_records(
    corpus_rows: list[dict[str, Any]],
    chunks: Iterable[Any],
) -> dict[str, list[dict[str, Any]]]:
    """Project the neutral corpus into deterministic Neo4j ingestion records.

    Text units retain the benchmark chunk IDs. Images are additional retrieval
    units with text representations and occurrence-level provenance. No
    questions, answers, qrels, image vectors, or query-time information enter
    this transformation.
    """

    chunks_by_doc: dict[str, list[Any]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_doc[str(chunk.doc_id)].append(chunk)

    projects: dict[str, dict[str, Any]] = {}
    documents: list[dict[str, Any]] = []
    sections: dict[str, dict[str, Any]] = {}
    units: list[dict[str, Any]] = []
    next_edges: list[dict[str, str]] = []
    assets: dict[str, dict[str, Any]] = {}
    occurrences: list[dict[str, Any]] = []
    near_edges: list[dict[str, str]] = []

    for row in corpus_rows:
        doc_id = str(row["doc_id"])
        title = str(row.get("title") or doc_id)
        project_id = _project(row)
        source_path = str(row.get("source_path") or "")
        text = str(row.get("rendered_text") or "")
        projects.setdefault(project_id, {"project_id": project_id})
        documents.append(
            {
                "doc_id": doc_id,
                "project_id": project_id,
                "title": title,
                "source_path": source_path,
                "route": str(row.get("route") or doc_id),
                "content_type": str(row.get("content_type") or ""),
                "variant_conditioned": bool(row.get("variant_conditioned")),
            }
        )

        positioned_units: list[tuple[int, int, dict[str, Any]]] = []
        doc_chunks = chunks_by_doc.get(doc_id, [])
        for ordinal, chunk in enumerate(doc_chunks):
            chunk_text = str(chunk.text)
            source_offset = _chunk_source_offset(text, chunk_text, ordinal * 1_400)
            heading_match = next(iter(HEADING_RE.finditer(chunk_text)), None)
            section_title = heading_match.group(2).strip() if heading_match else title
            section_level = len(heading_match.group(1)) if heading_match else 1
            section_id = _section_id(doc_id, section_title, section_level)
            sections.setdefault(
                section_id,
                {
                    "section_id": section_id,
                    "doc_id": doc_id,
                    "title": section_title,
                    "level": section_level,
                    "ordinal": ordinal,
                },
            )
            unit_id = str(chunk.chunk_id)
            unit = {
                "unit_id": unit_id,
                "doc_id": doc_id,
                "project_id": project_id,
                "section_id": section_id,
                "unit_type": "text_chunk",
                "ordinal": ordinal,
                "title": title,
                "search_text": chunk_text,
                "source_path": source_path,
                "source_line": _line_number(text, source_offset),
                "asset_id": None,
            }
            units.append(unit)
            positioned_units.append((source_offset, 0, unit))

        image_rows = _image_references(text)
        image_text_by_alt = {
            str(item.get("alt") or "").strip(): item
            for item in row.get("image_texts") or []
            if str(item.get("alt") or "").strip()
        }
        for image_ordinal, image in enumerate(image_rows):
            url = str(image["url"])
            heading, level = _heading_at(text, int(image["offset"]), title)
            section_id = _section_id(doc_id, heading, level)
            sections.setdefault(
                section_id,
                {
                    "section_id": section_id,
                    "doc_id": doc_id,
                    "title": heading,
                    "level": level,
                    "ordinal": len(sections),
                },
            )
            occurrence_id = _stable_id(
                "image-occurrence", doc_id, str(image_ordinal), url
            )
            asset_id = _stable_id("image-asset", project_id, url)
            alt = str(image.get("alt") or "")
            enriched = image_text_by_alt.get(alt.strip(), {})
            image_title = str(image.get("title") or "")
            ocr_text = ""
            description = str(enriched.get("text") or "")
            retrieval_parts = [
                title,
                heading,
                alt,
                image_title,
                description,
            ]
            retrieval_text = "\n".join(
                part for part in retrieval_parts if part.strip()
            )
            assets.setdefault(
                asset_id,
                {
                    "asset_id": asset_id,
                    "project_id": project_id,
                    "source_url": url,
                    "content_hash": enriched.get("sha256"),
                    "local_path": None,
                    "mime_type": None,
                    "width": None,
                    "height": None,
                    "description": description,
                    "ocr_text": ocr_text,
                },
            )
            unit = {
                "unit_id": occurrence_id,
                "doc_id": doc_id,
                "project_id": project_id,
                "section_id": section_id,
                "unit_type": "image_occurrence",
                "ordinal": len(doc_chunks) + image_ordinal,
                "title": title,
                "search_text": retrieval_text,
                "source_path": source_path,
                "source_line": _line_number(text, int(image["offset"])),
                "asset_id": asset_id,
            }
            units.append(unit)
            occurrences.append(
                {
                    **unit,
                    "occurrence_id": occurrence_id,
                    "alt_text": alt,
                    "image_title": image_title,
                    "source_url": url,
                    "source_syntax": str(image["source_syntax"]),
                }
            )

            nearest = next(
                (
                    str(chunk.chunk_id)
                    for chunk in doc_chunks
                    if url in str(chunk.text) or (alt and alt in str(chunk.text))
                ),
                str(doc_chunks[0].chunk_id) if doc_chunks else "",
            )
            if nearest:
                near_edges.append({"occurrence_id": occurrence_id, "unit_id": nearest})
            positioned_units.append((int(image["offset"]), 1, unit))

        positioned_units.sort(key=lambda item: (item[0], item[1], item[2]["unit_id"]))
        document_units = [row["unit_id"] for _, _, row in positioned_units]
        for ordinal, (_, _, unit) in enumerate(positioned_units):
            unit["ordinal"] = ordinal
        for before, after in zip(document_units, document_units[1:]):
            next_edges.append({"source": before, "target": after})

    return {
        "projects": sorted(projects.values(), key=lambda row: row["project_id"]),
        "documents": documents,
        "sections": sorted(sections.values(), key=lambda row: row["section_id"]),
        "units": units,
        "next_edges": next_edges,
        "image_assets": sorted(assets.values(), key=lambda row: row["asset_id"]),
        "image_occurrences": occurrences,
        "near_edges": near_edges,
}
