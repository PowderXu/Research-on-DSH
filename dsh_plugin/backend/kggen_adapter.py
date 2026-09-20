"""KGGen adapter that restores evidence provenance after graph clustering.

KGGen deliberately owns open entity/relation extraction and alias clustering.
This adapter owns only DocsQA-specific input/output boundaries: stable evidence
unit IDs, source excerpts, deterministic identifiers, and a serializable graph
artifact that can be ingested by Neo4j without reading benchmark questions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol


class KGGenLike(Protocol):
    def generate(self, input_data: str, **kwargs: Any) -> Any: ...
    def aggregate(self, graphs: list[Any]) -> Any: ...
    def cluster(self, graph: Any, **kwargs: Any) -> Any: ...


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}-{digest}"


def _values(value: Any) -> list[str]:
    if value is None:
        return []
    return sorted({str(item).strip() for item in value if str(item).strip()})


def _relations(graph: Any) -> list[tuple[str, str, str]]:
    output: list[tuple[str, str, str]] = []
    for relation in getattr(graph, "relations", ()) or ():
        if not isinstance(relation, (tuple, list)) or len(relation) != 3:
            continue
        subject, predicate, object_ = (str(value).strip() for value in relation)
        if subject and predicate and object_:
            output.append((subject, predicate, object_))
    return sorted(set(output))


def _cluster_map(graph: Any, attribute: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    raw = getattr(graph, attribute, {}) or {}
    for canonical, members in dict(raw).items():
        canonical_name = str(canonical).strip()
        if not canonical_name:
            continue
        aliases[canonical_name.casefold()] = canonical_name
        for member in members or ():
            value = str(member).strip()
            if value:
                aliases[value.casefold()] = canonical_name
    return aliases


def _canonical(value: str, aliases: dict[str, str]) -> str:
    return aliases.get(value.casefold(), value.strip())


def _evidence_excerpt(text: str, subject: str, object_: str, limit: int = 800) -> str:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    subject_key = subject.casefold()
    object_key = object_.casefold()
    for sentence in sentences:
        lowered = sentence.casefold()
        if subject_key in lowered and object_key in lowered:
            return sentence.strip()[:limit]
    for sentence in sentences:
        lowered = sentence.casefold()
        if subject_key in lowered or object_key in lowered:
            return sentence.strip()[:limit]
    return text.strip()[:limit]


def _cluster_members(graph: Any, attribute: str, canonical: str) -> list[str]:
    raw = getattr(graph, attribute, {}) or {}
    for key, members in dict(raw).items():
        if str(key).strip() == canonical:
            return sorted({canonical, *_values(members)})
    return [canonical]


def _project_id(unit: dict[str, Any]) -> str:
    explicit = str(unit.get("project_id") or "").strip()
    if explicit:
        return explicit
    doc_id = str(unit.get("doc_id") or "")
    return doc_id.split("::", 1)[0] if "::" in doc_id else "default"


def materialize_semantic_artifact(
    accepted_units: list[dict[str, Any]],
    raw_claims: list[dict[str, str]],
    clustered: Any,
    *,
    model: str,
    context: str,
    extractor_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert clustered KGGen output into the evidence-bound schema.

    The scalable extractor uses the same materializer after batching API calls
    and aligning each triple back to its source retrieval unit.
    """

    entity_aliases = _cluster_map(clustered, "entity_clusters")
    predicate_aliases = _cluster_map(clustered, "edge_clusters")

    entity_names: set[tuple[str, str]] = set()
    predicate_names: set[tuple[str, str]] = set()
    for raw in raw_claims:
        project_id = raw["project_id"]
        entity_names.add(
            (project_id, _canonical(raw["raw_subject"], entity_aliases))
        )
        entity_names.add(
            (project_id, _canonical(raw["raw_object"], entity_aliases))
        )
        predicate_names.add(
            (project_id, _canonical(raw["raw_predicate"], predicate_aliases))
        )

    entities = [
        {
            "entity_id": _stable_id("entity", project_id, name.casefold()),
            "project_id": project_id,
            "canonical_name": name,
            "aliases": _cluster_members(clustered, "entity_clusters", name),
        }
        for project_id, name in sorted(
            entity_names, key=lambda item: (item[0], item[1].casefold())
        )
        if name
    ]
    predicates = [
        {
            "predicate_id": _stable_id("predicate", project_id, name.casefold()),
            "project_id": project_id,
            "canonical_name": name,
            "aliases": _cluster_members(clustered, "edge_clusters", name),
            "family": "unclassified",
        }
        for project_id, name in sorted(
            predicate_names, key=lambda item: (item[0], item[1].casefold())
        )
        if name
    ]
    entity_ids = {
        (row["project_id"], row["canonical_name"]): row["entity_id"]
        for row in entities
    }
    predicate_ids = {
        (row["project_id"], row["canonical_name"]): row["predicate_id"]
        for row in predicates
    }

    claims: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for raw in raw_claims:
        subject = _canonical(raw["raw_subject"], entity_aliases)
        predicate = _canonical(raw["raw_predicate"], predicate_aliases)
        object_ = _canonical(raw["raw_object"], entity_aliases)
        project_id = raw["project_id"]
        key = (project_id, subject, predicate, object_, raw["unit_id"])
        if key in seen:
            continue
        seen.add(key)
        claims.append(
            {
                "claim_id": _stable_id("claim", *key),
                "project_id": project_id,
                "subject_id": entity_ids[(project_id, subject)],
                "predicate_id": predicate_ids[(project_id, predicate)],
                "object_id": entity_ids[(project_id, object_)],
                "evidence_unit_id": raw["unit_id"],
                "raw_subject": raw["raw_subject"],
                "raw_predicate": raw["raw_predicate"],
                "raw_object": raw["raw_object"],
                "evidence_excerpt": raw["evidence_excerpt"],
                "evidence_alignment": raw.get("evidence_alignment", "direct_unit"),
                "extraction_method": "kg-gen",
                "confidence": None,
                "polarity": "unspecified",
                "condition": "",
            }
        )

    claim_counts: dict[str, int] = {}
    for claim in claims:
        for entity_id in (claim["subject_id"], claim["object_id"]):
            claim_counts[entity_id] = claim_counts.get(entity_id, 0) + 1
    for entity in entities:
        entity["claim_count"] = claim_counts.get(entity["entity_id"], 0)

    return {
        "schema_version": 1,
        "extractor": "kg-gen",
        "extractor_version": "0.4.0",
        "model": model,
        "context": context,
        "units_processed": len(accepted_units),
        "projects_processed": sorted(
            {str(unit["project_id"]) for unit in accepted_units}
        ),
        "entities": entities,
        "predicates": predicates,
        "claims": claims,
        **(extractor_metadata or {}),
    }


def build_semantic_artifact(
    units: Iterable[dict[str, Any]],
    *,
    model: str,
    context: str,
    kggen_factory: Callable[..., KGGenLike],
    api_key: str | None = None,
    base_url: str | None = None,
    max_units: int | None = None,
) -> dict[str, Any]:
    """Run KGGen and return canonical triples with occurrence provenance."""

    constructor: dict[str, Any] = {
        "model": model,
        "temperature": 1.0 if "gpt-5" in model else 0.0,
    }
    if api_key:
        constructor["api_key"] = api_key
    if base_url:
        constructor["api_base"] = base_url
    kggen = kggen_factory(**constructor)

    accepted_units: list[dict[str, Any]] = []
    raw_graphs: list[Any] = []
    raw_claims: list[dict[str, str]] = []
    for unit in units:
        text = str(unit.get("search_text") or unit.get("text") or "").strip()
        unit_id = str(unit.get("unit_id") or "").strip()
        if not unit_id or len(text) < 20:
            continue
        project_id = _project_id(unit)
        accepted_units.append({**unit, "project_id": project_id, "search_text": text})
        graph = kggen.generate(input_data=text, context=context, cluster=False)
        raw_graphs.append(graph)
        for subject, predicate, object_ in _relations(graph):
            raw_claims.append(
                {
                    "unit_id": unit_id,
                    "project_id": project_id,
                    "raw_subject": subject,
                    "raw_predicate": predicate,
                    "raw_object": object_,
                    "evidence_excerpt": _evidence_excerpt(text, subject, object_),
                }
            )
        if max_units is not None and len(accepted_units) >= max_units:
            break

    if not raw_graphs:
        return {
            "schema_version": 1,
            "extractor": "kg-gen",
            "model": model,
            "context": context,
            "units_processed": 0,
            "projects_processed": [],
            "entities": [],
            "predicates": [],
            "claims": [],
        }

    combined = kggen.aggregate(raw_graphs)
    clustered = kggen.cluster(combined, context=context)
    return materialize_semantic_artifact(
        accepted_units,
        raw_claims,
        clustered,
        model=model,
        context=context,
    )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="deepseek/deepseek-chat")
    parser.add_argument(
        "--context",
        default=(
            "Technical product documentation. Preserve negation, prerequisites, "
            "version and product scope; merge only truly equivalent entities and predicates."
        ),
    )
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--base-url")
    parser.add_argument("--max-units", type=int)
    args = parser.parse_args()

    try:
        from kg_gen import KGGen
    except ImportError as error:
        raise SystemExit(
            "kg-gen is not installed; create the isolated KGGen environment from "
            "dsh_plugin/backend/requirements-kggen.txt"
        ) from error

    started = time.perf_counter()
    artifact = build_semantic_artifact(
        _jsonl(args.input),
        model=args.model,
        context=args.context,
        kggen_factory=KGGen,
        api_key=os.environ.get(args.api_key_env),
        base_url=args.base_url,
        max_units=args.max_units,
    )
    artifact["build_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "units": artifact["units_processed"],
                "entities": len(artifact["entities"]),
                "predicates": len(artifact["predicates"]),
                "claims": len(artifact["claims"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
