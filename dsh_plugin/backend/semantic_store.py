"""Validation and Neo4j ingestion for provenance-preserving KGGen artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from .graph_contract import (
    CLAIM_LABEL,
    ENTITY_LABEL,
    HAS_OBJECT,
    PREDICATE_LABEL,
    SUBJECT_OF,
    SUPPORTED_BY,
    UNIT_LABEL,
    USES_PREDICATE,
)


def _batched(values: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def load_semantic_artifact(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_semantic_artifact(payload)
    return payload


def validate_semantic_artifact(payload: dict[str, Any]) -> None:
    if int(payload.get("schema_version") or 0) != 1:
        raise ValueError("unsupported KGGen artifact schema_version")
    entities = list(payload.get("entities") or [])
    predicates = list(payload.get("predicates") or [])
    if any(not str(row.get("project_id") or "") for row in entities):
        raise ValueError("KGGen entities must be project scoped")
    if any(not str(row.get("project_id") or "") for row in predicates):
        raise ValueError("KGGen predicates must be project scoped")
    entity_ids = {str(row["entity_id"]) for row in entities}
    predicate_ids = {str(row["predicate_id"]) for row in predicates}
    claim_ids: set[str] = set()
    for row in payload.get("claims") or []:
        claim_id = str(row.get("claim_id") or "")
        if not claim_id or claim_id in claim_ids:
            raise ValueError("KGGen claims must have unique claim_id values")
        claim_ids.add(claim_id)
        if str(row.get("subject_id")) not in entity_ids:
            raise ValueError(f"claim {claim_id} has an unknown subject")
        if str(row.get("object_id")) not in entity_ids:
            raise ValueError(f"claim {claim_id} has an unknown object")
        if str(row.get("predicate_id")) not in predicate_ids:
            raise ValueError(f"claim {claim_id} has an unknown predicate")
        if not str(row.get("evidence_unit_id") or ""):
            raise ValueError(f"claim {claim_id} has no evidence unit")
        if not str(row.get("evidence_excerpt") or ""):
            raise ValueError(f"claim {claim_id} has no evidence excerpt")
        if not str(row.get("project_id") or ""):
            raise ValueError(f"claim {claim_id} has no project scope")


def ingest_semantic_artifact(
    execute: Callable[..., list[Any]],
    payload: dict[str, Any],
    *,
    batch_size: int = 250,
) -> dict[str, int]:
    """Write canonical KGGen entities, predicates and evidence-bound claims."""

    validate_semantic_artifact(payload)
    entities = list(payload.get("entities") or [])
    predicates = list(payload.get("predicates") or [])
    claims = list(payload.get("claims") or [])

    for batch in _batched(entities, batch_size):
        execute(
            f"""
            UNWIND $rows AS row
            CREATE (entity:{ENTITY_LABEL} {{
              entity_id: row.entity_id,
              project_id: row.project_id,
              canonical_name: row.canonical_name,
              aliases: row.aliases,
              claim_count: row.claim_count
            }})
            """,
            rows=batch,
        )
    for batch in _batched(predicates, batch_size):
        execute(
            f"""
            UNWIND $rows AS row
            CREATE (predicate:{PREDICATE_LABEL} {{
              predicate_id: row.predicate_id,
              project_id: row.project_id,
              canonical_name: row.canonical_name,
              aliases: row.aliases,
              family: row.family
            }})
            """,
            rows=batch,
        )
    for batch in _batched(claims, batch_size):
        execute(
            f"""
            UNWIND $rows AS row
            MATCH (subject:{ENTITY_LABEL} {{entity_id: row.subject_id}})
            MATCH (predicate:{PREDICATE_LABEL} {{predicate_id: row.predicate_id}})
            MATCH (object:{ENTITY_LABEL} {{entity_id: row.object_id}})
            MATCH (unit:{UNIT_LABEL} {{unit_id: row.evidence_unit_id}})
            CREATE (claim:{CLAIM_LABEL} {{
              claim_id: row.claim_id,
              project_id: row.project_id,
              raw_subject: row.raw_subject,
              raw_predicate: row.raw_predicate,
              raw_object: row.raw_object,
              evidence_excerpt: row.evidence_excerpt,
              extraction_method: row.extraction_method,
              confidence: row.confidence,
              polarity: row.polarity,
              condition: row.condition
            }})
            CREATE (subject)-[:{SUBJECT_OF}]->(claim)
            CREATE (claim)-[:{HAS_OBJECT}]->(object)
            CREATE (claim)-[:{USES_PREDICATE}]->(predicate)
            CREATE (claim)-[:{SUPPORTED_BY}]->(unit)
            """,
            rows=batch,
        )
    return {
        "entities": len(entities),
        "predicates": len(predicates),
        "claims": len(claims),
    }
