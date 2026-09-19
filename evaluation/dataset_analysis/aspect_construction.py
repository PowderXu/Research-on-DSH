"""Deterministic construction and checking of frozen answer aspects.

These utilities operate on individual records. Benchmark evaluation uses the
published frozen aspects for the whole question pool; no optimizer is run.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field


class AnswerAspect(BaseModel):
    aspect_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    requirement_ids: list[str] = Field(min_length=1)
    claim_ids: list[str] = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    importance: int = Field(ge=1, le=5)
    critical: bool
    rationale: str = Field(min_length=1)


class SourceSupport(BaseModel):
    aspect_id: str = Field(min_length=1)
    support: Literal["full", "partial", "none"]
    explanation: str = Field(min_length=1)


class AspectConstruction(BaseModel):
    aspects: list[AnswerAspect] = Field(min_length=1)
    source_support: list[SourceSupport] = Field(min_length=1)
    excluded_claim_ids: list[str] = Field(default_factory=list)
    coverage_summary: str = Field(min_length=1)


SUPPORT_VALUE = {"full": 1.0, "partial": 0.5, "none": 0.0}


DIRECT_TEXT_EVIDENCE_KINDS = {
    "accepted_answer",
    "image_derived_text",
    "question_context",
    "local_document_section",
}


def _normalize_space(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _compact_evidence(
    rows: list[dict[str, Any]], *, max_total_chars: int = 42_000
) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    remaining = max_total_chars
    for source in rows:
        if remaining <= 0:
            break
        text = str(source.get("text") or "")
        clipped = text[: min(6_000, remaining)]
        row = {
            "evidence_id": str(source.get("evidence_id") or ""),
            "kind": str(source.get("kind") or ""),
            "text": clipped,
        }
        for key in ("doc_id", "heading", "local_path", "constraint"):
            if source.get(key):
                row[key] = source[key]
        compact.append(row)
        remaining -= len(clipped)
    return compact


def normalize_item(row: dict[str, Any]) -> dict[str, Any]:
    question = row.get("normalized_question") or {}
    package = row.get("evidence_package") or {}
    requirements = (package.get("user_requirements") or {}).get("requirements") or []
    claims = question.get("normalized_claims") or []
    evidence = _compact_evidence(package.get("evidence") or [])
    return {
        "id": str(row.get("question_id") or ""),
        "question": str(question.get("query") or package.get("question") or ""),
        "reference_answer": str(question.get("normalized_answer") or ""),
        "project": str(row.get("project") or question.get("dataset") or "unknown"),
        "task_type": str(question.get("evidence_category") or "docsqa"),
        "requirements": requirements,
        "claims": claims,
        "evidence": evidence,
        "source_url": str(question.get("source_url") or ""),
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    start, end = value.find("{"), value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("target response contains no JSON object")
    parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("target response must be a JSON object")
    return parsed


def _target_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": item["id"],
        "question": item["question"],
        "normalized_source_answer": item["reference_answer"],
        "requirements": item["requirements"],
        "normalized_claims": item["claims"],
        "local_evidence": item["evidence"],
    }


def _system_prompt(rule: str) -> str:
    return f"""You construct evaluation aspects for linked-documentation QA.

Apply the same general rule to every record. Do not create case-specific rules,
use outside knowledge, follow URLs, or grade an agent answer. The accepted
source answer is weak supervision: assess its coverage of the aspects you
construct from the question and local evidence.

## Current general rule
{rule.strip()}

Return only JSON with this shape:
{{
  "aspects": [{{
    "aspect_id": "a1",
    "description": "atomic requirement",
    "requirement_ids": ["R1"],
    "claim_ids": ["C1"],
    "evidence_ids": ["accepted_answer"],
    "importance": 1,
    "critical": true,
    "rationale": "why this aspect is needed"
  }}],
  "source_support": [{{
    "aspect_id": "a1",
    "support": "full",
    "explanation": "how the normalized source answer covers it"
  }}],
  "excluded_claim_ids": [],
  "coverage_summary": "short summary"
}}
"""


def score_construction(
    construction: AspectConstruction,
    item: dict[str, Any],
    *,
    threshold: float,
) -> tuple[int, float, list[str]]:
    errors: list[str] = []
    aspects = construction.aspects
    requirement_by_id = {
        str(row.get("requirement_id")): row for row in item.get("requirements") or []
    }
    claim_ids = {str(row.get("claim_id")) for row in item.get("claims") or []}
    evidence_by_id = {
        str(row.get("evidence_id")): row for row in item.get("evidence") or []
    }
    aspect_ids = [aspect.aspect_id for aspect in aspects]
    if len(aspect_ids) != len(set(aspect_ids)):
        errors.append("aspect IDs are not unique")
    descriptions = [_normalize_space(aspect.description) for aspect in aspects]
    if len(descriptions) != len(set(descriptions)):
        errors.append("aspect descriptions are not unique")

    for aspect in aspects:
        unknown_requirements = sorted(set(aspect.requirement_ids) - set(requirement_by_id))
        unknown_claims = sorted(set(aspect.claim_ids) - claim_ids)
        unknown_evidence = sorted(set(aspect.evidence_ids) - set(evidence_by_id))
        if unknown_requirements:
            errors.append(f"{aspect.aspect_id} has unknown requirement IDs {unknown_requirements}")
        if unknown_claims:
            errors.append(f"{aspect.aspect_id} has unknown claim IDs {unknown_claims}")
        if unknown_evidence:
            errors.append(f"{aspect.aspect_id} has unknown evidence IDs {unknown_evidence}")
        for evidence_id in aspect.evidence_ids:
            evidence = evidence_by_id.get(evidence_id) or {}
            if evidence.get("doc_id") and not evidence.get("local_path"):
                errors.append(f"{aspect.aspect_id} maps to non-local document evidence {evidence_id}")

    critical_requirements = {
        requirement_id
        for requirement_id, row in requirement_by_id.items()
        if row.get("critical")
    }
    covered_critical = {
        requirement_id
        for aspect in aspects
        if aspect.critical
        for requirement_id in aspect.requirement_ids
    }
    missing_critical = sorted(critical_requirements - covered_critical)
    if missing_critical:
        errors.append(f"critical requirements lack critical aspects: {missing_critical}")

    support_by_id = {row.aspect_id: row.support for row in construction.source_support}
    if len(support_by_id) != len(construction.source_support):
        errors.append("source-support aspect IDs are not unique")
    missing_support = sorted(set(aspect_ids) - set(support_by_id))
    extra_support = sorted(set(support_by_id) - set(aspect_ids))
    if missing_support:
        errors.append(f"missing source-support ratings: {missing_support}")
    if extra_support:
        errors.append(f"source-support ratings reference unknown aspects: {extra_support}")

    total_weight = sum(aspect.importance for aspect in aspects)
    weighted = sum(
        aspect.importance * SUPPORT_VALUE.get(support_by_id.get(aspect.aspect_id, "none"), 0.0)
        for aspect in aspects
    )
    coverage = weighted / total_weight if total_weight else 0.0
    if errors:
        return 0, 0.0, errors
    return int(coverage >= threshold), round(coverage, 6), []


def bind_evidence_ids_from_claims(
    construction: AspectConstruction, item: dict[str, Any]
) -> tuple[AspectConstruction, list[dict[str, Any]]]:
    """Repair only evidence-ID transcription from normalized claim mappings.

    The LLM decides the aspect semantics and claim mappings. Exact evidence IDs
    are identifiers, not semantic judgments, so the final freeze binds them
    mechanically from the mapped normalized claims. A repair is attempted only
    when the generated list contains an unknown or ineligible ID. Unknown
    requirement or claim IDs remain validation failures.
    """

    evidence_by_id = {
        str(row.get("evidence_id")): row for row in item.get("evidence") or []
    }
    claim_by_id = {
        str(row.get("claim_id")): row for row in item.get("claims") or []
    }

    def eligible(evidence_id: str) -> bool:
        evidence = evidence_by_id.get(evidence_id) or {}
        if evidence.get("kind") not in DIRECT_TEXT_EVIDENCE_KINDS:
            return False
        return not evidence.get("doc_id") or bool(evidence.get("local_path"))

    repaired_aspects: list[AnswerAspect] = []
    repairs: list[dict[str, Any]] = []
    for aspect in construction.aspects:
        generated = list(aspect.evidence_ids)
        if all(eligible(evidence_id) for evidence_id in generated):
            repaired_aspects.append(aspect)
            continue

        candidates: list[str] = []
        seen: set[str] = set()
        for claim_id in aspect.claim_ids:
            claim = claim_by_id.get(claim_id) or {}
            for raw_evidence_id in claim.get("evidence_ids") or []:
                evidence_id = str(raw_evidence_id)
                if evidence_id not in seen and eligible(evidence_id):
                    seen.add(evidence_id)
                    candidates.append(evidence_id)

        if not candidates:
            repaired_aspects.append(aspect)
            continue

        repaired_aspects.append(aspect.model_copy(update={"evidence_ids": candidates}))
        repairs.append(
            {
                "aspect_id": aspect.aspect_id,
                "generated_evidence_ids": generated,
                "bound_evidence_ids": candidates,
            }
        )

    return construction.model_copy(update={"aspects": repaired_aspects}), repairs


def build_frozen_aspect_record(
    item: dict[str, Any], construction: AspectConstruction, rule_sha256: str
) -> dict[str, Any]:
    evidence_by_id = {
        str(row.get("evidence_id")): row for row in item.get("evidence") or []
    }
    total_importance = sum(aspect.importance for aspect in construction.aspects) or 1
    aspects: list[dict[str, Any]] = []
    for aspect in construction.aspects:
        evidence = [
            evidence_by_id[evidence_id]
            for evidence_id in aspect.evidence_ids
            if evidence_id in evidence_by_id
        ]
        aspects.append(
            {
                **aspect.model_dump(),
                "weight": round(aspect.importance / total_importance, 8),
                "evidence": evidence,
                "retrieval_doc_ids": sorted(
                    {
                        str(row["doc_id"])
                        for row in evidence
                        if row.get("kind") == "local_document_section"
                        and row.get("doc_id")
                    }
                ),
            }
        )
    support = {
        row.aspect_id: row.model_dump() for row in construction.source_support
    }
    return {
        "schema_version": "docsqa-frozen-aspects-v1",
        "question_id": item["id"],
        "project": item["project"],
        "question": item["question"],
        "reference_answer": item["reference_answer"],
        "task_type": item.get("task_type") or "docsqa",
        "rule_sha256": rule_sha256,
        "aspects": aspects,
        "source_support": support,
        "coverage_summary": construction.coverage_summary,
        "excluded_claim_ids": construction.excluded_claim_ids,
    }
