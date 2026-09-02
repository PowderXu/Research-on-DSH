"""Evaluate DSH final answers against frozen, evidence-backed answer aspects."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import tempfile
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

from .credentials import load_openai_key_from_configured_env
from .report import (
    EXPECTED_ARMS,
    EXPECTED_FULL_QUESTIONS,
    load_arm_rollouts,
    validate_arm_rollouts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUBRIC = (
    PROJECT_ROOT / "evaluation/dataset_analysis/rubrics/aspect_evaluation_v2.json"
)
ARM_ORDER = EXPECTED_ARMS


class AspectScore(BaseModel):
    aspect_id: str
    coverage: Literal[0.0, 0.5, 1.0]
    support: Literal["full", "partial", "absent", "unsupported", "contradicted"]
    evidence_ids: list[str]
    explanation: str


class ClaimIssue(BaseModel):
    issue_type: Literal["unsupported", "contradicted"]
    severity: Literal["minor", "material", "critical"]
    claim: str
    explanation: str


class CandidateAspectJudgment(BaseModel):
    candidate: str
    aspects: list[AspectScore] = Field(min_length=1)
    claim_issues: list[ClaimIssue]
    invalid_citations: list[str]
    overall_quality: int = Field(ge=1, le=5)
    explanation: str


class AspectBatchJudgment(BaseModel):
    candidates: list[CandidateAspectJudgment] = Field(min_length=1)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _unique_index(
    rows: list[dict[str, Any]], key: str, *, source: Path
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        value = str(row.get(key) or "")
        if not value:
            raise ValueError(f"row in {source} lacks {key}")
        if value in index:
            duplicates.append(value)
        index[value] = row
    if duplicates:
        raise ValueError(f"duplicate {key} values in {source}: {sorted(set(duplicates))}")
    return index


def load_judge_rubric(path: Path = DEFAULT_RUBRIC) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    forbidden = {
        "question_overrides",
        "project_overrides",
        "dataset_overrides",
        "case_overrides",
        "expected_labels_by_id",
    }
    present = sorted(forbidden & set(payload))
    if present:
        raise ValueError(f"judge rubric contains forbidden overrides: {present}")
    for field in ("rubric_version", "judge_system", "judge_rules"):
        if field not in payload:
            raise ValueError(f"judge rubric lacks {field}")
    return payload, hashlib.sha256(raw).hexdigest()


def _aliases(question_id: str, names: list[str]) -> dict[str, str]:
    ordered = sorted(names)
    offset = int(hashlib.sha256(question_id.encode()).hexdigest()[:8], 16) % len(ordered)
    rotated = ordered[offset:] + ordered[:offset]
    return {name: chr(ord("A") + index) for index, name in enumerate(rotated)}


def _candidate_payload(
    label: str,
    row: dict[str, Any],
    corpus: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    answer = str(row.get("answer") or row.get("predicted_answer") or "")
    ranked_ids = list(dict.fromkeys(map(str, row.get("ranked_ids") or [])))[:10]
    evidence = []
    for doc_id in ranked_ids:
        document = corpus.get(doc_id)
        if not document:
            continue
        evidence.append(
            {
                "evidence_id": doc_id,
                "doc_id": doc_id,
                "title": str(document.get("title") or ""),
                "text": str(document.get("rendered_text") or "")[:6_000],
            }
        )
    return {
        "candidate": label,
        "answer": answer[:12_000],
        "agent_completed": bool(row.get("agent_ok", True)),
        "cited_or_retrieved_ids": ranked_ids,
        "candidate_evidence": evidence,
    }


def build_judge_prompt(
    aspect_record: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    corpus: dict[str, dict[str, Any]],
    rubric: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    question_id = str(aspect_record["question_id"])
    aliases = _aliases(question_id, list(candidates))
    gold_evidence: dict[str, dict[str, Any]] = {}
    aspects = []
    for aspect in aspect_record.get("aspects") or []:
        aspects.append(
            {
                key: aspect.get(key)
                for key in (
                    "aspect_id",
                    "description",
                    "importance",
                    "weight",
                    "critical",
                    "requirement_ids",
                    "evidence_ids",
                )
            }
        )
        for evidence in aspect.get("evidence") or []:
            evidence_id = str(evidence.get("evidence_id") or "")
            if evidence_id:
                gold_evidence[evidence_id] = {
                    key: evidence.get(key)
                    for key in ("evidence_id", "kind", "doc_id", "heading", "local_path", "text")
                    if evidence.get(key) is not None
                }
    payload = {
        "rubric_version": rubric["rubric_version"],
        "question_id": question_id,
        "question": aspect_record["question"],
        "frozen_aspects": aspects,
        "reference_answer": aspect_record["reference_answer"],
        "gold_local_evidence": list(gold_evidence.values()),
        "rules": rubric["judge_rules"],
        "candidates": [
            _candidate_payload(aliases[name], candidates[name], corpus)
            for name in sorted(candidates, key=lambda value: aliases[value])
        ],
    }
    return payload, aliases


def _usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    details = getattr(usage, "input_tokens_details", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "cached_input_tokens": int(getattr(details, "cached_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def judge_batch(
    client: Any,
    *,
    model: str,
    reasoning_effort: str,
    rubric: dict[str, Any],
    payload: dict[str, Any],
) -> tuple[AspectBatchJudgment, dict[str, Any]]:
    prompt = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    started = time.perf_counter()
    response = client.responses.parse(
        model=model,
        instructions=str(rubric["judge_system"]),
        input=[{"role": "user", "content": prompt}],
        text_format=AspectBatchJudgment,
        reasoning={"effort": reasoning_effort},
        store=False,
    )
    parsed = response.output_parsed
    if parsed is None:
        raise ValueError("aspect judge returned no structured output")
    return parsed, {
        "latency_seconds": round(time.perf_counter() - started, 6),
        "usage": _usage(response),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }


def _call_cache_path(output_dir: Path, question_id: str) -> Path:
    digest = hashlib.sha256(question_id.encode("utf-8")).hexdigest()
    return output_dir / "call_cache" / f"{digest}.json"


def _load_cached_call(
    path: Path,
    *,
    question_id: str,
    rubric_sha256: str,
    model: str,
    reasoning_effort: str,
    prompt_sha256: str,
) -> tuple[AspectBatchJudgment, dict[str, Any], dict[str, str]] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "question_id": question_id,
        "rubric_sha256": rubric_sha256,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "prompt_sha256": prompt_sha256,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        return None
    return (
        AspectBatchJudgment.model_validate(payload["judgment"]),
        dict(payload["meta"]),
        {str(key): str(value) for key, value in payload["aliases"].items()},
    )


def _write_cached_call(
    path: Path,
    *,
    question_id: str,
    rubric_sha256: str,
    model: str,
    reasoning_effort: str,
    prompt_sha256: str,
    aliases: dict[str, str],
    judgment: AspectBatchJudgment,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "question_id": question_id,
        "rubric_sha256": rubric_sha256,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "prompt_sha256": prompt_sha256,
        "aliases": aliases,
        "judgment": judgment.model_dump(),
        "meta": meta,
    }
    # Each question has its own cache path. A unique temporary file plus atomic
    # replace also keeps writes safe if separate processes happen to evaluate
    # the same question concurrently.
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            temporary.flush()
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def compute_candidate_score(
    aspect_record: dict[str, Any],
    judgment: CandidateAspectJudgment,
    *,
    agent_ok: bool = True,
) -> dict[str, Any]:
    expected = {str(row["aspect_id"]): row for row in aspect_record.get("aspects") or []}
    actual: dict[str, AspectScore] = {}
    duplicate_ids: list[str] = []
    for row in judgment.aspects:
        if row.aspect_id in actual:
            duplicate_ids.append(row.aspect_id)
        actual[row.aspect_id] = row
    if set(actual) != set(expected) or duplicate_ids:
        raise ValueError(
            f"judge aspect IDs differ for {aspect_record['question_id']}: "
            f"actual={sorted(actual)} expected={sorted(expected)} duplicates={duplicate_ids}"
        )
    def aspect_weight(aspect_id: str) -> float:
        row = expected[aspect_id]
        return float(row.get("weight") or row.get("importance") or 1)

    def weighted_coverage(aspect_ids: set[str]) -> float | None:
        if not aspect_ids:
            return None
        denominator = sum(aspect_weight(aspect_id) for aspect_id in aspect_ids)
        return sum(
            aspect_weight(aspect_id) * float(actual[aspect_id].coverage)
            for aspect_id in aspect_ids
        ) / denominator

    def retrieval_doc_ids(row: dict[str, Any]) -> set[str]:
        explicit = row.get("retrieval_doc_ids")
        if explicit is not None:
            return {str(value) for value in explicit if str(value)}
        return {
            str(evidence["doc_id"])
            for evidence in row.get("evidence") or []
            if evidence.get("kind") == "local_document_section"
            and evidence.get("doc_id")
        }

    all_ids = set(expected)
    corpus_ids = {
        aspect_id
        for aspect_id, row in expected.items()
        if retrieval_doc_ids(row)
    }
    weighted = float(weighted_coverage(all_ids) or 0.0)
    corpus_weighted = weighted_coverage(corpus_ids)
    critical_ids = {key for key, row in expected.items() if bool(row.get("critical"))}
    corpus_critical_ids = critical_ids & corpus_ids
    critical_full = all(actual[key].coverage == 1.0 and actual[key].support == "full" for key in critical_ids)
    critical_error = any(
        actual[key].support in {"unsupported", "contradicted"} for key in critical_ids
    )
    consequential_issues = [
        row for row in judgment.claim_issues if row.severity in {"material", "critical"}
    ]
    material_unsupported_claims = [
        row.claim for row in consequential_issues if row.issue_type == "unsupported"
    ]
    material_contradictions = [
        row.claim for row in consequential_issues if row.issue_type == "contradicted"
    ]
    material_error = bool(consequential_issues)
    if not agent_ok:
        outcome = "incorrect"
        weighted = 0.0
    elif critical_error or material_error:
        outcome = "incorrect"
    elif critical_full:
        outcome = "complete"
    elif any(row.coverage > 0 for row in actual.values()):
        outcome = "partial"
    else:
        outcome = "incorrect"

    corpus_scorable = bool(corpus_critical_ids)
    corpus_critical_full = corpus_scorable and all(
        actual[key].coverage == 1.0 and actual[key].support == "full"
        for key in corpus_critical_ids
    )
    corpus_critical_error = any(
        actual[key].support in {"unsupported", "contradicted"}
        for key in corpus_critical_ids
    )
    if not agent_ok or corpus_critical_error or material_error:
        corpus_outcome = "incorrect" if corpus_scorable else "not_scorable"
        if not agent_ok and corpus_weighted is not None:
            corpus_weighted = 0.0
    elif not corpus_scorable:
        corpus_outcome = "not_scorable"
    elif corpus_critical_full:
        corpus_outcome = "complete"
    elif any(actual[key].coverage > 0 for key in corpus_ids):
        corpus_outcome = "partial"
    else:
        corpus_outcome = "incorrect"
    return {
        "grounded_weighted_aspect_coverage": round(weighted, 6),
        "corpus_conditioned_gwac": round(corpus_weighted, 6)
        if corpus_weighted is not None and corpus_scorable
        else None,
        "corpus_conditioned_outcome": corpus_outcome,
        "corpus_scorable": corpus_scorable,
        "corpus_supported_aspect_count": len(corpus_ids),
        "corpus_unsupported_aspect_count": len(all_ids - corpus_ids),
        "corpus_supported_critical_aspect_count": len(corpus_critical_ids),
        "corpus_unsupported_critical_aspect_count": len(critical_ids - corpus_ids),
        "critical_aspects_full": critical_full,
        "critical_aspect_success": (
            sum(actual[key].coverage == 1.0 and actual[key].support == "full" for key in critical_ids)
            / max(1, len(critical_ids))
        ),
        "outcome": outcome,
        "overall_quality": int(judgment.overall_quality),
        "material_unsupported_claims": material_unsupported_claims,
        "material_contradictions": material_contradictions,
        "claim_issues": [row.model_dump() for row in judgment.claim_issues],
        "invalid_citations": judgment.invalid_citations,
        "citation_integrity": not judgment.invalid_citations,
        "aspect_scores": [row.model_dump() for row in judgment.aspects],
        "explanation": judgment.explanation,
    }


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(map(float, values))
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["arm"])].append(row)
    output = []
    for arm, values in sorted(grouped.items()):
        corpus_values = [row for row in values if row.get("corpus_scorable")]
        output.append(
            {
                "arm": arm,
                "questions": len(values),
                "corpus_scorable_questions": len(corpus_values),
                "grounded_weighted_aspect_coverage": statistics.fmean(
                    float(row["grounded_weighted_aspect_coverage"]) for row in values
                ),
                "corpus_conditioned_gwac": statistics.fmean(
                    float(row["corpus_conditioned_gwac"]) for row in corpus_values
                )
                if corpus_values
                else None,
                "critical_aspect_success": statistics.fmean(
                    float(row["critical_aspect_success"]) for row in values
                ),
                "complete_rate": statistics.fmean(row["outcome"] == "complete" for row in values),
                "partial_rate": statistics.fmean(row["outcome"] == "partial" for row in values),
                "incorrect_rate": statistics.fmean(row["outcome"] == "incorrect" for row in values),
                "corpus_conditioned_complete_rate": statistics.fmean(
                    row["corpus_conditioned_outcome"] == "complete"
                    for row in corpus_values
                )
                if corpus_values
                else None,
                "corpus_conditioned_partial_rate": statistics.fmean(
                    row["corpus_conditioned_outcome"] == "partial"
                    for row in corpus_values
                )
                if corpus_values
                else None,
                "corpus_conditioned_incorrect_rate": statistics.fmean(
                    row["corpus_conditioned_outcome"] == "incorrect"
                    for row in corpus_values
                )
                if corpus_values
                else None,
                "unsupported_claim_rate": statistics.fmean(
                    bool(row["material_unsupported_claims"] or row["material_contradictions"])
                    for row in values
                ),
                "citation_integrity_rate": statistics.fmean(bool(row["citation_integrity"]) for row in values),
                "overall_quality_1_5": statistics.fmean(float(row["overall_quality"]) for row in values),
                "agent_latency_p50_seconds": percentile(
                    (float(row["agent_latency_seconds"]) for row in values), 0.5
                ),
                "agent_tokens_per_qa": statistics.fmean(float(row["agent_total_tokens"]) for row in values),
            }
        )
    return output


def summarize_by(
    rows: list[dict[str, Any]], field: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if field == "qrel_count":
            key = "1" if int(value or 0) == 1 else "2+"
        elif isinstance(value, bool):
            key = "image" if value else "text_only"
        else:
            key = str(value or "unknown")
        grouped[key].append(row)
    return [
        {field: key, **arm_summary}
        for key in sorted(grouped)
        for arm_summary in summarize(grouped[key])
    ]


def paired_bootstrap_deltas(
    rows: list[dict[str, Any]],
    *,
    metric: str = "corpus_conditioned_gwac",
    samples: int = 10_000,
    seed: int = 20_260_831,
) -> list[dict[str, Any]]:
    """Compute deterministic paired percentile-bootstrap CIs for arm deltas."""

    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    indexed: dict[tuple[str, str], float | None] = {}
    duplicates: list[tuple[str, str]] = []
    for row in rows:
        key = (str(row["question_id"]), str(row["arm"]))
        if key in indexed:
            duplicates.append(key)
        value = row.get(metric)
        indexed[key] = float(value) if value is not None else None
    if duplicates:
        raise ValueError(f"duplicate per-question arm scores: {sorted(set(duplicates))}")
    output: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for treatment, baseline in (
        ("hybrid", "fs"),
        ("neo4j", "hybrid"),
        ("neo4j", "fs"),
    ):
        question_ids = sorted(
            question_id
            for question_id, arm in indexed
            if arm == treatment
            and indexed[(question_id, treatment)] is not None
            and indexed.get((question_id, baseline)) is not None
        )
        deltas = [
            float(indexed[(question_id, treatment)])
            - float(indexed[(question_id, baseline)])
            for question_id in question_ids
        ]
        if not deltas:
            output.append(
                {
                    "contrast": f"{treatment}_minus_{baseline}",
                    "metric": metric,
                    "paired_questions": 0,
                    "mean_delta": None,
                    "ci95": None,
                    "bootstrap_samples": samples,
                    "bootstrap_seed": seed,
                }
            )
            continue
        bootstrap_means = [
            statistics.fmean(deltas[rng.randrange(len(deltas))] for _ in deltas)
            for _ in range(samples)
        ]
        output.append(
            {
                "contrast": f"{treatment}_minus_{baseline}",
                "metric": metric,
                "paired_questions": len(deltas),
                "mean_delta": statistics.fmean(deltas),
                "ci95": {
                    "lower": percentile(bootstrap_means, 0.025),
                    "upper": percentile(bootstrap_means, 0.975),
                    "method": "paired percentile bootstrap",
                },
                "bootstrap_samples": samples,
                "bootstrap_seed": seed,
            }
        )
    return output


def _usage_totals(calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "judgments": len(calls),
        "input_tokens": sum(int(row["usage"].get("input_tokens") or 0) for row in calls),
        "cached_input_tokens": sum(
            int(row["usage"].get("cached_input_tokens") or 0) for row in calls
        ),
        "output_tokens": sum(int(row["usage"].get("output_tokens") or 0) for row in calls),
        "total_tokens": sum(int(row["usage"].get("total_tokens") or 0) for row in calls),
        "latency_seconds": sum(float(row.get("latency_seconds") or 0.0) for row in calls),
        "latency_p50_seconds": percentile(
            (float(row.get("latency_seconds") or 0.0) for row in calls), 0.5
        ),
    }


def _validate_judged_candidates(
    parsed: AspectBatchJudgment,
    *,
    aliases: dict[str, str],
    question_id: str,
) -> dict[str, CandidateAspectJudgment]:
    candidate_labels = [row.candidate for row in parsed.candidates]
    duplicate_candidates = sorted(
        {value for value in candidate_labels if candidate_labels.count(value) > 1}
    )
    if duplicate_candidates:
        raise ValueError(
            f"duplicate candidate labels for {question_id}: {duplicate_candidates}"
        )
    judged = {row.candidate: row for row in parsed.candidates}
    if set(judged) != set(aliases.values()):
        raise ValueError(f"candidate labels differ for {question_id}")
    return judged


def _judge_one_question(
    question_id: str,
    *,
    aspects: dict[str, dict[str, Any]],
    by_arm: dict[str, dict[str, dict[str, Any]]],
    corpus: dict[str, dict[str, Any]],
    rubric: dict[str, Any],
    rubric_sha256: str,
    output_dir: Path,
    client: Any,
    model: str,
    reasoning_effort: str,
    resume: bool,
) -> dict[str, Any]:
    record = aspects[question_id]
    # The arm ordering is fixed independently of CLI argument order.
    paired = {arm: by_arm[arm][question_id] for arm in ARM_ORDER}
    payload, aliases = build_judge_prompt(record, paired, corpus, rubric)
    prompt_sha256 = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cache_path = _call_cache_path(output_dir, question_id)
    cached = (
        _load_cached_call(
            cache_path,
            question_id=question_id,
            rubric_sha256=rubric_sha256,
            model=model,
            reasoning_effort=reasoning_effort,
            prompt_sha256=prompt_sha256,
        )
        if resume
        else None
    )
    if cached is None:
        parsed, meta = judge_batch(
            client,
            model=model,
            reasoning_effort=reasoning_effort,
            rubric=rubric,
            payload=payload,
        )
        resumed = False
    else:
        parsed, meta, cached_aliases = cached
        if cached_aliases != aliases:
            raise ValueError(f"cached aliases differ for {question_id}")
        resumed = True

    judged = _validate_judged_candidates(
        parsed,
        aliases=aliases,
        question_id=question_id,
    )
    question_rows: list[dict[str, Any]] = []
    for arm in ARM_ORDER:
        rollout = paired[arm]
        result = compute_candidate_score(
            record, judged[aliases[arm]], agent_ok=bool(rollout.get("agent_ok"))
        )
        question_rows.append(
            {
                "question_id": question_id,
                "project": record.get("project"),
                "split": record.get("split"),
                "arm": arm,
                "intent_category": rollout.get("intent_category"),
                "evidence_structure": rollout.get("evidence_structure"),
                "qrel_count": rollout.get("qrel_count"),
                "question_has_image": bool(rollout.get("question_has_image")),
                **result,
                "agent_ok": bool(rollout.get("agent_ok")),
                "agent_latency_seconds": float(rollout.get("latency_seconds") or 0.0),
                "agent_total_tokens": int((rollout.get("usage") or {}).get("total") or 0),
                "hit_at_10": float(rollout.get("hit_at_10") or 0.0),
                "ndcg_at_10": float(rollout.get("ndcg_at_10") or 0.0),
            }
        )

    # Do not persist malformed model output. All label/aspect validation and
    # score construction above must succeed before a new cache entry is saved.
    if not resumed:
        _write_cached_call(
            cache_path,
            question_id=question_id,
            rubric_sha256=rubric_sha256,
            model=model,
            reasoning_effort=reasoning_effort,
            prompt_sha256=prompt_sha256,
            aliases=aliases,
            judgment=parsed,
            meta=meta,
        )
    return {
        "question_id": question_id,
        "call": {
            "question_id": question_id,
            **meta,
            "aliases": aliases,
            "resumed": resumed,
        },
        "rows": question_rows,
    }


def run_judgments(
    ordered_ids: list[str],
    *,
    aspects: dict[str, dict[str, Any]],
    by_arm: dict[str, dict[str, dict[str, Any]]],
    corpus: dict[str, dict[str, Any]],
    rubric: dict[str, Any],
    rubric_sha256: str,
    output_dir: Path,
    client: Any,
    model: str,
    reasoning_effort: str,
    resume: bool,
    workers: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one paired judge call per question with bounded concurrency.

    Per-question caches may be completed in any order, but benchmark rows and
    call metadata are returned in the frozen question and arm order. Any task
    error is propagated; callers therefore cannot write a partial report.
    """

    if workers <= 0:
        raise ValueError("--workers must be positive")
    cache_paths = [_call_cache_path(output_dir, question_id) for question_id in ordered_ids]
    if len(set(cache_paths)) != len(cache_paths):
        raise ValueError("question IDs map to duplicate judge cache paths")

    completed: dict[str, dict[str, Any]] = {}
    futures: dict[Future[dict[str, Any]], str] = {}
    finished = resumed_count = fresh_count = 0
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="aspect-judge")
    try:
        for question_id in ordered_ids:
            future = executor.submit(
                _judge_one_question,
                question_id,
                aspects=aspects,
                by_arm=by_arm,
                corpus=corpus,
                rubric=rubric,
                rubric_sha256=rubric_sha256,
                output_dir=output_dir,
                client=client,
                model=model,
                reasoning_effort=reasoning_effort,
                resume=resume,
            )
            futures[future] = question_id
        for future in as_completed(futures):
            result = future.result()
            question_id = str(result["question_id"])
            completed[question_id] = result
            finished += 1
            if result["call"]["resumed"]:
                resumed_count += 1
            else:
                fresh_count += 1
            if finished % 10 == 0 or finished == len(ordered_ids):
                print(
                    "aspect judge progress: "
                    f"{finished}/{len(ordered_ids)} "
                    f"(resumed={resumed_count}, new={fresh_count})",
                    flush=True,
                )
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        print(
            "aspect judge failed closed: no report or per-query artifact was written; "
            "valid completed call-cache entries can be reused with --resume",
            flush=True,
        )
        raise
    else:
        executor.shutdown(wait=True)

    calls = [completed[question_id]["call"] for question_id in ordered_ids]
    rows = [
        row
        for question_id in ordered_ids
        for row in completed[question_id]["rows"]
    ]
    return rows, calls


def evaluate(args: argparse.Namespace, *, client: Any | None = None) -> dict[str, Any]:
    rubric, rubric_sha256 = load_judge_rubric(args.rubric)
    aspects = _unique_index(_jsonl(args.aspects), "question_id", source=args.aspects)
    corpus = _unique_index(_jsonl(args.corpus), "doc_id", source=args.corpus)
    arm_entries = list(args.arm)
    arm_names = [str(arm) for arm, _ in arm_entries]
    duplicate_arms = sorted({arm for arm in arm_names if arm_names.count(arm) > 1})
    if duplicate_arms:
        raise ValueError(f"duplicate arms: {duplicate_arms}")
    if set(arm_names) != set(ARM_ORDER):
        raise ValueError(
            f"arms must be exactly {list(ARM_ORDER)}; actual={sorted(arm_names)}"
        )

    small_trial = bool(getattr(args, "small_trial", False))
    requested_ids = [str(value) for value in getattr(args, "question_id", [])]
    duplicate_requested = sorted(
        {value for value in requested_ids if requested_ids.count(value) > 1}
    )
    if duplicate_requested:
        raise ValueError(f"duplicate explicitly requested question IDs: {duplicate_requested}")
    limit = getattr(args, "limit", None)
    if (requested_ids or limit is not None) and not small_trial:
        raise ValueError(
            "--question-id/--limit are pilot selectors and require explicit --small-trial"
        )
    missing_aspects = sorted(set(requested_ids) - set(aspects))
    if missing_aspects:
        raise ValueError(f"requested IDs lack frozen aspects: {missing_aspects}")
    ordered_ids = sorted(requested_ids or aspects)
    if limit is not None:
        if int(limit) <= 0:
            raise ValueError("--limit must be positive")
        ordered_ids = ordered_ids[: int(limit)]
    if not ordered_ids:
        raise ValueError("explicit expected question IDs cannot be empty")
    if not small_trial and len(ordered_ids) != EXPECTED_FULL_QUESTIONS:
        raise ValueError(
            f"complete matched development judge run requires exactly {EXPECTED_FULL_QUESTIONS} frozen aspects; "
            f"got {len(ordered_ids)}"
        )

    expected_set = set(ordered_ids)
    rollout_lists: dict[str, list[dict[str, Any]]] = {}
    for arm, path in arm_entries:
        loaded = load_arm_rollouts(path)
        loaded_ids = [str(row.get("id") or row.get("question_id") or "") for row in loaded]
        duplicate_loaded = sorted(
            {value for value in loaded_ids if loaded_ids.count(value) > 1}
        )
        if duplicate_loaded:
            raise ValueError(f"duplicate rollout IDs in {path}: {duplicate_loaded}")
        # A pilot may explicitly select a subset from a complete rollout artifact.
        # The selector is visible and validated; there is no implicit intersection.
        selected = (
            [row for row in loaded if str(row.get("id") or row.get("question_id")) in expected_set]
            if small_trial
            else loaded
        )
        rollout_lists[str(arm)] = selected
    validate_arm_rollouts(
        rollout_lists,
        project_root=PROJECT_ROOT,
        expected_question_ids=ordered_ids,
        small_trial=small_trial,
    )
    by_arm = {
        arm: _unique_index(
            rows,
            "id" if all(row.get("id") is not None for row in rows) else "question_id",
            source=path,
        )
        for arm, path in arm_entries
        for rows in [rollout_lists[str(arm)]]
    }

    workers = int(getattr(args, "workers", 1))
    if workers <= 0:
        raise ValueError("--workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if client is None:
        from openai import OpenAI

        client = OpenAI(timeout=args.request_timeout)
    rows, calls = run_judgments(
        ordered_ids,
        aspects=aspects,
        by_arm=by_arm,
        corpus=corpus,
        rubric=rubric,
        rubric_sha256=rubric_sha256,
        output_dir=args.output_dir,
        client=client,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        resume=bool(args.resume),
        workers=workers,
    )
    incremental_calls = [row for row in calls if not row["resumed"]]
    report = {
        "benchmark": "DSH frozen-aspect agent answer evaluation",
        "questions": len(ordered_ids),
        "judge_model": args.model,
        "judge_workers": workers,
        "rubric_version": rubric["rubric_version"],
        "rubric_sha256": rubric_sha256,
        "primary_metric": {
            "name": "Corpus-Conditioned Grounded Weighted Aspect Coverage",
            "abbreviation": "C-GWAC",
            "field": "corpus_conditioned_gwac",
            "range": "0-1",
            "formula": "GWAC over aspects with pinned local-document support; a question is scored only when at least one critical aspect has local-document support",
            "aspect_scale": [0, 0.5, 1],
        },
        "diagnostic_metric": {
            "name": "All-Aspect Grounded Weighted Aspect Coverage",
            "field": "grounded_weighted_aspect_coverage",
            "note": "Includes accepted-answer-only aspects and therefore measures the historical support-answer gap, not solely the retrieval system.",
        },
        "arms": summarize(rows),
        "paired_primary_metric_deltas": paired_bootstrap_deltas(rows),
        "by_project": summarize_by(rows, "project"),
        "by_intent_category": summarize_by(rows, "intent_category"),
        "by_evidence_structure": summarize_by(rows, "evidence_structure"),
        "by_qrel_count": summarize_by(rows, "qrel_count"),
        "by_question_image": summarize_by(rows, "question_has_image"),
        "judge_usage": {
            "incremental_api_usage": {
                **_usage_totals(incremental_calls),
                "api_calls": len(incremental_calls),
            },
            "persisted_artifact_usage": _usage_totals(calls),
            "resumed_judgments": sum(row["resumed"] for row in calls),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for name, values in (("per_query.jsonl", rows), ("judge_calls.jsonl", calls)):
        with (args.output_dir / name).open("w", encoding="utf-8") as handle:
            for row in values:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return report


def replay_artifact_report(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Rebuild a frozen answer report from retained per-question judgments.

    This mode performs no model calls. It verifies that every derived table,
    paired interval, and persisted usage total still equals the frozen report,
    and refuses to overwrite the source evidence directory.
    """

    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if source_dir == output_dir:
        raise ValueError("artifact replay output must differ from the source directory")
    source_report = json.loads((source_dir / "report.json").read_text(encoding="utf-8"))
    rows = _jsonl(source_dir / "per_query.jsonl")
    calls = _jsonl(source_dir / "judge_calls.jsonl")
    arm_names = {str(row.get("arm") or "") for row in rows}
    if arm_names != set(ARM_ORDER):
        raise ValueError(
            f"artifact rows must contain exactly {list(ARM_ORDER)}; "
            f"actual={sorted(arm_names)}"
        )
    question_ids = {str(row.get("question_id") or "") for row in rows}
    if len(rows) != len(question_ids) * len(ARM_ORDER):
        raise ValueError("artifact does not contain one row per question and arm")

    rebuilt = {
        **source_report,
        "questions": len(question_ids),
        "arms": summarize(rows),
        "paired_primary_metric_deltas": paired_bootstrap_deltas(rows),
        "by_project": summarize_by(rows, "project"),
        "by_intent_category": summarize_by(rows, "intent_category"),
        "by_evidence_structure": summarize_by(rows, "evidence_structure"),
        "by_qrel_count": summarize_by(rows, "qrel_count"),
        "by_question_image": summarize_by(rows, "question_has_image"),
        "judge_usage": {
            "incremental_api_usage": {
                **_usage_totals([row for row in calls if not row.get("resumed")]),
                "api_calls": sum(not bool(row.get("resumed")) for row in calls),
            },
            "persisted_artifact_usage": _usage_totals(calls),
            "resumed_judgments": sum(bool(row.get("resumed")) for row in calls),
        },
    }
    if rebuilt != source_report:
        changed = sorted(
            key for key in set(rebuilt) | set(source_report) if rebuilt.get(key) != source_report.get(key)
        )
        raise ValueError(
            "recomputed artifact report differs from the frozen report in: "
            + ", ".join(changed)
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(rebuilt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return rebuilt


def _arm_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("arm must be ARM=ROLLOUT_PATH")
    arm, path = value.split("=", 1)
    return arm, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aspects", type=Path)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--arm", type=_arm_arg, action="append")
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--artifact-replay-dir",
        type=Path,
        help=(
            "Rebuild a frozen report from SOURCE/{report.json,per_query.jsonl,judge_calls.jsonl} "
            "without model calls; output must be a different directory."
        ),
    )
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Maximum concurrent paired judge calls (default: 1).",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument(
        "--small-trial",
        action="store_true",
        help="Explicitly run an engineering-pilot subset selected by --limit/--question-id rather than a complete matched development run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse per-question judgments only when rubric, prompt, model, and reasoning match.",
    )
    args = parser.parse_args()
    if args.artifact_replay_dir is not None:
        if args.aspects is not None or args.corpus is not None or args.arm:
            raise SystemExit(
                "--artifact-replay-dir cannot be combined with --aspects, --corpus, or --arm"
            )
        print(
            json.dumps(
                replay_artifact_report(args.artifact_replay_dir, args.output_dir),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.aspects is None or args.corpus is None or not args.arm:
        raise SystemExit("live evaluation requires --aspects, --corpus, and three --arm values")
    if len({arm for arm, _ in args.arm}) != len(args.arm):
        raise SystemExit("duplicate --arm values")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if not load_openai_key_from_configured_env():
        raise SystemExit("OPENAI_API_KEY is unset; set it or configure KBBENCH_OPENAI_ENV_FILE")
    print(json.dumps(evaluate(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
