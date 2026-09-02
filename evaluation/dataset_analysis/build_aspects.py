"""Build frozen weighted answer aspects from normalized local DocsQA records.

The constructor and reviewer use one versioned, domain-neutral rubric.  The
output is question-specific annotation data; it is never regenerated from an
agent answer during evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from dsh_plugin.agent_eval.credentials import load_openai_key_from_configured_env
from .question_partitions import PARTITIONS, load_question_partition


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_RUNS = PROJECT_ROOT / "results/runs/dataset-analysis"
DEFAULT_RUBRIC = Path(__file__).resolve().parent / "rubrics/aspect_evaluation_v2.json"
FORBIDDEN_OVERRIDE_FIELDS = {
    "question_overrides",
    "project_overrides",
    "dataset_overrides",
    "case_overrides",
    "expected_labels_by_id",
}


class AnswerAspect(BaseModel):
    aspect_id: str
    description: str = Field(min_length=1)
    requirement_ids: list[str] = Field(min_length=1)
    claim_ids: list[str] = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    importance: int = Field(ge=1, le=5)
    critical: bool
    rationale: str = Field(min_length=1)


class AspectPlan(BaseModel):
    aspects: list[AnswerAspect] = Field(min_length=1)
    excluded_claim_ids: list[str]
    coverage_summary: str


class AspectReview(BaseModel):
    decision: Literal["accept", "revise", "reject"]
    aspects: list[AnswerAspect]
    excluded_claim_ids: list[str]
    issues: list[str]
    coverage_summary: str


def load_rubric(path: Path = DEFAULT_RUBRIC) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    rubric = json.loads(raw)
    required = {
        "schema_version",
        "rubric_version",
        "constructor_system",
        "constructor_rules",
        "reviewer_system",
        "reviewer_rules",
        "candidate_system",
        "candidate_rules",
        "candidate_reviewer_system",
        "candidate_reviewer_rules",
        "judge_system",
        "judge_rules",
    }
    missing = sorted(required - set(rubric))
    if missing:
        raise ValueError(f"aspect rubric is missing required fields: {missing}")
    present = sorted(FORBIDDEN_OVERRIDE_FIELDS & set(rubric))
    if present:
        raise ValueError(f"aspect rubric contains forbidden override fields: {present}")
    if int(rubric["schema_version"]) != 1:
        raise ValueError("unsupported aspect rubric schema version")
    return rubric, hashlib.sha256(raw).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _value_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_source_rows(
    source_rows: list[dict[str, Any]],
    *,
    split: str,
    question_ids: list[str],
    limit: int | None,
    question_manifest: Path | None,
    manifest_partition: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select accepted normalization rows under one auditable selection mode."""

    selected = [row for row in source_rows if row.get("status") == "accepted"]
    available_ids = [str(row.get("question_id") or "") for row in selected]
    if not all(available_ids) or len(available_ids) != len(set(available_ids)):
        raise ValueError(
            "accepted normalization rows must have unique, non-empty question_id values"
        )
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")

    selection_metadata: dict[str, Any]
    if question_manifest is not None:
        if split != "all" or question_ids or limit is not None:
            raise ValueError(
                "--question-manifest cannot be combined with --split, --question-id, or --limit"
            )
        wanted, selection_metadata = load_question_partition(
            question_manifest,
            available_question_ids=available_ids,
            partition=manifest_partition,
            allow_all=True,
        )
        selected = [
            row for row in selected if str(row.get("question_id")) in wanted
        ]
    else:
        if manifest_partition != "all":
            raise ValueError("--manifest-partition requires --question-manifest")
        if split != "all":
            selected = [row for row in selected if str(row.get("split")) == split]
        if question_ids:
            wanted = set(question_ids)
            selected = [
                row for row in selected if str(row.get("question_id")) in wanted
            ]
        selection_metadata = {
            "selection_mode": "legacy",
            "legacy_split": split,
            "legacy_question_ids": sorted(map(str, question_ids)),
            "legacy_limit": limit,
        }

    selected.sort(key=lambda row: str(row["question_id"]))
    if limit is not None:
        selected = selected[:limit]
    selected_ids = [str(row["question_id"]) for row in selected]
    selection_metadata.update(
        {
            "selected_question_count": len(selected_ids),
            "selected_question_ids_sha256": _value_sha256(selected_ids),
        }
    )
    return selected, selection_metadata


def _usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "cached_input_tokens": int(getattr(input_details, "cached_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "reasoning_tokens": int(getattr(output_details, "reasoning_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _call_parse(
    client: Any,
    *,
    model: str,
    system: str,
    payload: dict[str, Any],
    schema: type[BaseModel],
    reasoning_effort: str,
    attempts: int,
) -> tuple[BaseModel, dict[str, Any]]:
    error: Exception | None = None
    prompt_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    for attempt in range(attempts):
        started = time.perf_counter()
        try:
            response = client.responses.parse(
                model=model,
                instructions=system,
                input=[{"role": "user", "content": prompt_text}],
                text_format=schema,
                reasoning={"effort": reasoning_effort},
                store=False,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise ValueError("model returned no structured output")
            return parsed, {
                "latency_seconds": round(time.perf_counter() - started, 6),
                "usage": _usage(response),
                "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
            }
        except Exception as exc:  # pragma: no cover - network failures vary
            error = exc
            if "credit_balance_exhausted" in str(exc) or "insufficient_quota" in str(exc):
                break
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    assert error is not None
    raise error


def _trim_evidence(rows: list[dict[str, Any]], max_total: int = 45_000) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    remaining = max_total
    for row in rows:
        if remaining <= 0:
            break
        text = str(row.get("text") or "")
        limit = min(6_000, remaining)
        record = {
            key: row.get(key)
            for key in ("evidence_id", "kind", "doc_id", "heading", "local_path", "constraint")
            if row.get(key) is not None
        }
        record["text"] = text[:limit]
        output.append(record)
        remaining -= len(record["text"])
    return output


def annotation_context(work_row: dict[str, Any]) -> dict[str, Any]:
    question = dict(work_row["normalized_question"])
    package = dict(work_row.get("evidence_package") or {})
    requirements = list((question.get("user_requirements") or {}).get("requirements") or [])
    claims = list(question.get("normalized_claims") or [])
    evidence = _trim_evidence(list(package.get("evidence") or []))
    return {
        "question_id": str(work_row["question_id"]),
        "project": str(work_row.get("project") or question.get("dataset") or ""),
        "split": str(work_row.get("split") or question.get("split") or ""),
        "question": str(question.get("query") or package.get("question") or ""),
        "reference_answer": str(question.get("normalized_answer") or ""),
        "requirements": requirements,
        "claims": claims,
        "evidence": evidence,
        "source_metadata": {
            "intent_category": question.get("intent_category"),
            "evidence_category": question.get("evidence_category"),
            "evidence_structure": question.get("evidence_structure"),
            "question_has_image": bool(question.get("question_images")),
        },
    }


def validate_aspects(
    aspects: list[dict[str, Any]], context: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    requirement_by_id = {
        str(row["requirement_id"]): row for row in context.get("requirements") or []
    }
    claim_ids = {str(row["claim_id"]) for row in context.get("claims") or []}
    evidence_ids = {str(row["evidence_id"]) for row in context.get("evidence") or []}
    if not aspects:
        return ["no aspects"]
    descriptions: set[str] = set()
    for index, aspect in enumerate(aspects, start=1):
        prefix = f"aspect[{index}]"
        description = " ".join(str(aspect.get("description") or "").casefold().split())
        if not description:
            errors.append(f"{prefix} has an empty description")
        elif description in descriptions:
            errors.append(f"{prefix} duplicates another description")
        descriptions.add(description)
        unknown_requirements = sorted(
            set(map(str, aspect.get("requirement_ids") or [])) - set(requirement_by_id)
        )
        unknown_claims = sorted(set(map(str, aspect.get("claim_ids") or [])) - claim_ids)
        unknown_evidence = sorted(set(map(str, aspect.get("evidence_ids") or [])) - evidence_ids)
        if unknown_requirements:
            errors.append(f"{prefix} has unknown requirements: {unknown_requirements}")
        if unknown_claims:
            errors.append(f"{prefix} has unknown claims: {unknown_claims}")
        if unknown_evidence:
            errors.append(f"{prefix} has unknown evidence: {unknown_evidence}")
        if not aspect.get("requirement_ids"):
            errors.append(f"{prefix} has no requirement mapping")
        if not aspect.get("claim_ids"):
            errors.append(f"{prefix} has no claim mapping")
        if not aspect.get("evidence_ids"):
            errors.append(f"{prefix} has no evidence mapping")
        importance = int(aspect.get("importance") or 0)
        if not 1 <= importance <= 5:
            errors.append(f"{prefix} importance is outside 1..5")
    critical_requirements = {
        req_id for req_id, row in requirement_by_id.items() if bool(row.get("critical"))
    }
    covered_critical = {
        str(req_id)
        for aspect in aspects
        if bool(aspect.get("critical"))
        for req_id in aspect.get("requirement_ids") or []
    }
    missing = sorted(critical_requirements - covered_critical)
    if missing:
        errors.append(f"critical requirements lack a critical aspect: {missing}")
    return errors


def _canonicalize_aspects(
    aspects: list[dict[str, Any]], context: dict[str, Any]
) -> list[dict[str, Any]]:
    evidence_by_id = {
        str(row["evidence_id"]): row for row in context.get("evidence") or []
    }
    output: list[dict[str, Any]] = []
    for index, source in enumerate(aspects, start=1):
        row = dict(source)
        row["aspect_id"] = f"a{index}"
        row["requirement_ids"] = list(dict.fromkeys(map(str, row.get("requirement_ids") or [])))
        row["claim_ids"] = list(dict.fromkeys(map(str, row.get("claim_ids") or [])))
        row["evidence_ids"] = list(dict.fromkeys(map(str, row.get("evidence_ids") or [])))
        row["weight"] = 0.0
        evidence = [evidence_by_id[value] for value in row["evidence_ids"] if value in evidence_by_id]
        row["evidence"] = evidence
        row["retrieval_doc_ids"] = sorted(
            {str(item["doc_id"]) for item in evidence if item.get("doc_id")}
        )
        output.append(row)
    total = sum(int(row["importance"]) for row in output) or 1
    for row in output:
        row["weight"] = round(int(row["importance"]) / total, 8)
    return output


def construct_one(
    client: Any,
    *,
    work_row: dict[str, Any],
    rubric: dict[str, Any],
    rubric_sha256: str,
    model: str,
    reasoning_effort: str,
    attempts: int,
    max_review_repairs: int,
) -> dict[str, Any]:
    context = annotation_context(work_row)
    base = {
        "question_id": context["question_id"],
        "project": context["project"],
        "split": context["split"],
        "rubric_version": rubric["rubric_version"],
        "rubric_sha256": rubric_sha256,
    }
    constructor_payload = {
        "rubric_version": rubric["rubric_version"],
        "rubric_sha256": rubric_sha256,
        "rules": rubric["constructor_rules"],
        "record": context,
    }
    try:
        proposal, constructor_meta = _call_parse(
            client,
            model=model,
            system=str(rubric["constructor_system"]),
            payload=constructor_payload,
            schema=AspectPlan,
            reasoning_effort=reasoning_effort,
            attempts=attempts,
        )
        proposed = proposal.model_dump()
        review_history: list[dict[str, Any]] = []
        review_input = proposed
        validation_errors: list[str] = []
        final_review: dict[str, Any] | None = None
        for review_round in range(max_review_repairs + 1):
            reviewer_payload = {
                "rubric_version": rubric["rubric_version"],
                "rubric_sha256": rubric_sha256,
                "rules": rubric["reviewer_rules"],
                "record": context,
                "proposed_schema": review_input,
                "deterministic_validation_errors": validation_errors,
            }
            reviewed, reviewer_meta = _call_parse(
                client,
                model=model,
                system=str(rubric["reviewer_system"]),
                payload=reviewer_payload,
                schema=AspectReview,
                reasoning_effort=reasoning_effort,
                attempts=attempts,
            )
            candidate = reviewed.model_dump()
            candidate_aspects = [row.model_dump() for row in reviewed.aspects]
            validation_errors = validate_aspects(candidate_aspects, context)
            review_history.append(
                {
                    "round": review_round,
                    "review": candidate,
                    "validation_errors": validation_errors,
                    "call": reviewer_meta,
                }
            )
            final_review = candidate
            if candidate["decision"] != "reject" and not validation_errors:
                aspects = _canonicalize_aspects(candidate_aspects, context)
                doc_ids = sorted(
                    {doc_id for aspect in aspects for doc_id in aspect["retrieval_doc_ids"]}
                )
                return {
                    **base,
                    "status": "accepted",
                    "question": context["question"],
                    "reference_answer": context["reference_answer"],
                    "requirements": context["requirements"],
                    "claims": context["claims"],
                    "aspects": aspects,
                    "aspect_count": len(aspects),
                    "critical_aspect_count": sum(bool(row["critical"]) for row in aspects),
                    "retrieval_doc_ids": doc_ids,
                    "retrieval_eligible_aspect_count": sum(bool(row["retrieval_doc_ids"]) for row in aspects),
                    "source_metadata": context["source_metadata"],
                    "excluded_claim_ids": candidate.get("excluded_claim_ids") or [],
                    "coverage_summary": candidate.get("coverage_summary") or "",
                    "constructor": {"proposal": proposed, "call": constructor_meta},
                    "reviews": review_history,
                }
            review_input = candidate
        return {
            **base,
            "status": "rejected",
            "rejection_reason": "aspect_review_not_accepted",
            "validation_errors": validation_errors,
            "constructor": {"proposal": proposed, "call": constructor_meta},
            "reviews": review_history,
            "last_review": final_review,
        }
    except Exception as error:  # pragma: no cover - exercised by live failures
        return {
            **base,
            "status": "error",
            "rejection_reason": "aspect_pipeline_error",
            "error": f"{type(error).__name__}: {error}",
        }


def _aggregate_usage(rows: list[dict[str, Any]]) -> dict[str, int]:
    calls: list[dict[str, int]] = []
    for row in rows:
        constructor = (row.get("constructor") or {}).get("call") or {}
        if constructor.get("usage"):
            calls.append(constructor["usage"])
        for review in row.get("reviews") or []:
            call = review.get("call") or {}
            if call.get("usage"):
                calls.append(call["usage"])
    return {
        "calls": len(calls),
        "input_tokens": sum(row.get("input_tokens", 0) for row in calls),
        "cached_input_tokens": sum(row.get("cached_input_tokens", 0) for row in calls),
        "output_tokens": sum(row.get("output_tokens", 0) for row in calls),
        "reasoning_tokens": sum(row.get("reasoning_tokens", 0) for row in calls),
        "total_tokens": sum(row.get("total_tokens", 0) for row in calls),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _resumable_results(path: Path) -> dict[str, dict[str, Any]]:
    """Load terminal rows while leaving transient API errors eligible for retry."""

    completed: dict[str, dict[str, Any]] = {}
    for row in _jsonl(path):
        question_id = str(row.get("question_id") or "")
        if not question_id:
            raise ValueError(f"aspect result in {path} lacks question_id")
        if row.get("status") == "error":
            completed.pop(question_id, None)
        else:
            completed[question_id] = row
    return completed


def _report(rows: list[dict[str, Any]], rubric: dict[str, Any], rubric_sha256: str) -> dict[str, Any]:
    accepted = [row for row in rows if row.get("status") == "accepted"]
    counts = Counter(str(row.get("status")) for row in rows)
    aspect_counts = [int(row["aspect_count"]) for row in accepted]
    project_counts = Counter(str(row.get("project")) for row in accepted)
    return {
        "rubric_version": rubric["rubric_version"],
        "rubric_sha256": rubric_sha256,
        "records": len(rows),
        "accepted": len(accepted),
        "acceptance_rate": len(accepted) / max(1, len(rows)),
        "status_counts": dict(sorted(counts.items())),
        "accepted_by_project": dict(sorted(project_counts.items())),
        "mean_aspects": statistics.fmean(aspect_counts) if aspect_counts else 0.0,
        "median_aspects": statistics.median(aspect_counts) if aspect_counts else 0.0,
        "questions_with_multiple_aspects": sum(value > 1 for value in aspect_counts),
        "questions_with_retrieval_ineligible_aspects": sum(
            int(row["retrieval_eligible_aspect_count"]) < int(row["aspect_count"])
            for row in accepted
        ),
        "model_usage": _aggregate_usage(rows),
    }


def render_report(report: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Frozen DocsQA aspect construction",
            "",
            f"- Rubric: `{report['rubric_version']}`",
            f"- Records: {report['records']}",
            f"- Accepted: {report['accepted']} ({report['acceptance_rate']:.1%})",
            f"- Mean aspects: {report['mean_aspects']:.2f}",
            f"- Median aspects: {report['median_aspects']:.1f}",
            f"- Multi-aspect questions: {report['questions_with_multiple_aspects']}",
            f"- Questions with at least one answer-only aspect: {report['questions_with_retrieval_ineligible_aspects']}",
            f"- Model calls: {report['model_usage']['calls']}",
            f"- Model tokens: {report['model_usage']['total_tokens']:,}",
            "",
            "The aspect content is question-specific. The construction and review rules are one domain-neutral, hash-frozen rubric with no per-case override mechanism.",
            "",
        ]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    from openai import OpenAI

    rubric, rubric_sha256 = load_rubric(args.rubric)
    source_rows = _jsonl(args.normalization_work)
    selected, selection_metadata = select_source_rows(
        source_rows,
        split=args.split,
        question_ids=args.question_id,
        limit=args.limit,
        question_manifest=args.question_manifest,
        manifest_partition=args.manifest_partition,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        rubric_path = args.rubric.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        rubric_path = args.rubric.resolve().as_posix()
    contract = {
        "schema_version": 2,
        "normalization_work": str(args.normalization_work),
        "normalization_work_sha256": _sha256(args.normalization_work),
        "rubric_path": rubric_path,
        "rubric_version": rubric["rubric_version"],
        "rubric_sha256": rubric_sha256,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "max_review_repairs": args.max_review_repairs,
        "selection": selection_metadata,
    }
    contract_path = args.output_dir / "run_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text(encoding="utf-8")) != contract:
        raise ValueError("aspect run contract differs; use a new output directory")
    contract_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    completed: dict[str, dict[str, Any]] = {}
    result_path = args.output_dir / "per_question.jsonl"
    if args.resume and result_path.exists():
        completed = _resumable_results(result_path)
    pending = [row for row in selected if str(row["question_id"]) not in completed]
    client = OpenAI(timeout=args.request_timeout)
    lock = threading.Lock()
    with result_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    construct_one,
                    client,
                    work_row=row,
                    rubric=rubric,
                    rubric_sha256=rubric_sha256,
                    model=args.model,
                    reasoning_effort=args.reasoning_effort,
                    attempts=args.attempts,
                    max_review_repairs=args.max_review_repairs,
                ): str(row["question_id"])
                for row in pending
            }
            for future in as_completed(futures):
                question_id = futures[future]
                result = future.result()
                completed[question_id] = result
                with lock:
                    handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
                    handle.flush()
                    done = len(completed)
                    if done == len(selected) or done % 20 == 0:
                        print(f"aspect progress: {done}/{len(selected)}", flush=True)
    ordered = [completed[str(row["question_id"])] for row in selected]
    _write_jsonl(result_path, ordered)
    accepted = [row for row in ordered if row.get("status") == "accepted"]
    _write_jsonl(args.output_dir / "aspects.jsonl", accepted)
    _write_jsonl(
        args.output_dir / "rejected.jsonl",
        [row for row in ordered if row.get("status") != "accepted"],
    )
    report = _report(ordered, rubric, rubric_sha256)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(render_report(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--normalization-work",
        type=Path,
        default=ANALYSIS_RUNS / "normalization-v14/per_question.jsonl",
    )
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC)
    parser.add_argument(
        "--output-dir", type=Path, default=ANALYSIS_RUNS / "aspects-v2"
    )
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-review-repairs", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--split", choices=("all", "train", "validation", "test"), default="all")
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--question-manifest", type=Path)
    parser.add_argument(
        "--manifest-partition",
        choices=("all", *PARTITIONS),
        default="all",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.attempts < 1 or args.max_review_repairs < 0:
        raise SystemExit("workers and attempts must be positive; repairs must be non-negative")
    if not load_openai_key_from_configured_env():
        raise SystemExit("OPENAI_API_KEY is unset; set it or configure KBBENCH_OPENAI_ENV_FILE")
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
