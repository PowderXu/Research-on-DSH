"""Evaluate a frozen answer judge under the benchmark's weak supervision.

Platform-selected answers are treated as noisy positive references, not human
gold labels.  Controlled partial and incorrect answers prevent an always-pass
judge from satisfying the positive-reference assumption.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from dsh_plugin.agent_eval.aspect_judge import (
    build_judge_prompt,
    compute_candidate_score,
    judge_batch,
    load_judge_rubric,
)
from dsh_plugin.agent_eval.credentials import load_openai_key_from_configured_env

from .question_partitions import PARTITIONS, load_question_partition


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEAK_SUPERVISION_ASSUMPTION = (
    "For most retained questions, the platform-selected answer combined with "
    "its resolved internal documentation is sufficiently complete and correct "
    "to resolve the question."
)
DEFAULT_MIN_SOURCE_ANSWER_COMPLETE_RECALL = 0.80
DEFAULT_MIN_CONTROL_RECALL = 0.80
DEFAULT_MAX_INCORRECT_FALSE_COMPLETE_RATE = 0.05
DEFAULT_MIN_MACRO_F1 = 0.80
DEFAULT_MIN_BALANCED_CASE_RATE = 0.98
CASE_GENERATION_RUBRIC_FIELDS = (
    "candidate_system",
    "candidate_rules",
    "candidate_reviewer_system",
    "candidate_reviewer_rules",
)


class GeneratedVariant(BaseModel):
    requested_label: Literal["partial", "incorrect"]
    answer: str = Field(min_length=1)
    target_aspect_ids: list[str] = Field(min_length=1)
    transformation_summary: str


class CalibrationVariants(BaseModel):
    variants: list[GeneratedVariant] = Field(min_length=2, max_length=2)


class CalibrationAspectScore(BaseModel):
    aspect_id: str
    coverage: Literal[0.0, 0.5, 1.0]


class CandidateLabel(BaseModel):
    candidate: Literal["complete", "partial", "incorrect"]
    actual_label: Literal["complete", "partial", "incorrect"]
    valid_for_calibration: bool
    aspect_coverage: list[CalibrationAspectScore]
    decisive_errors: list[str]
    explanation: str


class CalibrationReview(BaseModel):
    candidates: list[CandidateLabel] = Field(min_length=3, max_length=3)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _value_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_case_generation_rubric(rubric: dict[str, Any]) -> None:
    missing = [field for field in CASE_GENERATION_RUBRIC_FIELDS if field not in rubric]
    if missing:
        raise ValueError(
            "rubric cannot generate calibration cases; missing "
            f"{missing}. Supply --reuse-cases-from for a judge-only rubric."
        )


def load_reused_cases(
    source: Path,
    records: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], Path, str]:
    """Load an exact, fixed calibration case set for a judge-prompt ablation."""

    case_path = source / "calibration_cases.jsonl" if source.is_dir() else source
    rows = _jsonl(case_path)
    cases: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        question_id = str(row.get("question_id") or "")
        if not question_id:
            raise ValueError(f"reused calibration case in {case_path} lacks question_id")
        if question_id in cases:
            duplicates.append(question_id)
        cases[question_id] = row
    if duplicates:
        raise ValueError(
            f"duplicate reused calibration question IDs in {case_path}: "
            f"{sorted(set(duplicates))}"
        )
    expected = {str(row["question_id"]) for row in records}
    actual = set(cases)
    if actual != expected:
        raise ValueError(
            "reused calibration cases do not exactly match selected aspect records: "
            f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
        )
    return cases, case_path, _sha256(case_path)


def load_seed_cases(
    source: Path,
    records: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], Path, str, dict[str, Any]]:
    """Load accepted fixed cases while leaving failed cases to regenerate.

    The source is still validated as a whole: duplicate IDs and IDs outside the
    selected question set are errors even when the affected row is rejected or
    errored.  Only accepted triples are safe to reuse as completed calibration
    cases; every other status remains pending in the new run.
    """

    case_path = source / "calibration_cases.jsonl" if source.is_dir() else source
    rows = _jsonl(case_path)
    source_rows: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        question_id = str(row.get("question_id") or "")
        if not question_id:
            raise ValueError(f"seed calibration case in {case_path} lacks question_id")
        if question_id in source_rows:
            duplicates.append(question_id)
        source_rows[question_id] = row
    if duplicates:
        raise ValueError(
            f"duplicate seed calibration question IDs in {case_path}: "
            f"{sorted(set(duplicates))}"
        )
    expected = {str(row["question_id"]) for row in records}
    unknown = set(source_rows) - expected
    if unknown:
        raise ValueError(
            "seed calibration cases contain question IDs outside the selected "
            f"partition: {sorted(unknown)}"
        )
    cases = {
        question_id: row
        for question_id, row in source_rows.items()
        if str(row.get("status") or "") == "accepted"
    }
    accepted_ids = sorted(cases)
    status_counts = Counter(
        str(row.get("status") or "missing") for row in source_rows.values()
    )
    provenance = {
        "source_row_count": len(rows),
        "source_status_counts": dict(sorted(status_counts.items())),
        "accepted_seeded_question_count": len(accepted_ids),
        "accepted_seeded_question_ids_sha256": _value_sha256(accepted_ids),
    }
    return cases, case_path, _sha256(case_path), provenance


def _resumable_calibration_cases(path: Path) -> dict[str, dict[str, Any]]:
    """Load completed cases while allowing transient generation errors to retry."""

    cases: dict[str, dict[str, Any]] = {}
    for row in _jsonl(path):
        question_id = str(row.get("question_id") or "")
        if not question_id:
            raise ValueError(f"calibration case in {path} lacks question_id")
        if row.get("status") == "error":
            cases.pop(question_id, None)
        else:
            cases[question_id] = row
    return cases


def _usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    details = getattr(usage, "input_tokens_details", None)
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "cached_input_tokens": int(getattr(details, "cached_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _call(
    client: Any,
    *,
    model: str,
    reasoning_effort: str,
    system: str,
    payload: dict[str, Any],
    schema: type[BaseModel],
    attempts: int,
) -> tuple[BaseModel, dict[str, Any]]:
    prompt = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    error: Exception | None = None
    for attempt in range(attempts):
        started = time.perf_counter()
        try:
            response = client.responses.parse(
                model=model,
                instructions=system,
                input=[{"role": "user", "content": prompt}],
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
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
        except Exception as exc:  # pragma: no cover - live API failures vary
            error = exc
            if "credit_balance_exhausted" in str(exc) or "insufficient_quota" in str(exc):
                break
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    assert error is not None
    raise error


def _gold_package(record: dict[str, Any]) -> dict[str, Any]:
    evidence: dict[str, dict[str, Any]] = {}
    aspects = []
    for row in record.get("aspects") or []:
        aspects.append(
            {
                key: row.get(key)
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
        for item in row.get("evidence") or []:
            evidence_id = str(item.get("evidence_id") or "")
            if evidence_id:
                evidence[evidence_id] = item
    return {
        "question_id": record["question_id"],
        "question": record["question"],
        "reference_answer": record["reference_answer"],
        "aspects": aspects,
        "evidence": list(evidence.values()),
    }


def _variants_by_label(parsed: CalibrationVariants) -> dict[str, GeneratedVariant]:
    output = {row.requested_label: row for row in parsed.variants}
    if set(output) != {"partial", "incorrect"}:
        raise ValueError("variant generator must return exactly partial and incorrect")
    return output


def _review_by_candidate(parsed: CalibrationReview) -> dict[str, CandidateLabel]:
    output = {row.candidate: row for row in parsed.candidates}
    if set(output) != {"complete", "partial", "incorrect"}:
        raise ValueError("candidate reviewer must return exactly three candidate names")
    return output


def create_calibration_case(
    client: Any,
    *,
    record: dict[str, Any],
    rubric: dict[str, Any],
    rubric_sha256: str,
    model: str,
    reasoning_effort: str,
    attempts: int,
    max_variant_repairs: int,
) -> dict[str, Any]:
    package = _gold_package(record)
    feedback: list[str] = []
    history: list[dict[str, Any]] = []
    try:
        for round_index in range(max_variant_repairs + 1):
            generated, generation_meta = _call(
                client,
                model=model,
                reasoning_effort=reasoning_effort,
                system=str(rubric["candidate_system"]),
                payload={
                    "rubric_version": rubric["rubric_version"],
                    "rubric_sha256": rubric_sha256,
                    "rules": rubric["candidate_rules"],
                    "gold": package,
                    "review_feedback": feedback,
                },
                schema=CalibrationVariants,
                attempts=attempts,
            )
            variants = _variants_by_label(generated)
            answers = {
                "complete": package["reference_answer"],
                "partial": variants["partial"].answer,
                "incorrect": variants["incorrect"].answer,
            }
            reviewed, review_meta = _call(
                client,
                model=model,
                reasoning_effort=reasoning_effort,
                system=str(rubric["candidate_reviewer_system"]),
                payload={
                    "rubric_version": rubric["rubric_version"],
                    "rubric_sha256": rubric_sha256,
                    "rules": rubric["candidate_reviewer_rules"],
                    "gold": package,
                    "candidates": answers,
                },
                schema=CalibrationReview,
                attempts=attempts,
            )
            labels = _review_by_candidate(reviewed)
            valid = all(
                labels[name].valid_for_calibration and labels[name].actual_label == name
                for name in ("complete", "partial", "incorrect")
            )
            history.append(
                {
                    "round": round_index,
                    "variants": generated.model_dump(),
                    "review": reviewed.model_dump(),
                    "generation_call": generation_meta,
                    "review_call": review_meta,
                }
            )
            if valid:
                return {
                    "question_id": record["question_id"],
                    "project": record.get("project"),
                    "split": record.get("split"),
                    "status": "accepted",
                    "answers": answers,
                    "expected_labels": {name: labels[name].actual_label for name in labels},
                    "target_aspects": {
                        name: variants[name].target_aspect_ids for name in ("partial", "incorrect")
                    },
                    "history": history,
                }
            feedback = [
                f"{name}: expected {name}, reviewer found {labels[name].actual_label}; "
                f"{labels[name].explanation}"
                for name in ("complete", "partial", "incorrect")
                if not labels[name].valid_for_calibration or labels[name].actual_label != name
            ]
        return {
            "question_id": record["question_id"],
            "project": record.get("project"),
            "split": record.get("split"),
            "status": "rejected",
            "rejection_reason": "balanced_variants_not_validated",
            "history": history,
        }
    except Exception as error:  # pragma: no cover - live failures vary
        return {
            "question_id": record["question_id"],
            "project": record.get("project"),
            "split": record.get("split"),
            "status": "error",
            "rejection_reason": "calibration_case_error",
            "error": f"{type(error).__name__}: {error}",
            "history": history,
        }


def grade_calibration_case(
    client: Any,
    *,
    record: dict[str, Any],
    case: dict[str, Any],
    rubric: dict[str, Any],
    model: str,
    reasoning_effort: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = {
        name: {"answer": answer, "ranked_ids": [], "agent_ok": True}
        for name, answer in case["answers"].items()
    }
    payload, aliases = build_judge_prompt(record, candidates, {}, rubric)
    parsed, meta = judge_batch(
        client,
        model=model,
        reasoning_effort=reasoning_effort,
        rubric=rubric,
        payload=payload,
    )
    judgments = {row.candidate: row for row in parsed.candidates}
    if set(judgments) != set(aliases.values()):
        raise ValueError("calibration judge candidate labels differ")
    rows = []
    for name in ("complete", "partial", "incorrect"):
        score = compute_candidate_score(record, judgments[aliases[name]])
        rows.append(
            {
                "question_id": record["question_id"],
                "project": record.get("project"),
                "split": record.get("split"),
                "candidate": name,
                "expected_label": case["expected_labels"][name],
                "predicted_label": score["outcome"],
                **score,
            }
        )
    return rows, {"question_id": record["question_id"], "aliases": aliases, **meta}


def classification_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ("complete", "partial", "incorrect")
    confusion = {
        expected: {predicted: 0 for predicted in labels} for expected in labels
    }
    for row in rows:
        confusion[str(row["expected_label"])][str(row["predicted_label"])] += 1
    per_class: dict[str, dict[str, float]] = {}
    for label in labels:
        true_positive = confusion[label][label]
        actual = sum(confusion[label].values())
        predicted = sum(confusion[expected][label] for expected in labels)
        precision = true_positive / max(1, predicted)
        recall = true_positive / max(1, actual)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "recall_95ci": wilson_interval(true_positive, actual),
            "f1": f1,
        }
    return {
        "examples": len(rows),
        "accuracy": sum(row["expected_label"] == row["predicted_label"] for row in rows)
        / max(1, len(rows)),
        "macro_f1": statistics.fmean(per_class[label]["f1"] for label in labels),
        "complete_recall": per_class["complete"]["recall"],
        "partial_recall": per_class["partial"]["recall"],
        "incorrect_recall": per_class["incorrect"]["recall"],
        "incorrect_false_complete_rate": confusion["incorrect"]["complete"]
        / max(1, sum(confusion["incorrect"].values())),
        "per_class": per_class,
        "confusion": confusion,
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""

    if total <= 0:
        return [0.0, 0.0]
    proportion = successes / total
    z_squared = z * z
    denominator = 1 + z_squared / total
    center = (proportion + z_squared / (2 * total)) / denominator
    radius = (
        z
        * (
            proportion * (1 - proportion) / total
            + z_squared / (4 * total * total)
        )
        ** 0.5
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _sample_records(records: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    if not limit or limit >= len(records):
        return sorted(records, key=lambda row: str(row["question_id"]))
    rng = random.Random(seed)
    by_project: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_project[str(row.get("project") or "")].append(row)
    selected: list[dict[str, Any]] = []
    projects = sorted(by_project)
    while len(selected) < limit and any(by_project.values()):
        for project in projects:
            if len(selected) >= limit:
                break
            pool = by_project[project]
            if pool:
                selected.append(pool.pop(rng.randrange(len(pool))))
    return sorted(selected, key=lambda row: str(row["question_id"]))


def _validate_question_selection_args(args: argparse.Namespace) -> None:
    """Validate the orthogonal weak-supervision selection interface."""

    manifest = getattr(args, "question_manifest", None)
    partition = getattr(args, "manifest_partition", None)
    legacy_splits = list(getattr(args, "split", []) or [])
    limit = getattr(args, "limit", None)
    if (
        getattr(args, "reuse_cases_from", None) is not None
        and getattr(args, "seed_cases_from", None) is not None
    ):
        raise ValueError("--reuse-cases-from and --seed-cases-from are mutually exclusive")
    if manifest is None and partition is not None:
        raise ValueError("--manifest-partition requires --question-manifest")
    if manifest is not None and partition is None:
        raise ValueError("--question-manifest requires --manifest-partition")
    if manifest is None:
        return
    if legacy_splits:
        raise ValueError("--question-manifest cannot be combined with --split")
    if limit is not None:
        raise ValueError("--question-manifest cannot be combined with --limit")
    if partition == "final_test" and getattr(args, "rubric_role", None) != "frozen":
        raise ValueError("final_test requires --rubric-role frozen")


def _select_records(
    args: argparse.Namespace,
    all_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select aspect rows and return immutable selection provenance."""

    _validate_question_selection_args(args)
    manifest = getattr(args, "question_manifest", None)
    if manifest is not None:
        question_ids = [str(row["question_id"]) for row in all_records]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("aspect records contain duplicate question IDs")
        wanted, selection_metadata = load_question_partition(
            Path(manifest),
            available_question_ids=question_ids,
            partition=str(getattr(args, "manifest_partition")),
            allow_all=False,
        )
        records = [
            row for row in all_records if str(row["question_id"]) in wanted
        ]
        records = _sample_records(records, None, int(getattr(args, "seed", 42)))
    else:
        splits = list(getattr(args, "split", []) or [])
        records = [
            row for row in all_records if str(row.get("split")) in set(splits)
        ]
        limit = getattr(args, "limit", None)
        seed = int(getattr(args, "seed", 42))
        records = _sample_records(records, limit, seed)
        selection_metadata = {
            "selection_mode": "legacy_physical_split",
            "legacy_splits": sorted(set(map(str, splits))),
            "legacy_limit": limit,
            "seed": seed,
        }
    selected_ids = [str(row["question_id"]) for row in records]
    selection_metadata.update(
        {
            "selected_question_count": len(selected_ids),
            "selected_question_ids_sha256": _value_sha256(selected_ids),
        }
    )
    return records, selection_metadata


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def reference_review_passed(case: dict[str, Any]) -> bool:
    """Return whether the independent variant reviewer accepted the reference.

    Rejected variant generation must not remove a reviewed reference answer from
    the benchmark's noisy-positive source-answer check.
    """

    history = list(case.get("history") or [])
    if not history:
        return False
    candidates = list((history[-1].get("review") or {}).get("candidates") or [])
    complete = next(
        (row for row in candidates if str(row.get("candidate")) == "complete"), None
    )
    return bool(
        complete
        and complete.get("valid_for_calibration")
        and complete.get("actual_label") == "complete"
    )


def _aggregate_usage(
    cases: list[dict[str, Any]], judge_calls: list[dict[str, Any]]
) -> dict[str, int]:
    calls: list[dict[str, int]] = []
    for case in cases:
        for history in case.get("history") or []:
            for key in ("generation_call", "review_call"):
                usage = (history.get(key) or {}).get("usage")
                if usage:
                    calls.append(usage)
    for call in judge_calls:
        usage = call.get("usage")
        if usage:
            calls.append(usage)
    return {
        "calls": len(calls),
        "input_tokens": sum(int(row.get("input_tokens", 0)) for row in calls),
        "cached_input_tokens": sum(
            int(row.get("cached_input_tokens", 0)) for row in calls
        ),
        "output_tokens": sum(int(row.get("output_tokens", 0)) for row in calls),
        "total_tokens": sum(int(row.get("total_tokens", 0)) for row in calls),
    }


def _project_reports(
    cases: list[dict[str, Any]], grade_rows: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    projects = sorted(
        {str(row.get("project") or "unknown") for row in cases}
        | {str(row.get("project") or "unknown") for row in grade_rows}
    )
    output: dict[str, dict[str, Any]] = {}
    for project in projects:
        project_cases = [
            row for row in cases if str(row.get("project") or "unknown") == project
        ]
        project_grades = [
            row
            for row in grade_rows
            if str(row.get("project") or "unknown") == project
        ]
        accepted = sum(reference_review_passed(row) for row in project_cases)
        output[project] = {
            "source_records": len(project_cases),
            "reference_complete_rate": accepted / max(1, len(project_cases)),
            "metrics": classification_report(project_grades),
        }
    return output


def build_weak_supervision_gate(
    *,
    metrics: dict[str, Any],
    independent_source_review_rate: float,
    balanced_case_rate: float,
    min_source_answer_complete_recall: float,
    min_partial_recall: float,
    min_incorrect_recall: float,
    max_incorrect_false_complete_rate: float,
    min_macro_f1: float,
    min_balanced_case_rate: float,
) -> dict[str, Any]:
    """Apply the predeclared aggregate gates for noisy-positive supervision."""

    criteria = {
        "independent_source_review_rate": {
            "target_minimum": min_source_answer_complete_recall,
            "observed": independent_source_review_rate,
            "passed": independent_source_review_rate
            >= min_source_answer_complete_recall,
        },
        "source_answer_complete_recall": {
            "target_minimum": min_source_answer_complete_recall,
            "observed": metrics["complete_recall"],
            "passed": metrics["complete_recall"]
            >= min_source_answer_complete_recall,
        },
        "partial_control_recall": {
            "target_minimum": min_partial_recall,
            "observed": metrics["partial_recall"],
            "passed": metrics["partial_recall"] >= min_partial_recall,
        },
        "incorrect_control_recall": {
            "target_minimum": min_incorrect_recall,
            "observed": metrics["incorrect_recall"],
            "passed": metrics["incorrect_recall"] >= min_incorrect_recall,
        },
        "incorrect_false_complete_rate": {
            "target_maximum": max_incorrect_false_complete_rate,
            "observed": metrics["incorrect_false_complete_rate"],
            "passed": metrics["incorrect_false_complete_rate"]
            <= max_incorrect_false_complete_rate,
        },
        "macro_f1": {
            "target_minimum": min_macro_f1,
            "observed": metrics["macro_f1"],
            "passed": metrics["macro_f1"] >= min_macro_f1,
        },
        "balanced_case_rate": {
            "target_minimum": min_balanced_case_rate,
            "observed": balanced_case_rate,
            "passed": balanced_case_rate >= min_balanced_case_rate,
        },
    }
    return {
        "passed": all(row["passed"] for row in criteria.values()),
        "criteria": criteria,
    }


def render_report(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# Benchmark weak-supervision check",
        "",
        f"- Assumption: {report['weak_supervision']['assumption']}",
        "- Label status: noisy positives plus controlled silver contrasts",
        f"- Rubric: `{report['rubric_version']}`",
        f"- Rubric role: `{report['rubric_status']}`",
        f"- Model: `{report['model']}`",
        f"- Source questions: {report['source_records']}",
        f"- Valid balanced cases: {report['valid_balanced_cases']}",
        f"- Case provenance: `{report['case_provenance']['mode']}`",
        f"- Balanced answers: {metrics['examples']}",
        f"- Weak-supervision gate: {'PASS' if report['weak_supervision_gate']['passed'] else 'FAIL'}",
        f"- Accuracy: {metrics['accuracy']:.3f}",
        f"- Macro-F1: {metrics['macro_f1']:.3f}",
        f"- Source-answer complete recall: {metrics['complete_recall']:.3f}",
        "- Source-answer complete-recall 95% Wilson interval: "
        f"[{metrics['per_class']['complete']['recall_95ci'][0]:.3f}, "
        f"{metrics['per_class']['complete']['recall_95ci'][1]:.3f}]",
        f"- Partial-control recall: {metrics['partial_recall']:.3f}",
        f"- Incorrect-control recall: {metrics['incorrect_recall']:.3f}",
        f"- Incorrect false-complete rate: {metrics['incorrect_false_complete_rate']:.3f}",
        "",
        "## By project",
        "",
        "| Project | Questions | Reference complete | Accuracy | Macro-F1 | Complete recall | Partial recall | Incorrect recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for project, row in report["by_project"].items():
        values = row["metrics"]
        lines.append(
            f"| {project} | {row['source_records']} | {row['reference_complete_rate']:.3f} "
            f"| {values['accuracy']:.3f} | {values['macro_f1']:.3f} "
            f"| {values['complete_recall']:.3f} | {values['partial_recall']:.3f} "
            f"| {values['incorrect_recall']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The gate checks compatibility with the benchmark's source-answer assumption and controlled contrasts.",
            "It does not verify source-answer correctness against independent human expert labels; that verification is future work.",
            "",
        ]
    )
    return "\n".join(lines)


def _case_mode(args: argparse.Namespace) -> str:
    if getattr(args, "reuse_cases_from", None) is not None:
        return "reused_fixed_cases"
    if getattr(args, "seed_cases_from", None) is not None:
        return "seeded_fixed_cases_plus_generated"
    return "generated_cases"


def _case_generation_configuration(args: argparse.Namespace) -> dict[str, Any]:
    """Return the controls that can change generated calibration cases.

    Workers are deliberately omitted: concurrency changes scheduling, not the
    requested candidate-generation policy.  The request timeout is retained
    because it can change which questions finish or remain retryable errors.
    """

    mode = _case_mode(args)
    return {
        "mode": mode,
        "attempts": int(getattr(args, "attempts", 3)),
        "max_variant_repairs": int(getattr(args, "max_variant_repairs", 1)),
        "seed": int(getattr(args, "seed", 42)),
        "request_timeout_seconds": float(
            getattr(args, "request_timeout", 120.0)
        ),
    }


def _immutable_contract(
    args: argparse.Namespace,
    *,
    rubric: dict[str, Any],
    rubric_sha256: str,
    records: list[dict[str, Any]],
    selection_metadata: dict[str, Any],
    reused_case_sha256: str | None,
    seeded_case_provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    question_ids = [str(row["question_id"]) for row in records]
    contract = {
        "evaluation_name": "docsqa_weak_supervision_v1",
        "scorer_version": "docsqa-weak-supervision-scorer-v1",
        "aspects_sha256": _sha256(args.aspects),
        "selected_question_ids_sha256": _value_sha256(question_ids),
        "selected_question_count": len(question_ids),
        "splits": sorted(set(getattr(args, "split", []) or [])),
        "limit": getattr(args, "limit", None),
        "seed": args.seed,
        "rubric_version": rubric["rubric_version"],
        "rubric_sha256": rubric_sha256,
        "rubric_role": args.rubric_role,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "case_mode": _case_mode(args),
        "case_generation": _case_generation_configuration(args),
        "reused_case_sha256": reused_case_sha256,
        "thresholds": {
            "min_source_answer_complete_recall": args.min_source_answer_complete_recall,
            "min_partial_recall": args.min_partial_recall,
            "min_incorrect_recall": args.min_incorrect_recall,
            "max_incorrect_false_complete_rate": args.max_incorrect_false_complete_rate,
            "min_macro_f1": args.min_macro_f1,
            "min_balanced_case_rate": args.min_balanced_case_rate,
        },
    }
    if selection_metadata.get("selection_mode") == "weak_supervision_manifest":
        contract["question_selection"] = selection_metadata
    if seeded_case_provenance is not None:
        contract["seeded_case_provenance"] = seeded_case_provenance
    return contract


def _validate_resume_contract(output_dir: Path, expected: dict[str, Any]) -> None:
    path = output_dir / "run_contract.json"
    if not path.exists():
        artifacts = [
            output_dir / name
            for name in ("calibration_cases.jsonl", "per_candidate.jsonl", "judge_calls.jsonl")
        ]
        if any(artifact.exists() for artifact in artifacts):
            raise ValueError(
                "existing weak-supervision artifacts have no run_contract.json; "
                "use --report-only once to validate and adopt them before --resume"
            )
        return
    actual = json.loads(path.read_text(encoding="utf-8"))
    actual_immutable = actual.get("immutable")
    exact_match = actual_immutable == expected
    legacy_match = False
    expected_generation = expected.get("case_generation")
    legacy_uses_historical_defaults = bool(
        isinstance(expected_generation, dict)
        and expected_generation.get("mode") == expected.get("case_mode", "generated_cases")
        and expected_generation.get("attempts") == 3
        and expected_generation.get("max_variant_repairs") == 1
        and expected_generation.get("seed") == expected.get("seed")
        and expected_generation.get("request_timeout_seconds") == 120.0
    )
    if (
        isinstance(actual_immutable, dict)
        and "question_selection" not in expected
        and "case_generation" not in actual_immutable
        and "case_generation" in expected
        and legacy_uses_historical_defaults
    ):
        # Contracts written before case-generation controls were frozen can be
        # resumed only for the legacy physical-split interface.  Manifest runs
        # never receive this exception because they are the held-out protocol.
        upgraded = {**actual_immutable, "case_generation": expected["case_generation"]}
        legacy_match = upgraded == expected
    if not exact_match and not legacy_match:
        raise ValueError(
            "weak-supervision resume contract mismatch; use a new output directory"
        )
    artifact_paths = {
        "calibration_cases_sha256": output_dir / "calibration_cases.jsonl",
        "per_candidate_sha256": output_dir / "per_candidate.jsonl",
        "judge_calls_sha256": output_dir / "judge_calls.jsonl",
    }
    for field, expected_hash in (actual.get("artifacts") or {}).items():
        artifact_path = artifact_paths.get(str(field))
        if artifact_path is None or not artifact_path.exists():
            raise ValueError(f"weak-supervision contracted artifact is missing: {field}")
        if _sha256(artifact_path) != str(expected_hash):
            raise ValueError(
                f"weak-supervision contracted artifact hash mismatch: {field}"
            )


def _write_initial_contract(output_dir: Path, immutable: dict[str, Any]) -> None:
    path = output_dir / "run_contract.json"
    if path.exists():
        return
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "immutable": immutable,
                "artifacts": {},
                "result": {"status": "running"},
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_report_only_rows(
    records: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    grade_rows: list[dict[str, Any]],
    grade_calls: list[dict[str, Any]],
) -> None:
    expected_ids = {str(row["question_id"]) for row in records}
    case_ids = [str(row.get("question_id") or "") for row in cases]
    if len(case_ids) != len(set(case_ids)) or set(case_ids) != expected_ids:
        raise ValueError("report-only cases do not exactly match selected question IDs")
    accepted_ids = {
        str(row["question_id"]) for row in cases if row.get("status") == "accepted"
    }
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in grade_rows:
        candidate = str(row.get("candidate") or "")
        if str(row.get("expected_label") or "") != candidate:
            raise ValueError("report-only candidate expected labels are not frozen")
        if str(row.get("predicted_label") or "") not in {
            "complete",
            "partial",
            "incorrect",
        }:
            raise ValueError("report-only candidate has an invalid predicted label")
        grouped[str(row.get("question_id") or "")].append(candidate)
    if set(grouped) != accepted_ids:
        raise ValueError("report-only candidate rows do not match accepted cases")
    expected_candidates = ["complete", "incorrect", "partial"]
    invalid = {
        question_id: sorted(names)
        for question_id, names in grouped.items()
        if sorted(names) != expected_candidates
    }
    if invalid:
        raise ValueError(f"report-only candidate groups are incomplete: {invalid}")
    call_ids = [str(row.get("question_id") or "") for row in grade_calls]
    if len(call_ids) != len(set(call_ids)) or set(call_ids) != accepted_ids:
        raise ValueError("report-only judge calls do not match accepted cases")


def run(args: argparse.Namespace) -> dict[str, Any]:
    rubric, rubric_sha256 = load_judge_rubric(args.rubric)
    all_records = _jsonl(args.aspects)
    records, selection_metadata = _select_records(args, all_records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases: dict[str, dict[str, Any]] = {}
    case_path = args.output_dir / "calibration_cases.jsonl"
    candidate_path = args.output_dir / "per_candidate.jsonl"
    judge_call_path = args.output_dir / "judge_calls.jsonl"
    reused_cases_from = getattr(args, "reuse_cases_from", None)
    seed_cases_from = getattr(args, "seed_cases_from", None)
    report_only = bool(getattr(args, "report_only", False))
    reused_case_sha256: str | None = None
    seeded_case_sha256: str | None = None
    seeded_question_ids: set[str] = set()
    seeded_case_provenance: dict[str, Any] | None = None
    if reused_cases_from is not None:
        cases, reused_case_path, reused_case_sha256 = load_reused_cases(
            reused_cases_from, records
        )
        case_provenance = {
            "mode": "reused_fixed_cases",
            "source": str(reused_case_path),
            "source_sha256": reused_case_sha256,
        }
    else:
        _require_case_generation_rubric(rubric)
        if seed_cases_from is not None:
            (
                cases,
                seeded_case_path,
                seeded_case_sha256,
                seeded_source_provenance,
            ) = load_seed_cases(
                seed_cases_from, records
            )
            seeded_question_ids = set(cases)
            seeded_case_provenance = {
                "source": str(seeded_case_path),
                "source_sha256": seeded_case_sha256,
                **seeded_source_provenance,
            }
            case_provenance = {
                "mode": "seeded_fixed_cases_plus_generated",
                **seeded_case_provenance,
            }
        else:
            case_provenance = {
                "mode": "generated_frozen_cases"
                if report_only
                else "generated_for_this_run"
            }
        if (args.resume or report_only) and case_path.exists():
            if report_only:
                cases.update(
                    {str(row["question_id"]): row for row in _jsonl(case_path)}
                )
            else:
                cases.update(_resumable_calibration_cases(case_path))
    immutable_contract = _immutable_contract(
        args,
        rubric=rubric,
        rubric_sha256=rubric_sha256,
        records=records,
        selection_metadata=selection_metadata,
        reused_case_sha256=reused_case_sha256,
        seeded_case_provenance=seeded_case_provenance,
    )
    if args.resume and not report_only:
        _validate_resume_contract(args.output_dir, immutable_contract)
    if not report_only:
        existing_artifacts = [
            path for path in (case_path, candidate_path, judge_call_path) if path.exists()
        ]
        if existing_artifacts and not args.resume:
            raise ValueError(
                "weak-supervision output already contains artifacts; use --resume "
                "with its matching contract or choose a new output directory"
            )
        _write_initial_contract(args.output_dir, immutable_contract)
    if report_only:
        missing = [
            str(path)
            for path in (case_path, candidate_path, judge_call_path)
            if not path.exists()
        ]
        if missing:
            raise ValueError(f"--report-only requires existing artifacts: {missing}")
        frozen_case_rows = _jsonl(case_path)
        frozen_case_ids = [str(row.get("question_id") or "") for row in frozen_case_rows]
        expected_case_ids = {str(row["question_id"]) for row in records}
        if (
            len(frozen_case_ids) != len(set(frozen_case_ids))
            or set(frozen_case_ids) != expected_case_ids
        ):
            raise ValueError(
                "--report-only cases do not exactly and uniquely match selected questions"
            )
        cases = {str(row["question_id"]): row for row in frozen_case_rows}
        pending: list[dict[str, Any]] = []
        client = None
    else:
        from openai import OpenAI

        client = OpenAI(timeout=args.request_timeout)
        pending = [row for row in records if str(row["question_id"]) not in cases]
    lock = threading.Lock()
    if not report_only:
        with case_path.open("a", encoding="utf-8") as handle:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(
                        create_calibration_case,
                        client,
                        record=row,
                        rubric=rubric,
                        rubric_sha256=rubric_sha256,
                        model=args.model,
                        reasoning_effort=args.reasoning_effort,
                        attempts=args.attempts,
                        max_variant_repairs=args.max_variant_repairs,
                    ): str(row["question_id"])
                    for row in pending
                }
                for future in as_completed(futures):
                    question_id = futures[future]
                    case = future.result()
                    cases[question_id] = case
                    with lock:
                        handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")
                        handle.flush()
    ordered_cases = [cases[str(row["question_id"])] for row in records]
    _write_jsonl(case_path, ordered_cases)
    if case_provenance["mode"] == "generated_frozen_cases":
        case_provenance.update(
            {"source": str(case_path), "source_sha256": _sha256(case_path)}
        )
    elif case_provenance["mode"] == "seeded_fixed_cases_plus_generated":
        case_provenance["generated_question_count"] = (
            len(ordered_cases) - len(seeded_question_ids)
        )
    record_by_id = {str(row["question_id"]): row for row in records}
    valid_cases = [row for row in ordered_cases if row.get("status") == "accepted"]
    accepted_references = sum(reference_review_passed(row) for row in ordered_cases)
    reference_complete_review_rate = accepted_references / max(1, len(ordered_cases))
    grade_rows = (
        _jsonl(candidate_path)
        if (args.resume or report_only) and candidate_path.exists()
        else []
    )
    grade_calls = (
        _jsonl(judge_call_path)
        if (args.resume or report_only) and judge_call_path.exists()
        else []
    )
    completed_judges = {
        question_id
        for question_id, count in Counter(
            str(row["question_id"]) for row in grade_rows
        ).items()
        if count == 3
    }
    grade_rows = [
        row for row in grade_rows if str(row["question_id"]) in completed_judges
    ]
    grade_calls = [
        row for row in grade_calls if str(row["question_id"]) in completed_judges
    ]
    pending_judges = [
        case
        for case in valid_cases
        if str(case["question_id"]) not in completed_judges
    ]
    if not report_only:
        with candidate_path.open("a", encoding="utf-8") as candidate_handle:
            with judge_call_path.open("a", encoding="utf-8") as call_handle:
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = {
                        pool.submit(
                            grade_calibration_case,
                            client,
                            record=record_by_id[str(case["question_id"])],
                            case=case,
                            rubric=rubric,
                            model=args.model,
                            reasoning_effort=args.reasoning_effort,
                        ): str(case["question_id"])
                        for case in pending_judges
                    }
                    for index, future in enumerate(as_completed(futures), start=1):
                        rows, call = future.result()
                        grade_rows.extend(rows)
                        grade_calls.append(call)
                        with lock:
                            for row in rows:
                                candidate_handle.write(
                                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                )
                            call_handle.write(
                                json.dumps(call, ensure_ascii=False, sort_keys=True) + "\n"
                            )
                            candidate_handle.flush()
                            call_handle.flush()
                        done = len(completed_judges) + index
                        if done % 10 == 0 or done == len(valid_cases):
                            print(
                                f"weak-supervision judge progress: {done}/{len(valid_cases)}",
                                flush=True,
                            )
    else:
        _validate_report_only_rows(records, ordered_cases, grade_rows, grade_calls)
    candidate_order = {"complete": 0, "partial": 1, "incorrect": 2}
    grade_rows.sort(
        key=lambda row: (
            str(row["question_id"]),
            candidate_order[str(row["candidate"])],
        )
    )
    grade_calls.sort(key=lambda row: str(row["question_id"]))
    _write_jsonl(candidate_path, grade_rows)
    _write_jsonl(judge_call_path, grade_calls)
    metrics = classification_report(grade_rows)
    balanced_case_rate = len(valid_cases) / max(1, len(records))
    weak_supervision_gate = build_weak_supervision_gate(
        metrics=metrics,
        independent_source_review_rate=reference_complete_review_rate,
        balanced_case_rate=balanced_case_rate,
        min_source_answer_complete_recall=args.min_source_answer_complete_recall,
        min_partial_recall=args.min_partial_recall,
        min_incorrect_recall=args.min_incorrect_recall,
        max_incorrect_false_complete_rate=args.max_incorrect_false_complete_rate,
        min_macro_f1=args.min_macro_f1,
        min_balanced_case_rate=args.min_balanced_case_rate,
    )
    report = {
        "schema_version": 2,
        "evaluation_name": "docsqa_weak_supervision_v1",
        "rubric_version": rubric["rubric_version"],
        "rubric_sha256": rubric_sha256,
        "rubric_status": args.rubric_role,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "case_generation": immutable_contract["case_generation"],
        "question_selection": selection_metadata,
        "source_records": len(records),
        "valid_balanced_cases": len(valid_cases),
        "balanced_case_rate": balanced_case_rate,
        "case_status": dict(sorted(Counter(str(row.get("status")) for row in ordered_cases).items())),
        "reference_review": {
            "accepted_complete": accepted_references,
            "source_records": len(ordered_cases),
            "complete_rate": reference_complete_review_rate,
        },
        "metrics": metrics,
        "by_project": _project_reports(ordered_cases, grade_rows),
        "model_usage": _aggregate_usage(
            []
            if reused_cases_from is not None
            else [
                row
                for row in ordered_cases
                if str(row["question_id"]) not in seeded_question_ids
            ],
            grade_calls,
        ),
        "case_provenance": case_provenance,
        "weak_supervision": {
            "assumption": WEAK_SUPERVISION_ASSUMPTION,
            "positive_label": (
                "The normalized platform-selected answer and its resolved local "
                "documentation are treated as a noisy complete-answer reference."
            ),
            "controls": ["useful_partial", "decisively_incorrect"],
            "rule_selection": {
                "policy": "freeze the first domain-general rubric satisfying every gate",
                "role_of_this_run": args.rubric_role,
                "selected_for_agent_evaluation": args.rubric_role == "frozen",
                "stopped_without_rule_edit": (
                    args.rubric_role == "frozen" and weak_supervision_gate["passed"]
                ),
                "agent_answers_used_for_selection": False,
                "case_specific_overrides_forbidden": True,
            },
            "human_verification_status": "not_performed_deferred_to_future_work",
        },
        "weak_supervision_gate": weak_supervision_gate,
        "boundary": (
            "Passing this gate establishes compatibility with a noisy-positive "
            "benchmark assumption and discrimination on controlled silver contrasts. "
            "It does not establish independent human correctness of source answers or "
            "human alignment of the judge."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(
        render_report(report), encoding="utf-8"
    )
    contract = {
        "schema_version": 1,
        "immutable": immutable_contract,
        "artifacts": {
            "calibration_cases_sha256": _sha256(case_path),
            "per_candidate_sha256": _sha256(candidate_path),
            "judge_calls_sha256": _sha256(judge_call_path),
        },
        "result": {
            "weak_supervision_gate_passed": weak_supervision_gate["passed"],
            "rubric_frozen": (
                args.rubric_role == "frozen" and weak_supervision_gate["passed"]
            ),
            "report_sha256": _sha256(args.output_dir / "report.json"),
        },
    }
    (args.output_dir / "run_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aspects", type=Path, required=True)
    parser.add_argument(
        "--rubric",
        type=Path,
        default=Path(__file__).resolve().parent / "rubrics/aspect_evaluation_v2.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    case_source = parser.add_mutually_exclusive_group()
    case_source.add_argument(
        "--reuse-cases-from",
        type=Path,
        help=(
            "reuse an exact calibration_cases.jsonl (or its parent directory) "
            "instead of generating variants; intended for judge-prompt ablations"
        ),
    )
    case_source.add_argument(
        "--seed-cases-from",
        type=Path,
        help=(
            "seed the run from a subset calibration_cases.jsonl (or its parent "
            "directory), then generate cases for selected questions not present"
        ),
    )
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument(
        "--rubric-role",
        choices=("frozen", "candidate", "ablation"),
        default="frozen",
        help="provenance role of this rubric; only frozen is used for agent scoring",
    )
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-variant-repairs", type=int, default=1)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--split", action="append", choices=("train", "validation"), default=[])
    parser.add_argument("--question-manifest", type=Path)
    parser.add_argument("--manifest-partition", choices=PARTITIONS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-source-answer-complete-recall",
        "--target-complete-recall",
        dest="min_source_answer_complete_recall",
        type=float,
        default=DEFAULT_MIN_SOURCE_ANSWER_COMPLETE_RECALL,
        help=(
            "minimum recall for noisy-positive source answers; the legacy "
            "--target-complete-recall spelling is retained as an alias"
        ),
    )
    parser.add_argument(
        "--min-partial-recall", type=float, default=DEFAULT_MIN_CONTROL_RECALL
    )
    parser.add_argument(
        "--min-incorrect-recall", type=float, default=DEFAULT_MIN_CONTROL_RECALL
    )
    parser.add_argument(
        "--max-incorrect-false-complete-rate",
        type=float,
        default=DEFAULT_MAX_INCORRECT_FALSE_COMPLETE_RATE,
    )
    parser.add_argument("--min-macro-f1", type=float, default=DEFAULT_MIN_MACRO_F1)
    parser.add_argument(
        "--min-balanced-case-rate",
        type=float,
        default=DEFAULT_MIN_BALANCED_CASE_RATE,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="rebuild and validate reports from frozen local artifacts without an API call",
    )
    parser.add_argument(
        "--allow-gate-failure",
        action="store_true",
        help="write a failed development report instead of exiting with status 2",
    )
    args = parser.parse_args()
    try:
        _validate_question_selection_args(args)
    except ValueError as error:
        parser.error(str(error))
    if args.question_manifest is None and not args.split:
        args.split = ["validation"]
    for name in (
        "min_source_answer_complete_recall",
        "min_partial_recall",
        "min_incorrect_recall",
        "max_incorrect_false_complete_rate",
        "min_macro_f1",
        "min_balanced_case_rate",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1")
    if not args.report_only and not load_openai_key_from_configured_env():
        raise SystemExit("OPENAI_API_KEY is unset; set it or configure KBBENCH_OPENAI_ENV_FILE")
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["weak_supervision_gate"]["passed"] and not args.allow_gate_failure:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
