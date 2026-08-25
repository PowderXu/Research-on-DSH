#!/usr/bin/env python3
"""Standalone evaluator for the GitHub Docs KB retrieval benchmark."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import unquote, urlsplit


CUTOFFS = (1, 5, 10, 20)
VERSION_SEGMENT = re.compile(
    r"^(?:enterprise-cloud|enterprise-server|free-pro-team)(?:@[^/]+)?$",
    re.IGNORECASE,
)


def normalize_doc_id(value: str) -> str:
    path = unquote(str(value or "").strip()).replace("\\", "/")
    path = path.split("#", 1)[0].split("?", 1)[0]
    path = re.sub(r"/+", "/", path)
    if path.endswith(".md"):
        path = path[:-3]
    if path.endswith("/index"):
        path = path[:-6] or "/"
    return "/" + path.strip("/") if path.strip("/") else "/"


def read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return value
    return [json.loads(line) for line in text.splitlines() if line.strip()]


class SourceResolver:
    """Resolve common citation forms to canonical corpus document IDs."""

    def __init__(self, corpus_path: Path) -> None:
        self.aliases: dict[str, str] = {}
        with corpus_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                doc_id = normalize_doc_id(str(row["doc_id"]))
                source_path = str(row.get("source_path") or "").replace("\\", "/")
                route = str(row.get("route") or "")
                candidates = {
                    doc_id,
                    doc_id.lstrip("/"),
                    source_path,
                    source_path.removeprefix("content/"),
                    f"content/{source_path.removeprefix('content/')}",
                    normalize_doc_id(source_path),
                    normalize_doc_id(route),
                }
                for candidate in candidates:
                    key = str(candidate).strip().casefold()
                    if key:
                        self.aliases[key] = doc_id

    def resolve(self, source: str) -> str | None:
        raw = str(source or "").strip()
        if not raw:
            return None
        if raw.startswith("viking://"):
            raw = urlsplit(raw).path
            marker = "/docsqa/"
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
            normalize_doc_id(raw),
            normalize_doc_id(raw).lstrip("/"),
        ]
        for candidate in candidates:
            resolved = self.aliases.get(str(candidate).strip().casefold())
            if resolved:
                return resolved
        return None

    def resolve_ranked(self, sources: Iterable[str], limit: int = 20) -> tuple[list[str], list[str]]:
        ranked: list[str] = []
        unresolved: list[str] = []
        seen: set[str] = set()
        for source in sources:
            resolved = self.resolve(source)
            if resolved and resolved not in seen:
                ranked.append(resolved)
                seen.add(resolved)
            elif not resolved:
                unresolved.append(source)
            if len(ranked) >= limit:
                break
        return ranked, unresolved


def source_strings(prediction: dict[str, Any]) -> list[str]:
    values = prediction.get("ranked_ids")
    if values is None:
        values = prediction.get("sources")
    if not isinstance(values, list):
        raise ValueError(f"Prediction {prediction.get('question_id')} has no ranked source list")
    result: list[str] = []
    for item in values:
        if isinstance(item, str):
            result.append(item)
            continue
        if isinstance(item, dict):
            for key in ("doc_id", "source", "path", "uri", "url"):
                if item.get(key):
                    result.append(str(item[key]))
                    break
            else:
                result.append("")
            continue
        result.append(str(item))
    return result


def score(ranked_ids: Iterable[str], relevant_ids: Iterable[str]) -> dict[str, float]:
    ranked = list(dict.fromkeys(normalize_doc_id(value) for value in ranked_ids))
    relevant = {normalize_doc_id(value) for value in relevant_ids}
    metrics: dict[str, float] = {}
    for cutoff in CUTOFFS:
        found = sum(doc_id in relevant for doc_id in ranked[:cutoff])
        metrics[f"Recall@{cutoff}"] = found / max(1, len(relevant))
        metrics[f"Hit@{cutoff}"] = float(found > 0)

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(ranked[:10], start=1)
        if doc_id in relevant
    )
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(10, len(relevant)) + 1)
    )
    metrics["nDCG@10"] = dcg / ideal if ideal else 0.0
    metrics["AllSupport@10"] = float(bool(relevant) and relevant.issubset(set(ranked[:10])))
    return metrics


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def prediction_latency_ms(prediction: dict[str, Any]) -> float | None:
    value = prediction.get("latency_ms")
    if value is None and prediction.get("latency_seconds") is not None:
        value = float(prediction["latency_seconds"]) * 1000.0
    if value is None:
        return None
    value = float(value)
    if value < 0:
        raise ValueError("Latency cannot be negative")
    return value


def prediction_tokens(prediction: dict[str, Any]) -> float | None:
    usage = prediction.get("usage") or {}
    if not isinstance(usage, dict):
        return None
    if usage.get("total_tokens") is not None:
        return float(usage["total_tokens"])
    available = [usage.get("input_tokens"), usage.get("output_tokens")]
    if any(value is not None for value in available):
        return sum(float(value or 0) for value in available)
    return None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = [*(f"Recall@{k}" for k in CUTOFFS), *(f"Hit@{k}" for k in CUTOFFS), "nDCG@10", "AllSupport@10"]
    result: dict[str, Any] = {"questions": len(rows)}
    for key in metric_keys:
        result[key] = statistics.fmean(row["metrics"][key] for row in rows) if rows else 0.0
    latencies = [row["latency_ms"] for row in rows if row["latency_ms"] is not None]
    tokens = [row["total_tokens"] for row in rows if row["total_tokens"] is not None]
    result["latency_observations"] = len(latencies)
    result["p50_latency_ms"] = percentile(latencies, 0.50)
    result["p95_latency_ms"] = percentile(latencies, 0.95)
    result["token_observations"] = len(tokens)
    result["mean_total_tokens"] = statistics.fmean(tokens) if tokens else None
    result["p50_total_tokens"] = percentile(tokens, 0.50)
    return result


def evaluate(
    questions_path: Path,
    corpus_path: Path,
    predictions_path: Path,
    allow_partial: bool = False,
) -> dict[str, Any]:
    questions = read_records(questions_path)
    predictions = read_records(predictions_path)
    question_by_id = {str(row["question_id"]): row for row in questions}
    if len(question_by_id) != len(questions):
        raise ValueError("Question file contains duplicate IDs")

    prediction_by_id: dict[str, dict[str, Any]] = {}
    for prediction in predictions:
        question_id = str(prediction.get("question_id") or "")
        if not question_id:
            raise ValueError("Prediction is missing question_id")
        if question_id in prediction_by_id:
            raise ValueError(f"Duplicate prediction for question {question_id}")
        if question_id not in question_by_id:
            raise ValueError(f"Unknown prediction question_id {question_id}")
        prediction_by_id[question_id] = prediction

    missing = sorted(set(question_by_id) - set(prediction_by_id))
    if missing and not allow_partial:
        preview = ", ".join(missing[:10])
        raise ValueError(f"Missing predictions for {len(missing)} questions: {preview}")

    resolver = SourceResolver(corpus_path)
    per_question: list[dict[str, Any]] = []
    unresolved_total = 0
    for question in questions:
        question_id = str(question["question_id"])
        prediction = prediction_by_id.get(question_id)
        if prediction is None:
            continue
        ranked, unresolved = resolver.resolve_ranked(source_strings(prediction), limit=20)
        unresolved_total += len(unresolved)
        qrel_ids = list(dict.fromkeys(question.get("qrel_ids") or []))
        qrel_count = int(question.get("qrel_count") or len(qrel_ids))
        structure = question.get("evidence_structure")
        if not structure:
            structure = "single" if qrel_count <= 1 else (
                "linked" if question.get("evidence_category") == "multi_page_linked" else "dispersed"
            )
        per_question.append({
            "question_id": question_id,
            "qrel_count": qrel_count,
            "qrel_count_group": "1" if qrel_count == 1 else "2+",
            "intent_category": question.get("intent_category", "unknown"),
            "evidence_category": question.get("evidence_category", "unknown"),
            "evidence_structure": structure,
            "ranked_ids": ranked,
            "unresolved_sources": unresolved,
            "latency_ms": prediction_latency_ms(prediction),
            "total_tokens": prediction_tokens(prediction),
            "metrics": score(ranked, qrel_ids),
        })

    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        field: defaultdict(list)
        for field in ("intent_category", "evidence_category", "evidence_structure", "qrel_count_group")
    }
    for row in per_question:
        for field, values in groups.items():
            values[str(row[field])].append(row)

    return {
        "benchmark": "github-docs-kb-retrieval-v2.1",
        "questions_file": str(questions_path),
        "corpus_file": str(corpus_path),
        "predictions_file": str(predictions_path),
        "allow_partial": allow_partial,
        "overall": aggregate(per_question),
        "slices": {
            field: {name: aggregate(rows) for name, rows in sorted(values.items())}
            for field, values in groups.items()
        },
        "diagnostics": {
            "expected_questions": len(questions),
            "evaluated_questions": len(per_question),
            "missing_question_ids": missing,
            "unresolved_source_count": unresolved_total,
        },
        "per_question": per_question,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(args.questions, args.corpus, args.predictions, args.allow_partial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))


if __name__ == "__main__":
    main()
