"""Separate question inputs from reference answers, joined only by evaluators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


QUESTION_FIELDS = frozenset({
    "question_id", "source_question_id", "project", "dataset", "title", "query",
    "source_url", "community_category", "question_images", "question_modalities",
})
PARTITION_FIELDS = frozenset({"split", "benchmark_split"})
RECORD_LAYOUT = "separate-questions-answers-v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        key = row.get("question_id")
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label}: missing or invalid question_id")
        if key in result:
            raise ValueError(f"{label}: duplicate question_id {key}")
        result[key] = row
    return result


def load_records(dataset_dir: Path) -> list[dict[str, Any]]:
    """Load evaluator records; support archived combined source packages too."""
    questions = read_jsonl(dataset_dir / "questions.jsonl")
    answer_path = dataset_dir / "answers.jsonl"
    if not answer_path.exists():
        manifest_path = dataset_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if manifest.get("record_layout") == RECORD_LAYOUT:
            raise ValueError("separate question package is missing answers.jsonl")
        return questions
    by_question = _index(questions, "questions")
    answers = _index(read_jsonl(answer_path), "answers")
    if by_question.keys() != answers.keys():
        raise ValueError(
            f"question/answer ID mismatch: missing={sorted(by_question.keys() - answers.keys())}, "
            f"orphan={sorted(answers.keys() - by_question.keys())}"
        )
    joined = []
    for question in questions:
        answer = answers[question["question_id"]]
        if set(question) - QUESTION_FIELDS:
            raise ValueError(f"non-input fields in questions.jsonl: {question['question_id']}")
        if (set(answer) & (QUESTION_FIELDS - {"question_id"})) or set(answer) & PARTITION_FIELDS:
            raise ValueError(f"question or partition fields in answers.jsonl: {question['question_id']}")
        joined.append({**question, **answer})
    return joined


def write_records(dataset_dir: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write one unpartitioned pair; preserve all non-partition field values."""
    if (dataset_dir / "splits").exists():
        raise ValueError("migrate the legacy dataset before writing an unpartitioned package")
    rows = list(rows)
    _index(rows, "records")
    questions, answers = [], []
    for row in rows:
        questions.append({key: value for key, value in row.items() if key in QUESTION_FIELDS})
        answers.append({
            "question_id": row["question_id"],
            **{key: value for key, value in row.items() if key not in QUESTION_FIELDS | PARTITION_FIELDS},
        })
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("questions.jsonl", questions), ("answers.jsonl", answers)):
        with (dataset_dir / name).open("w", encoding="utf-8") as handle:
            for row in values:
                encoded = json.dumps(row, ensure_ascii=False, sort_keys=True)
                # Keep one record per line even for readers using splitlines().
                for character in ("\u0085", "\u2028", "\u2029"):
                    encoded = encoded.replace(character, f"\\u{ord(character):04x}")
                handle.write(encoded + "\n")
