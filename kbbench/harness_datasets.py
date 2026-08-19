from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


FORMATS = ("auto", "hotpot", "techqa")


def load_harness_questions(
    path: Path,
    question_format: str = "auto",
    answerable_only: bool = True,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw):
        raise ValueError("harness questions must be a JSON array of objects")
    detected = _detect_format(raw) if question_format == "auto" else question_format
    if detected not in FORMATS[1:]:
        raise ValueError(f"unknown question format: {detected}")
    if detected == "techqa":
        questions = _normalize_techqa(raw, answerable_only=answerable_only)
        dataset_id = "nvidia/TechQA-RAG-Eval"
        score_namespace = "nvidia-techqa-shared-harness-development-v1"
    else:
        questions = _normalize_hotpot(raw)
        dataset_id = "HotpotQA linked-Markdown derivative"
        score_namespace = "hotpot-linked-markdown-shared-harness-v1"
    ids = [str(question["id"]) for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("harness question IDs must be unique")
    manifest = {
        "dataset_id": dataset_id,
        "score_namespace": score_namespace,
        "question_format": detected,
        "source_path": str(path.resolve()),
        "source_sha256": _sha256_file(path),
        "source_questions": len(raw),
        "eligible_questions": len(questions),
        "answerable_only": answerable_only,
        "prompt_fields": ["id", "question"],
        "scoring_only_fields": ["answer", "relevant_ids"],
        "provided_context_text_used": False,
        "candidate_document_lists_used": False,
    }
    return questions, manifest


def _detect_format(rows: list[dict[str, Any]]) -> str:
    if not rows:
        raise ValueError("question file is empty")
    first = rows[0]
    if "is_impossible" in first and "contexts" in first:
        return "techqa"
    if "supporting" in first:
        return "hotpot"
    raise ValueError("could not infer question format")


def _normalize_techqa(
    rows: list[dict[str, Any]], answerable_only: bool
) -> list[dict[str, object]]:
    questions: list[dict[str, object]] = []
    for row in rows:
        impossible = bool(row.get("is_impossible"))
        if answerable_only and impossible:
            continue
        relevant_ids = sorted(
            {
                str(context.get("filename") or "").strip()
                for context in row.get("contexts") or []
                if isinstance(context, dict) and str(context.get("filename") or "").strip()
            }
        )
        if not impossible and not relevant_ids:
            raise ValueError(f"answerable TechQA row {row.get('id')} has no qrel filename")
        questions.append(
            {
                "id": str(row["id"]),
                "question": str(row["question"]),
                "answer": "NOT FOUND" if impossible else str(row.get("answer") or ""),
                "type": "unanswerable" if impossible else "technical-support",
                "answer_style": "technical",
                "relevant_ids": relevant_ids,
            }
        )
    return questions


def _normalize_hotpot(rows: list[dict[str, Any]]) -> list[dict[str, object]]:
    questions: list[dict[str, object]] = []
    for row in rows:
        questions.append(
            {
                "id": str(row["id"]),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "type": str(row.get("type") or ""),
                "answer_style": "short",
                "supporting": [str(value) for value in row.get("supporting") or []],
            }
        )
    return questions


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()
