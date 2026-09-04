"""Describe the frozen DocsQA corpus, questions, and aspect support conditions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _counts(values: list[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _aspect_has_document_support(aspect: dict[str, Any]) -> bool:
    return bool(aspect.get("retrieval_doc_ids"))


def describe(dataset_dir: Path, aspects_path: Path | None = None) -> dict[str, Any]:
    corpus = _jsonl(dataset_dir / "corpus.jsonl")
    questions = _jsonl(dataset_dir / "questions.jsonl")
    report: dict[str, Any] = {
        "dataset": str(dataset_dir),
        "documents": len(corpus),
        "questions": len(questions),
        "projects": _counts([row.get("project") for row in questions]),
        "splits": _counts([row.get("split") for row in questions]),
        "qrel_count": _counts([row.get("qrel_count") for row in questions]),
        "qrels": sum(int(row.get("qrel_count") or 0) for row in questions),
        "evidence_structure": _counts(
            [row.get("evidence_structure") for row in questions]
        ),
        "evidence_category": _counts(
            [row.get("evidence_category") for row in questions]
        ),
        "intent_category": _counts(
            [row.get("intent_category") for row in questions]
        ),
        "question_images": {
            "questions_with_images": sum(bool(row.get("question_images")) for row in questions),
            "questions_without_images": sum(not row.get("question_images") for row in questions),
        },
        "corpus_structure": {
            "documents_with_markdown_links": sum(
                bool(row.get("link_edges")) for row in corpus
            ),
            "markdown_link_edges": sum(
                len(row.get("link_edges") or []) for row in corpus
            ),
            "documents_with_image_derived_text": sum(
                bool(row.get("image_texts")) for row in corpus
            ),
            "image_text_occurrences": sum(
                len(row.get("image_texts") or []) for row in corpus
            ),
            "documents_with_fenced_code": sum(
                "```" in str(row.get("rendered_text") or "") for row in corpus
            ),
        },
    }
    if aspects_path is None:
        return report

    aspects = _jsonl(aspects_path)
    aspect_rows = [aspect for row in aspects for aspect in row.get("aspects") or []]
    report["aspect_support"] = {
        "questions": len(aspects),
        "aspects": len(aspect_rows),
        "mean_aspects_per_question": len(aspect_rows) / max(1, len(aspects)),
        "document_supported_aspects": sum(
            _aspect_has_document_support(aspect) for aspect in aspect_rows
        ),
        "accepted_answer_or_question_only_aspects": sum(
            not _aspect_has_document_support(aspect) for aspect in aspect_rows
        ),
        "questions_with_any_answer_or_question_only_aspect": sum(
            any(not _aspect_has_document_support(aspect) for aspect in row["aspects"])
            for row in aspects
        ),
        "questions_with_no_document_supported_aspect": sum(
            not any(_aspect_has_document_support(aspect) for aspect in row["aspects"])
            for row in aspects
        ),
        "questions_with_document_supported_critical_aspect": sum(
            any(
                aspect.get("critical") and _aspect_has_document_support(aspect)
                for aspect in row["aspects"]
            )
            for row in aspects
        ),
        "questions_with_all_critical_aspects_document_supported": sum(
            all(
                not aspect.get("critical") or _aspect_has_document_support(aspect)
                for aspect in row["aspects"]
            )
            for row in aspects
        ),
        "questions_with_all_aspects_document_supported": sum(
            all(_aspect_has_document_support(aspect) for aspect in row["aspects"])
            for row in aspects
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--aspects", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = describe(args.dataset_dir, args.aspects)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
