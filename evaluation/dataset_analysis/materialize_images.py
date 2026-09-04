"""Materialize every image referenced by the validated DocsQA dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .image_evidence import (
    ImageResolver,
    collect_image_references,
    load_project_image_configs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_RUNS = PROJECT_ROOT / "results/runs/dataset-analysis"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    questions = _load_jsonl(args.dataset_dir / "questions.jsonl")
    if args.question_id:
        wanted = set(args.question_id)
        questions = [row for row in questions if str(row["question_id"]) in wanted]
    if args.limit:
        questions = questions[: args.limit]
    corpus = {
        str(row["doc_id"]): row
        for row in _load_jsonl(args.dataset_dir / "corpus.jsonl")
    }
    configs = load_project_image_configs(args.source_config, args.source_root)
    resolver = ImageResolver(
        configs,
        args.cache_dir,
        allow_remote=not args.offline,
        max_bytes=args.max_image_bytes,
        timeout_seconds=args.timeout,
    )
    per_question: list[dict[str, Any]] = []
    per_image: list[dict[str, Any]] = []
    for question in questions:
        references = collect_image_references(question, corpus)
        images = [resolver.resolve(reference) for reference in references]
        per_image.extend(images)
        failures = [image for image in images if image["status"] == "failed"]
        text_fallbacks = [image for image in images if image["status"] == "text_fallback"]
        per_question.append(
            {
                "question_id": question["question_id"],
                "project": question.get("project") or question.get("dataset"),
                "requires_multimodal_judgment": bool(
                    question.get("requires_multimodal_judgment")
                ),
                "requested_images": len(images),
                "resolved_images": len(images) - len(failures),
                "pixel_resolved_images": sum(image["status"] == "resolved" for image in images),
                "text_fallback_images": len(text_fallbacks),
                "failed_images": len(failures),
                "all_images_resolved": not failures,
                "images": images,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_image.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_image:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_dir / "per_question.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_question:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    errors = Counter(
        str(row.get("error_code") or "unknown")
        for row in per_image
        if row["status"] == "failed"
    )
    report = {
        "questions": len(per_question),
        "multimodal_questions": sum(
            row["requires_multimodal_judgment"] for row in per_question
        ),
        "image_references": len(per_image),
        "resolved_images": sum(row["status"] in {"resolved", "text_fallback"} for row in per_image),
        "pixel_resolved_images": sum(row["status"] == "resolved" for row in per_image),
        "text_fallback_images": sum(row["status"] == "text_fallback" for row in per_image),
        "failed_images": sum(row["status"] == "failed" for row in per_image),
        "questions_with_complete_images": sum(
            row["all_images_resolved"] for row in per_question
        ),
        "multimodal_questions_with_complete_images": sum(
            row["requires_multimodal_judgment"] and row["all_images_resolved"]
            for row in per_question
        ),
        "failure_types": dict(sorted(errors.items())),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Image evidence materialization",
        "",
        f"- Questions: {report['questions']}",
        f"- Multimodal questions: {report['multimodal_questions']}",
        f"- Image references: {report['image_references']}",
        f"- Resolved images: {report['resolved_images']}",
        f"- Pixel-resolved images: {report['pixel_resolved_images']}",
        f"- Pinned alt-text fallbacks: {report['text_fallback_images']}",
        f"- Failed images: {report['failed_images']}",
        f"- Multimodal questions with complete image evidence: {report['multimodal_questions_with_complete_images']}",
        "",
        "A multimodal semantic judgment is valid only when every image required by that question is resolved.",
        "",
        "## Failure types",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in report["failure_types"].items())
    (args.output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=PROJECT_ROOT / "evaluation/dataset/evaluation_data/combined",
    )
    parser.add_argument(
        "--source-config",
        type=Path,
        default=PROJECT_ROOT / "evaluation/dataset/templates/public_sources.json",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=PROJECT_ROOT / "results/cache/docsqa-source-repos",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "results/cache/dataset_images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ANALYSIS_RUNS / "image-validation",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-image-bytes", type=int, default=20 * 1024 * 1024)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    print(json.dumps(materialize(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
