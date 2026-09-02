#!/usr/bin/env python3
"""Combine validated per-project DocsQA packages into one namespaced dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .prepare_unified_corpus import SourceSpec, load_source_specs


VALIDATION_DATASET_NAMES = {
    "github-docs": "github_docs",
    "tailwind-css": "tailwind",
    "prisma": "prisma",
    "supabase": "supabase",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _namespace(project: str, value: str) -> str:
    return f"{project}::{value}"


def _namespace_corpus(
    project: str,
    rows: list[dict[str, Any]],
    spec: SourceSpec,
) -> list[dict[str, Any]]:
    old_ids = {str(row["doc_id"]) for row in rows}
    content_root = spec.content_roots[0].as_posix()
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        source_doc_id = str(source["doc_id"])
        repository_relative = f"{content_root}/{source['source_path']}"
        row.update(
            {
                "doc_id": _namespace(project, source_doc_id),
                "source_doc_id": source_doc_id,
                "project": project,
                "repository_source_path": repository_relative,
                "source_path": f"{project}/{repository_relative}",
                "route": _namespace(project, str(source.get("route") or "/")),
            }
        )
        for field in ("outgoing_ids", "outgoing_paths"):
            row[field] = [
                _namespace(project, str(value))
                for value in source.get(field) or []
                if str(value) in old_ids
            ]
        row["link_edges"] = [
            {
                **edge,
                "target_id": _namespace(project, str(edge["target_id"])),
            }
            for edge in source.get("link_edges") or []
            if str(edge.get("target_id") or "") in old_ids
        ]
        output.append(row)
    return output


def _split_membership(dataset_dir: Path) -> dict[str, str]:
    membership: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        path = dataset_dir / "splits" / f"{split}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload:
            question_id = str(row["question_id"])
            previous = membership.get(question_id)
            if previous and previous != split:
                raise ValueError(
                    f"question {question_id} appears in both {previous} and {split}"
                )
            membership[question_id] = split
    return membership


def _namespace_question(
    project: str,
    source: dict[str, Any],
    eligible: dict[str, Any],
    split: str,
    corpus_ids: set[str],
) -> dict[str, Any]:
    source_question_id = str(source["question_id"])
    qrel_ids = [_namespace(project, str(value)) for value in eligible["qrel_ids"]]
    qrel_id_set = set(qrel_ids)
    missing = sorted(qrel_id_set - corpus_ids)
    if missing:
        raise ValueError(
            f"validated question {source_question_id} has missing namespaced qrels: {missing}"
        )
    row = {key: value for key, value in source.items() if key != "graph_opportunity"}
    anchors = source.get("qrel_anchors") or {}
    document_images = eligible.get("document_images") or {}
    row.update(
        {
            "question_id": _namespace(project, source_question_id),
            "source_question_id": source_question_id,
            "project": project,
            "dataset": project,
            "split": split,
            "reference_answer": str(eligible["expanded_reference_answer"]),
            "question_images": eligible.get("question_images") or [],
            "reference_answer_images": eligible.get("answer_images") or [],
            "reference_answer_links": eligible.get("resolved_internal_docs_urls") or [],
            "docs_urls": eligible.get("resolved_internal_docs_urls") or [],
            "qrel_ids": qrel_ids,
            "qrel_count": len(qrel_ids),
            "qrel_anchors": {
                _namespace(project, str(doc_id)): value
                for doc_id, value in anchors.items()
                if _namespace(project, str(doc_id)) in qrel_id_set
            },
            "document_images": {
                _namespace(project, str(doc_id)): images
                for doc_id, images in document_images.items()
            },
            "linked_answer_sources": eligible.get("linked_answer_sources") or [],
            "requires_multimodal_judgment": bool(
                eligible.get("requires_multimodal_judgment")
            ),
            "source_validation": "accepted_structurally",
        }
    )
    if source.get("url_resolutions"):
        row["url_resolutions"] = {
            str(url): {
                **dict(resolution),
                "doc_id": _namespace(
                    project, str(resolution.get("doc_id") or "")
                ),
            }
            for url, resolution in source["url_resolutions"].items()
            if resolution.get("doc_id")
            and _namespace(project, str(resolution.get("doc_id"))) in qrel_id_set
        }
    return row


def combine_datasets(
    dataset_dirs: dict[str, Path],
    specs: list[SourceSpec],
    eligible_path: Path,
    output_dir: Path,
    *,
    force: bool,
) -> dict[str, Any]:
    if output_dir.exists() and not force:
        raise RuntimeError(f"output already exists; pass --force to replace it: {output_dir}")
    spec_by_id = {spec.source_id: spec for spec in specs}
    if set(dataset_dirs) != set(spec_by_id):
        missing = sorted(set(spec_by_id) - set(dataset_dirs))
        extra = sorted(set(dataset_dirs) - set(spec_by_id))
        raise ValueError(f"dataset inputs must match configured projects; missing={missing}, extra={extra}")

    eligible_rows = _load_jsonl(eligible_path)
    eligible_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in eligible_rows:
        if row.get("status") != "accepted_structurally" or row.get("rejection_reasons"):
            raise ValueError("eligible input contains a rejected or non-accepted row")
        key = (str(row["dataset"]), str(row["question_id"]))
        if key in eligible_by_key:
            raise ValueError(f"duplicate eligible question: {key}")
        eligible_by_key[key] = row

    corpus: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    project_summaries: list[dict[str, Any]] = []
    for project in [spec.source_id for spec in specs]:
        dataset_dir = dataset_dirs[project]
        source_corpus = _load_jsonl(dataset_dir / "corpus.jsonl")
        project_corpus = _namespace_corpus(project, source_corpus, spec_by_id[project])
        corpus.extend(project_corpus)
        corpus_ids = {str(row["doc_id"]) for row in project_corpus}
        source_questions = {
            str(row["question_id"]): row
            for row in _load_jsonl(dataset_dir / "questions.jsonl")
        }
        split_membership = _split_membership(dataset_dir)
        validation_name = VALIDATION_DATASET_NAMES[project]
        accepted_for_project = {
            question_id: row
            for (dataset_name, question_id), row in eligible_by_key.items()
            if dataset_name == validation_name
        }
        for question_id, eligible in sorted(accepted_for_project.items()):
            source = source_questions.get(question_id)
            if source is None:
                raise ValueError(
                    f"eligible question is absent from {project} source package: {question_id}"
                )
            split = split_membership.get(question_id)
            if split is None:
                original = str(source.get("split") or "")
                split = "validation" if original == "dev" else original
            if split not in {"train", "validation", "test"}:
                raise ValueError(f"question {question_id} has invalid split: {split!r}")
            questions.append(
                _namespace_question(project, source, eligible, split, corpus_ids)
            )
        project_summaries.append(
            {
                "project": project,
                "repository": spec_by_id[project].repository,
                "revision": spec_by_id[project].revision,
                "documents": len(project_corpus),
                "questions": len(accepted_for_project),
            }
        )

    corpus.sort(key=lambda row: str(row["doc_id"]))
    questions.sort(key=lambda row: str(row["question_id"]))
    if len({str(row["doc_id"]) for row in corpus}) != len(corpus):
        raise ValueError("combined corpus contains duplicate document IDs")
    if len({str(row["question_id"]) for row in questions}) != len(questions):
        raise ValueError("combined dataset contains duplicate question IDs")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    completed = False
    try:
        _write_jsonl(staging / "corpus.jsonl", corpus)
        _write_jsonl(staging / "questions.jsonl", questions)
        split_dir = staging / "splits"
        split_dir.mkdir()
        split_counts: dict[str, int] = {}
        for split in ("train", "validation", "test"):
            rows = [row for row in questions if row["split"] == split]
            split_counts[split] = len(rows)
            (split_dir / f"{split}.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        manifest = {
            "schema_version": 1,
            "name": "docsqa-unified-validated",
            "description": "Four public documentation corpora with structurally validated accepted-answer QA cases.",
            "document_id_format": "<project>::<canonical-source-document-id>",
            "question_id_format": "<project>::<source-question-id>",
            "source_validation_sha256": _sha256(eligible_path),
            "documents": len(corpus),
            "questions": len(questions),
            "qrels": sum(int(row["qrel_count"]) for row in questions),
            "splits": split_counts,
            "question_count_by_project": dict(
                sorted(Counter(str(row["project"]) for row in questions).items())
            ),
            "projects": project_summaries,
            "routing_indexes_in_corpus": False,
            "reference_answers_in_corpus": False,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging.replace(output_dir)
        completed = True
        return manifest
    finally:
        if not completed and staging.exists():
            shutil.rmtree(staging)


def _dataset_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("dataset must be PROJECT=DATASET_DIR")
    project, path = value.split("=", 1)
    return project, Path(path)


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=_dataset_arg, default=[])
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "evaluation/dataset/templates/public_sources.json",
    )
    parser.add_argument(
        "--eligible",
        type=Path,
        default=project_root
        / "results/runs/dataset-analysis/source-validation/structurally_eligible.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "evaluation/dataset/evaluation_data/combined",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    defaults = {
        "github-docs": project_root / "evaluation/dataset/evaluation_data/projects/github-docs",
        "tailwind-css": project_root / "evaluation/dataset/evaluation_data/projects/tailwind-css",
        "prisma": project_root / "evaluation/dataset/evaluation_data/projects/prisma",
        "supabase": project_root / "evaluation/dataset/evaluation_data/projects/supabase",
    }
    dataset_dirs = dict(args.dataset) if args.dataset else defaults
    manifest = combine_datasets(
        dataset_dirs,
        load_source_specs(args.config),
        args.eligible,
        args.output_dir,
        force=args.force,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
