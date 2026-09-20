#!/usr/bin/env python3
"""Verify a generated DocsQA evaluation package."""

from __future__ import annotations

from dataset.scripts.records import PARTITION_FIELDS, load_records

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def verify_dataset(data_dir: Path) -> dict[str, int]:
    if (data_dir / "graph_schema.json").exists():
        raise ValueError("graph_schema.json belongs to the plugin, not evaluation data")
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path = data_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    for relative, expected in (manifest.get("files") or {}).items():
        path = data_dir / relative
        if path.stat().st_size != expected["bytes"]:
            raise ValueError(f"size mismatch: {relative}")
        if sha256(path) != expected["sha256"]:
            raise ValueError(f"hash mismatch: {relative}")

    corpus = _load_jsonl(data_dir / "corpus.jsonl")
    questions = load_records(data_dir)
    doc_ids = {str(row["doc_id"]) for row in corpus}
    question_ids = {str(row["question_id"]) for row in questions}
    if len(doc_ids) != len(corpus):
        raise ValueError("duplicate document IDs")
    if len(question_ids) != len(questions):
        raise ValueError("duplicate question IDs")
    if int(manifest["documents"]) != len(corpus):
        raise ValueError("document count mismatch")
    if int(manifest["questions"]) != len(questions):
        raise ValueError("question count mismatch")
    if any("graph_opportunity" in row for row in questions):
        raise ValueError("forbidden graph_opportunity oracle field is present")
    if any("reference_answer" in row or "expanded_reference_answer" in row for row in corpus):
        raise ValueError("reference-answer leakage is present in the searchable corpus")

    qrel_count = sum(len(set(map(str, row.get("qrel_ids") or []))) for row in questions)
    if int(manifest["qrels"]) != qrel_count:
        raise ValueError("qrel count mismatch")
    missing_qrels = sorted(
        {
            str(qrel)
            for row in questions
            for qrel in row.get("qrel_ids") or []
            if str(qrel) not in doc_ids
        }
    )
    if missing_qrels:
        raise ValueError(f"qrels missing from corpus: {missing_qrels[:5]}")

    if (data_dir / "splits").exists() or any(key in manifest for key in ("splits", "split_counts", "questions_by_split")):
        raise ValueError("evaluation package must not contain dataset partitions")
    if manifest.get("partitioning", "none") != "none" or any(set(row) & PARTITION_FIELDS for row in questions):
        raise ValueError("evaluation records must belong to one unpartitioned pool")

    return {
        "documents": len(corpus),
        "questions": len(questions),
        "qrels": qrel_count,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=project_root / "evaluation/dataset/evaluation_data/normalized",
    )
    args = parser.parse_args()
    counts = verify_dataset(args.dataset_dir)
    print(
        f"verified {counts['documents']} documents, "
        f"{counts['questions']} questions, and {counts['qrels']} qrels"
    )


if __name__ == "__main__":
    main()
