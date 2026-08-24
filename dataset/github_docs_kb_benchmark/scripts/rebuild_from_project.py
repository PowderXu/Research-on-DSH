#!/usr/bin/env python3
"""Build the portable GitHub Docs KB dataset from project evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DATA = PROJECT_ROOT / "data" / "build_input" / "github_docs_v2"
SOURCE_SPLITS = PROJECT_ROOT / "data" / "build_input" / "github_docs_v2_split"
OUTPUT_DATA = PACKAGE_ROOT / "data"
SOURCE_COMMIT = "c34e3dccad00f61133c799d20e7d1208a0e6cc92"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence_structure(row: dict[str, Any]) -> str:
    qrel_count = len(set(row.get("qrel_ids") or []))
    if qrel_count <= 1:
        return "single"
    category = row.get("evidence_category")
    if category == "multi_page_linked":
        return "linked"
    return "dispersed"


def clean_question(row: dict[str, Any]) -> dict[str, Any]:
    cleaned = {key: value for key, value in row.items() if key != "graph_opportunity"}
    cleaned["qrel_count"] = len(set(cleaned.get("qrel_ids") or []))
    cleaned["evidence_structure"] = evidence_structure(cleaned)
    return cleaned


def counter_dict(rows: Iterable[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def qrel_distribution(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row["qrel_count"]) for row in rows).items(), key=lambda item: int(item[0])))


def write_questions() -> list[dict[str, Any]]:
    rows = [clean_question(row) for row in read_jsonl(SOURCE_DATA / "questions.jsonl")]
    target = OUTPUT_DATA / "questions.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def write_splits() -> dict[str, list[dict[str, Any]]]:
    split_dir = OUTPUT_DATA / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    source_names = {"train": "train", "validation": "val", "test": "test"}
    splits: dict[str, list[dict[str, Any]]] = {}
    for output_name, source_name in source_names.items():
        source_path = SOURCE_SPLITS / source_name / "items.json"
        rows = [clean_question(row) for row in json.loads(source_path.read_text(encoding="utf-8"))]
        splits[output_name] = rows
        write_json(split_dir / f"{output_name}.json", rows)

    ids = {
        name: [str(row["question_id"]) for row in rows]
        for name, rows in splits.items()
    }
    manifest = {
        "algorithm": "evidence-category stratified stable SHA-256 ordering",
        "seed": 20260821,
        "leakage_policy": (
            "Only the original source split 'dev' is partitioned into train/validation; "
            "the original source split 'test' remains untouched."
        ),
        "counts": {name: len(rows) for name, rows in splits.items()},
        "evidence_category_counts": {
            name: counter_dict(rows, "evidence_category") for name, rows in splits.items()
        },
        "intent_category_counts": {
            name: counter_dict(rows, "intent_category") for name, rows in splits.items()
        },
        "evidence_structure_counts": {
            name: counter_dict(rows, "evidence_structure") for name, rows in splits.items()
        },
        "qrel_count_distribution": {
            name: qrel_distribution(rows) for name, rows in splits.items()
        },
        "question_ids_sha256": {
            name: hashlib.sha256("\n".join(values).encode()).hexdigest()
            for name, values in ids.items()
        },
        "oracle_features_excluded": ["graph_opportunity"],
    }
    write_json(split_dir / "split_manifest.json", manifest)
    return splits


def file_metadata(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        result[relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    return result


def main() -> None:
    required = [
        SOURCE_DATA / "questions.jsonl",
        SOURCE_DATA / "corpus.jsonl",
        SOURCE_DATA / "graph_schema_v2.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing source artifacts: {missing}")

    OUTPUT_DATA.mkdir(parents=True, exist_ok=True)
    questions = write_questions()
    splits = write_splits()
    shutil.copyfile(SOURCE_DATA / "corpus.jsonl", OUTPUT_DATA / "corpus.jsonl")
    shutil.copyfile(SOURCE_DATA / "graph_schema_v2.json", OUTPUT_DATA / "graph_schema.json")

    source_manifest = {
        "documentation": {
            "repository": "https://github.com/github/docs.git",
            "commit": SOURCE_COMMIT,
            "content_root": "content/",
            "license": "CC-BY-4.0",
            "license_url": "https://github.com/github/docs/blob/main/LICENSE"
        },
        "questions": {
            "source": "https://github.com/orgs/community/discussions",
            "selection": "public discussions with an accepted answer containing at least one resolvable docs.github.com URL",
            "qrel_provenance": "distinct docs.github.com URLs in the accepted answer only",
            "query_leakage_control": "all visible HTTP(S) URLs removed from query text"
        },
        "generated_on": "2026-08-21"
    }
    write_json(OUTPUT_DATA / "source_manifest.json", source_manifest)

    manifest_paths = [
        OUTPUT_DATA / "corpus.jsonl",
        OUTPUT_DATA / "questions.jsonl",
        OUTPUT_DATA / "graph_schema.json",
        OUTPUT_DATA / "source_manifest.json",
        OUTPUT_DATA / "splits" / "train.json",
        OUTPUT_DATA / "splits" / "validation.json",
        OUTPUT_DATA / "splits" / "test.json",
        OUTPUT_DATA / "splits" / "split_manifest.json",
    ]
    corpus_documents = sum(1 for _ in read_jsonl(OUTPUT_DATA / "corpus.jsonl"))
    qrels_total = sum(row["qrel_count"] for row in questions)
    dataset_manifest = {
        "name": "GitHub Docs KB retrieval benchmark",
        "version": "2.1",
        "documents": corpus_documents,
        "questions": len(questions),
        "qrels": qrels_total,
        "corpus_revision": SOURCE_COMMIT,
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "intent_category_counts": counter_dict(questions, "intent_category"),
        "evidence_category_counts": counter_dict(questions, "evidence_category"),
        "evidence_structure_counts": counter_dict(questions, "evidence_structure"),
        "qrel_count_distribution": qrel_distribution(questions),
        "evaluation_unit": "canonical documentation page",
        "qrel_provenance": "distinct docs.github.com URLs in accepted answers",
        "qrels_exhaustive": False,
        "schema_version": "v2",
        "oracle_features_excluded": ["graph_opportunity"],
        "files": file_metadata(manifest_paths),
    }
    write_json(OUTPUT_DATA / "dataset_manifest.json", dataset_manifest)
    print(f"Built {PACKAGE_ROOT} ({corpus_documents} documents, {len(questions)} questions, {qrels_total} qrels)")


if __name__ == "__main__":
    main()
