"""Export a portable, versionable dataset repository without model calls."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from .records import RECORD_LAYOUT, load_records, read_jsonl


def identity(path: Path) -> dict:
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def export_dataset(source: Path, destination: Path, sources: Path, aspects: Path | None = None) -> dict:
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"export destination must be empty: {destination}")
    records = load_records(source)
    corpus = read_jsonl(source / "corpus.jsonl")
    doc_ids = {row["doc_id"] for row in corpus}
    if len(doc_ids) != len(corpus):
        raise ValueError("duplicate corpus IDs")
    if any(set(row["qrel_ids"]) - doc_ids for row in records):
        raise ValueError("reference document is absent from corpus")
    data = destination / "data"
    data.mkdir(parents=True)
    files, downloads = {}, {}
    for name in ("questions.jsonl", "answers.jsonl", "corpus.jsonl", "image_text.jsonl"):
        original = source / name
        files[name] = identity(original)
        stored = data / (name + ".gz" if name == "corpus.jsonl" else name)
        if name == "corpus.jsonl":
            with stored.open("wb") as output:
                with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
                    with original.open("rb") as input_file:
                        shutil.copyfileobj(input_file, compressed)
        else:
            shutil.copyfile(original, stored)
        downloads[name] = {
            "path": "data/" + stored.name,
            "compression": "gzip" if name == "corpus.jsonl" else "none",
            **identity(stored),
        }
    if aspects is not None:
        rows = read_jsonl(aspects)
        if len(rows) != len(records) or {row["question_id"] for row in rows} != {row["question_id"] for row in records}:
            raise ValueError("frozen aspects must match the complete question pool")
        by_id = {row["question_id"]: row for row in records}
        for row in rows:
            canonical = by_id[row["question_id"]]
            if row["reference_answer"] != canonical["normalized_answer"]:
                raise ValueError("aspect reference answer differs from the canonical answer")
            question = canonical["query"]
            if row["question"] != question and not (
                question.startswith(row["question"])
                and "Image-derived question evidence:" in question[len(row["question"]):]
            ):
                raise ValueError("aspect question differs from the canonical question")
        # The evaluator joins question/reference text by ID at runtime.
        rows = [{k: v for k, v in row.items() if k not in {"question", "reference_answer", "split", "benchmark_split"}} for row in rows]
        stored = data / "aspects.jsonl"
        stored.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
        files[stored.name] = identity(stored)
        downloads[stored.name] = {"path": "data/" + stored.name, "compression": "none", **files[stored.name]}
    source_manifest = json.loads((source / "manifest.json").read_text())
    manifest = {
        "schema_version": 3,
        "distribution_version": 1,
        "name": "docsqa-data",
        "record_layout": RECORD_LAYOUT,
        "partitioning": "none",
        "questions": len(records),
        "accepted_questions": len(records),
        "answers": len(records),
        "documents": len(corpus),
        "qrels": sum(len(set(row["qrel_ids"])) for row in records),
        "question_count_by_project": dict(sorted(Counter(row["project"] for row in records).items())),
        "model": source_manifest["model"],
        "normalizer_version": source_manifest["normalizer_version"],
        "rubric_version": source_manifest["rubric_version"],
        "image_text_version": source_manifest["image_text_version"],
        "image_policy": source_manifest["image_policy"],
        "reference_answers_in_corpus": False,
        "files": files,
        "downloads": downloads,
        "sources": json.loads(sources.read_text())["sources"],
    }
    (data / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    shutil.copyfile(sources, destination / "sources.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--aspects", type=Path)
    args = parser.parse_args()
    manifest = export_dataset(args.source, args.destination, args.sources, args.aspects)
    print(json.dumps({key: manifest[key] for key in ("questions", "answers", "documents", "qrels")}))


if __name__ == "__main__":
    main()
