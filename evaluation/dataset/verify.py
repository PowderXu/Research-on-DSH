#!/usr/bin/env python3
"""Verify bundled benchmark hashes, counts, and forbidden oracle fields."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


DATASET_ROOT = Path(__file__).resolve().parent
DATA = DATASET_ROOT / "data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((DATA / "dataset_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = DATASET_ROOT / relative
        if path.stat().st_size != expected["bytes"]:
            raise SystemExit(f"size mismatch: {relative}")
        if sha256(path) != expected["sha256"]:
            raise SystemExit(f"hash mismatch: {relative}")

    questions = [
        json.loads(line)
        for line in (DATA / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(questions) != manifest["questions"]:
        raise SystemExit("question count mismatch")
    if sum(row["qrel_count"] for row in questions) != manifest["qrels"]:
        raise SystemExit("qrel count mismatch")
    if any("graph_opportunity" in row for row in questions):
        raise SystemExit("forbidden graph_opportunity oracle field is present")
    print(
        f"verified {manifest['documents']} documents, "
        f"{manifest['questions']} questions, and {manifest['qrels']} qrels"
    )


if __name__ == "__main__":
    main()
