#!/usr/bin/env python3
"""Freeze a leakage-safe GitHub Docs split for DSH SkillOpt experiments.

The source benchmark already has an 82-question development split and a
246-question held-out test split.  SkillOpt needs separate train and selection
sets, so this script deterministically partitions *only* the development rows.
The original test rows are copied byte-for-data and are never sampled into the
optimization pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SEED = 20260821
DEFAULT_TRAIN_SIZE = 55


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_order_key(row: dict[str, Any], seed: int) -> tuple[str, str]:
    item_id = str(row["question_id"])
    digest = hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()
    return digest, item_id


def _allocate_train_counts(
    strata: dict[str, list[dict[str, Any]]], train_size: int
) -> dict[str, int]:
    total = sum(len(rows) for rows in strata.values())
    if not 0 < train_size < total:
        raise ValueError(f"train_size must be between 1 and {total - 1}")

    exact = {
        name: train_size * len(rows) / total for name, rows in strata.items()
    }
    allocated = {name: int(value) for name, value in exact.items()}
    remaining = train_size - sum(allocated.values())
    remainder_order = sorted(
        strata,
        key=lambda name: (exact[name] - allocated[name], len(strata[name]), name),
        reverse=True,
    )
    for name in remainder_order[:remaining]:
        allocated[name] += 1

    # Every non-trivial evidence type must remain represented in both sets.
    for name, rows in strata.items():
        if len(rows) >= 2 and not 0 < allocated[name] < len(rows):
            raise ValueError(
                f"stratum {name!r} cannot be represented in train and val: "
                f"size={len(rows)}, allocated={allocated[name]}"
            )
    return allocated


def build_split(
    rows: list[dict[str, Any]], *, seed: int, train_size: int
) -> dict[str, list[dict[str, Any]]]:
    dev = [row for row in rows if row.get("split") == "dev"]
    test = [row for row in rows if row.get("split") == "test"]
    unknown = sorted({str(row.get("split")) for row in rows} - {"dev", "test"})
    if unknown:
        raise ValueError(f"unexpected source split values: {unknown}")
    if not dev or not test:
        raise ValueError("source benchmark must contain non-empty dev and test splits")

    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dev:
        stratum = str(row.get("evidence_category") or "unknown")
        strata[stratum].append(row)
    train_counts = _allocate_train_counts(strata, train_size)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for name in sorted(strata):
        ordered = sorted(strata[name], key=lambda row: _stable_order_key(row, seed))
        train.extend(ordered[: train_counts[name]])
        val.extend(ordered[train_counts[name] :])

    train.sort(key=lambda row: _stable_order_key(row, seed))
    val.sort(key=lambda row: _stable_order_key(row, seed))
    test.sort(key=lambda row: str(row["question_id"]))

    def annotate(selected: list[dict[str, Any]], skillopt_split: str) -> list[dict[str, Any]]:
        output = []
        for row in selected:
            copied = dict(row)
            copied["source_split"] = copied.pop("split")
            copied["split"] = skillopt_split
            output.append(copied)
        return output

    split = {
        "train": annotate(train, "train"),
        "val": annotate(val, "val"),
        "test": annotate(test, "test"),
    }
    ids = {
        name: {str(row["question_id"]) for row in items}
        for name, items in split.items()
    }
    if any(ids[left] & ids[right] for left, right in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise AssertionError("materialized split contains overlapping question IDs")
    return split


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(field)) for row in rows).items()))


def materialize(
    source: Path, output_dir: Path, *, seed: int, train_size: int
) -> dict[str, Any]:
    source = source.resolve()
    output_dir = output_dir.resolve()
    split = build_split(_read_jsonl(source), seed=seed, train_size=train_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, rows in split.items():
        split_dir = output_dir / name
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "items.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "benchmark": "GitHub Docs real support questions v2 / DSH SkillOpt",
        "source_questions": str(source),
        "source_questions_sha256": _sha256(source),
        "algorithm": "evidence-category stratified stable SHA-256 ordering",
        "seed": seed,
        "leakage_policy": (
            "Only source_split=dev is partitioned into train/val; the original "
            "source_split=test set is copied without optimization or selection."
        ),
        "counts": {name: len(rows) for name, rows in split.items()},
        "evidence_category_counts": {
            name: _counts(rows, "evidence_category") for name, rows in split.items()
        },
        "intent_category_counts": {
            name: _counts(rows, "intent_category") for name, rows in split.items()
        },
        "evidence_structure_counts": {
            name: _counts(rows, "evidence_structure") for name, rows in split.items()
        },
        "qrel_count_distribution": {
            name: _counts(rows, "qrel_count") for name, rows in split.items()
        },
        "question_ids_sha256": {
            name: hashlib.sha256(
                "\n".join(str(row["question_id"]) for row in rows).encode()
            ).hexdigest()
            for name, rows in split.items()
        },
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("evaluation/github_docs_v2/questions.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/skillopt/github_docs_v2_split"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    args = parser.parse_args()
    manifest = materialize(
        args.source, args.output_dir, seed=args.seed, train_size=args.train_size
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
