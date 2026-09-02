"""Validate and select IDs from a frozen weak-supervision partition manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


PARTITIONS = ("calibration_train", "calibration_validation", "final_test")
EXPECTED_SCHEMA = "docsqa-weak-supervision-splits-v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(payload)


def load_question_partition(
    path: Path,
    *,
    available_question_ids: Iterable[str],
    partition: str,
    allow_all: bool,
) -> tuple[set[str], dict[str, Any]]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema_version") != EXPECTED_SCHEMA:
        raise ValueError("unsupported weak-supervision manifest schema")
    partitions = payload.get("partitions")
    if not isinstance(partitions, dict) or set(partitions) != set(PARTITIONS):
        raise ValueError("weak-supervision manifest must define exactly three partitions")
    partition_ids: dict[str, list[str]] = {}
    seen: set[str] = set()
    for name in PARTITIONS:
        values = partitions.get(name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"weak-supervision partition {name} must be a non-empty list")
        ids = [str(value) for value in values]
        if not all(ids) or len(ids) != len(set(ids)):
            raise ValueError(f"weak-supervision partition {name} has duplicate or empty IDs")
        overlap = seen & set(ids)
        if overlap:
            raise ValueError(
                f"weak-supervision partitions overlap: {sorted(overlap)}"
            )
        seen.update(ids)
        partition_ids[name] = ids
    available = set(map(str, available_question_ids))
    if seen != available:
        raise ValueError(
            "weak-supervision manifest does not exactly cover available questions: "
            f"missing={sorted(available - seen)} unknown={sorted(seen - available)}"
        )
    if partition == "all":
        if not allow_all:
            raise ValueError("partition 'all' is not allowed for this operation")
        selected = seen
    elif partition in partition_ids:
        selected = set(partition_ids[partition])
    else:
        raise ValueError(f"unknown weak-supervision partition: {partition}")
    metadata = {
        "selection_mode": "weak_supervision_manifest",
        "manifest_path": str(path),
        "manifest_sha256": _sha256_bytes(raw),
        "manifest_canonical_sha256": _canonical_sha256(payload),
        "manifest_schema_version": payload["schema_version"],
        "manifest_name": str(payload.get("name") or ""),
        "manifest_partition": partition,
        "manifest_partition_counts": {
            name: len(partition_ids[name]) for name in PARTITIONS
        },
        "selected_question_count": len(selected),
        "selected_question_ids_sha256": _canonical_sha256(sorted(selected)),
    }
    return selected, metadata
