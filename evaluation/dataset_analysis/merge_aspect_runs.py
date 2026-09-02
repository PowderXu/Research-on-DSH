"""Merge validated aspect JSONL runs into one deterministic artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import string
from pathlib import Path
from typing import Any, Iterable

from .question_partitions import load_question_partition


SCHEMA_VERSION = "docsqa-merged-aspects-v1"
ASPECTS_FILENAME = "aspects.jsonl"
PROVENANCE_FILENAME = "merge_provenance.json"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _validate_rubric_sha256(value: Any, *, source: str) -> str:
    digest = str(value or "")
    if len(digest) != 64 or any(character not in string.hexdigits for character in digest):
        raise ValueError(f"{source} has an invalid rubric_sha256")
    return digest.lower()


def _load_input(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    locations: dict[str, int] = {}
    rubric_pairs: set[tuple[str, str]] = set()
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} is not valid JSON: {error.msg}") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        question_id = str(row.get("question_id") or "")
        if not question_id:
            raise ValueError(f"{path}:{line_number} has an empty question_id")
        if question_id in locations:
            raise ValueError(
                f"duplicate question_id {question_id!r} in {path}:"
                f"{locations[question_id]} and {line_number}"
            )
        rubric_version = str(row.get("rubric_version") or "")
        if not rubric_version:
            raise ValueError(f"{path}:{line_number} has an empty rubric_version")
        rubric_sha256 = _validate_rubric_sha256(
            row.get("rubric_sha256"), source=f"{path}:{line_number}"
        )
        if row.get("status") != "accepted":
            raise ValueError(f"{path}:{line_number} is not an accepted aspect record")
        # Normalize hexadecimal case so semantically identical digests serialize identically.
        row["rubric_sha256"] = rubric_sha256
        rows.append(row)
        locations[question_id] = line_number
        rubric_pairs.add((rubric_version, rubric_sha256))
    if not rows:
        raise ValueError(f"aspect input is empty: {path}")
    if len(rubric_pairs) != 1:
        raise ValueError(f"aspect input has inconsistent rubric metadata: {path}")
    rubric_version, rubric_sha256 = next(iter(rubric_pairs))
    question_ids = sorted(locations)
    return rows, {
        "path": path.as_posix(),
        "sha256": _sha256_bytes(raw),
        "records": len(rows),
        "question_ids_sha256": _canonical_sha256(question_ids),
        "rubric_version": rubric_version,
        "rubric_sha256": rubric_sha256,
    }


def _validate_unique_inputs(paths: Iterable[Path]) -> list[Path]:
    ordered = sorted(paths, key=lambda path: path.as_posix())
    if not ordered:
        raise ValueError("at least one aspect input is required")
    resolved: dict[Path, Path] = {}
    for path in ordered:
        identity = path.resolve()
        if identity in resolved:
            raise ValueError(
                f"aspect input was provided more than once: {resolved[identity]} and {path}"
            )
        resolved[identity] = path
    return ordered


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(
        _canonical_bytes(row) + b"\n"
        for row in sorted(rows, key=lambda row: str(row["question_id"]))
    )


def _atomic_write(path: Path, value: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


def merge_aspect_runs(
    inputs: Iterable[Path],
    *,
    output_dir: Path,
    question_manifest: Path | None = None,
) -> dict[str, Any]:
    """Validate and merge aspect files, then return the written provenance record."""

    input_paths = _validate_unique_inputs(inputs)
    all_rows: list[dict[str, Any]] = []
    input_provenance: list[dict[str, Any]] = []
    seen_at: dict[str, str] = {}
    rubric_pairs: set[tuple[str, str]] = set()
    for path in input_paths:
        rows, provenance = _load_input(path)
        for row in rows:
            question_id = str(row["question_id"])
            previous = seen_at.get(question_id)
            if previous is not None:
                raise ValueError(
                    f"duplicate question_id {question_id!r} across {previous} and {path}"
                )
            seen_at[question_id] = path.as_posix()
        all_rows.extend(rows)
        input_provenance.append(provenance)
        rubric_pairs.add(
            (provenance["rubric_version"], provenance["rubric_sha256"])
        )
    if len(rubric_pairs) != 1:
        details = sorted(f"{version}:{digest}" for version, digest in rubric_pairs)
        raise ValueError(f"aspect inputs use inconsistent rubric metadata: {details}")
    rubric_version, rubric_sha256 = next(iter(rubric_pairs))
    question_ids = sorted(seen_at)

    expected_union: dict[str, Any] | None = None
    if question_manifest is not None:
        _, expected_union = load_question_partition(
            question_manifest,
            available_question_ids=question_ids,
            partition="all",
            allow_all=True,
        )

    artifact = _jsonl_bytes(all_rows)
    semantic_identity = {
        "schema_version": SCHEMA_VERSION,
        "rubric_version": rubric_version,
        "rubric_sha256": rubric_sha256,
        "records": len(all_rows),
        "question_ids_sha256": _canonical_sha256(question_ids),
        "aspects_jsonl_sha256": _sha256_bytes(artifact),
        "input_sha256s": sorted(row["sha256"] for row in input_provenance),
        "question_manifest_sha256": (
            expected_union["manifest_sha256"] if expected_union is not None else None
        ),
    }
    provenance: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "rubric_version": rubric_version,
        "rubric_sha256": rubric_sha256,
        "records": len(all_rows),
        "question_ids_sha256": semantic_identity["question_ids_sha256"],
        "aspects_jsonl_sha256": semantic_identity["aspects_jsonl_sha256"],
        "merge_identity_sha256": _canonical_sha256(semantic_identity),
        "inputs": input_provenance,
        "expected_question_union": expected_union,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(output_dir / ASPECTS_FILENAME, artifact)
    _atomic_write(
        output_dir / PROVENANCE_FILENAME,
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n",
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="inputs", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--question-manifest", type=Path)
    args = parser.parse_args()
    report = merge_aspect_runs(
        args.inputs,
        output_dir=args.output_dir,
        question_manifest=args.question_manifest,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
