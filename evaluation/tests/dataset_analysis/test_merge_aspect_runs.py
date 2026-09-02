from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dataset_analysis.merge_aspect_runs import (
    ASPECTS_FILENAME,
    PROVENANCE_FILENAME,
    merge_aspect_runs,
)
from dataset_analysis.question_partitions import EXPECTED_SCHEMA


RUBRIC_HASH = "a" * 64


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(
    question_id: str,
    *,
    rubric_version: str = "rubric-v2",
    rubric_sha256: str = RUBRIC_HASH,
    status: str = "accepted",
) -> dict:
    return {
        "question_id": question_id,
        "project": question_id.split("::", 1)[0],
        "status": status,
        "rubric_version": rubric_version,
        "rubric_sha256": rubric_sha256,
        "aspects": [{"aspect_id": "a1", "description": question_id}],
    }


def _manifest(path: Path, ids: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": EXPECTED_SCHEMA,
                "name": "test-60-20-20",
                "partitions": {
                    "calibration_train": ids[:2],
                    "calibration_validation": ids[2:3],
                    "final_test": ids[3:],
                },
            }
        ),
        encoding="utf-8",
    )


def test_merge_is_deterministic_and_records_provenance(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_jsonl(first, [_row("b::2"), _row("a::1")])
    _write_jsonl(second, [_row("c::4"), _row("b::3")])
    manifest = tmp_path / "splits.json"
    _manifest(manifest, ["a::1", "b::2", "b::3", "c::4"])

    output_one = tmp_path / "merged-one"
    output_two = tmp_path / "merged-two"
    report_one = merge_aspect_runs(
        [second, first], output_dir=output_one, question_manifest=manifest
    )
    report_two = merge_aspect_runs(
        [first, second], output_dir=output_two, question_manifest=manifest
    )

    artifact_one = (output_one / ASPECTS_FILENAME).read_bytes()
    artifact_two = (output_two / ASPECTS_FILENAME).read_bytes()
    assert artifact_one == artifact_two
    assert [
        json.loads(line)["question_id"]
        for line in artifact_one.decode("utf-8").splitlines()
    ] == ["a::1", "b::2", "b::3", "c::4"]
    assert report_one == report_two
    assert report_one["records"] == 4
    assert report_one["rubric_version"] == "rubric-v2"
    assert report_one["rubric_sha256"] == RUBRIC_HASH
    assert report_one["aspects_jsonl_sha256"] == hashlib.sha256(artifact_one).hexdigest()
    assert report_one["expected_question_union"]["selected_question_count"] == 4
    assert [row["path"] for row in report_one["inputs"]] == sorted(
        [first.as_posix(), second.as_posix()]
    )
    assert json.loads(
        (output_one / PROVENANCE_FILENAME).read_text(encoding="utf-8")
    ) == report_one


@pytest.mark.parametrize(
    "changed,match",
    [
        ({"rubric_version": "rubric-v3"}, "inconsistent rubric"),
        ({"rubric_sha256": "b" * 64}, "inconsistent rubric"),
    ],
)
def test_merge_rejects_inconsistent_rubrics(
    tmp_path: Path, changed: dict[str, str], match: str
) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_jsonl(first, [_row("a::1")])
    _write_jsonl(second, [{**_row("b::2"), **changed}])
    with pytest.raises(ValueError, match=match):
        merge_aspect_runs([first, second], output_dir=tmp_path / "merged")


def test_merge_rejects_duplicate_question_ids(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_jsonl(first, [_row("a::1")])
    _write_jsonl(second, [_row("a::1")])
    with pytest.raises(ValueError, match="duplicate question_id"):
        merge_aspect_runs([first, second], output_dir=tmp_path / "merged")


def test_merge_requires_exact_manifest_union(tmp_path: Path) -> None:
    source = tmp_path / "aspects.jsonl"
    _write_jsonl(source, [_row("a::1"), _row("b::2"), _row("b::3")])
    manifest = tmp_path / "splits.json"
    _manifest(manifest, ["a::1", "b::2", "b::3", "c::4"])
    with pytest.raises(ValueError, match="does not exactly cover"):
        merge_aspect_runs(
            [source],
            output_dir=tmp_path / "merged",
            question_manifest=manifest,
        )
    assert not (tmp_path / "merged").exists()


def test_merge_rejects_nonaccepted_rows(tmp_path: Path) -> None:
    source = tmp_path / "aspects.jsonl"
    _write_jsonl(source, [_row("a::1", status="rejected")])
    with pytest.raises(ValueError, match="not an accepted aspect record"):
        merge_aspect_runs([source], output_dir=tmp_path / "merged")
