from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_analysis.question_partitions import (
    EXPECTED_SCHEMA,
    load_question_partition,
)


def _write(path: Path, partitions: dict[str, list[str]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": EXPECTED_SCHEMA,
                "name": "test",
                "partitions": partitions,
            }
        ),
        encoding="utf-8",
    )


def test_manifest_selection_is_independent_of_legacy_split(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    partitions = {
        "calibration_train": ["q1", "q2", "q3"],
        "calibration_validation": ["q4"],
        "final_test": ["q5"],
    }
    _write(path, partitions)
    selected, metadata = load_question_partition(
        path,
        available_question_ids=["q5", "q1", "q4", "q3", "q2"],
        partition="calibration_validation",
        allow_all=False,
    )
    assert selected == {"q4"}
    assert metadata["selected_question_count"] == 1
    assert len(metadata["selected_question_ids_sha256"]) == 64


@pytest.mark.parametrize(
    "partitions,match",
    [
        (
            {
                "calibration_train": ["q1", "q1"],
                "calibration_validation": ["q2"],
                "final_test": ["q3"],
            },
            "duplicate",
        ),
        (
            {
                "calibration_train": ["q1"],
                "calibration_validation": ["q1"],
                "final_test": ["q3"],
            },
            "overlap",
        ),
        (
            {
                "calibration_train": ["q1"],
                "calibration_validation": ["q2"],
                "final_test": ["unknown"],
            },
            "does not exactly cover",
        ),
    ],
)
def test_manifest_rejects_invalid_partitions(
    tmp_path: Path, partitions: dict[str, list[str]], match: str
) -> None:
    path = tmp_path / "manifest.json"
    _write(path, partitions)
    with pytest.raises(ValueError, match=match):
        load_question_partition(
            path,
            available_question_ids=["q1", "q2", "q3"],
            partition="calibration_train",
            allow_all=False,
        )


def test_all_partition_requires_explicit_permission(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    _write(
        path,
        {
            "calibration_train": ["q1"],
            "calibration_validation": ["q2"],
            "final_test": ["q3"],
        },
    )
    with pytest.raises(ValueError, match="not allowed"):
        load_question_partition(
            path,
            available_question_ids=["q1", "q2", "q3"],
            partition="all",
            allow_all=False,
        )
