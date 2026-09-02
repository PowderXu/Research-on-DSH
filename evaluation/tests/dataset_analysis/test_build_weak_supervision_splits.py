from __future__ import annotations

from dataset_analysis.build_weak_supervision_splits import build_splits


def _row(project: str, index: int, query: str | None = None) -> dict[str, str]:
    return {
        "question_id": f"{project}::{index}",
        "project": project,
        "query": query or f"question {project} {index}",
        "source_question_id": str(index),
        "source_url": f"https://example.test/{project}/{index}",
        "accepted_answer_url": f"https://example.test/{project}/{index}#answer",
        "split": "legacy",
    }


def test_split_is_exact_deterministic_and_project_stratified() -> None:
    rows = [_row("a", index) for index in range(7)] + [
        _row("b", index) for index in range(8)
    ]
    first, report = build_splits(rows, seed=17)
    second, _ = build_splits(list(reversed(rows)), seed=17)
    assert first == second
    assert report["counts"] == {
        "calibration_train": 9,
        "calibration_validation": 3,
        "final_test": 3,
    }
    assert report["project_counts"] == {
        "a": {
            "calibration_train": 4,
            "calibration_validation": 1,
            "final_test": 2,
        },
        "b": {
            "calibration_train": 5,
            "calibration_validation": 2,
            "final_test": 1,
        },
    }
    ids = [row["question_id"] for values in first.values() for row in values]
    assert len(ids) == len(set(ids)) == 15


def test_duplicate_group_never_crosses_splits() -> None:
    rows = [_row("a", index) for index in range(10)]
    rows[1]["query"] = rows[0]["query"]
    splits, report = build_splits(rows, seed=5)
    assigned = {
        row["question_id"]: split
        for split, values in splits.items()
        for row in values
    }
    assert assigned["a::0"] == assigned["a::1"]
    assert report["multi_record_groups"] == 1
