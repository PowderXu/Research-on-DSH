"""Build grouped, project-stratified weak-supervision train/validation/test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from .audit_integrity import normalize_identifier, normalize_url


SCHEMA_VERSION = "docsqa-weak-supervision-splits-v1"
SPLITS = ("calibration_train", "calibration_validation", "final_test")
RATIOS = {
    "calibration_train": 0.60,
    "calibration_validation": 0.20,
    "final_test": 0.20,
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def _normalized_query(value: Any) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", str(value or "")).casefold().split()
    )


class _UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    ids = [str(row.get("question_id") or "") for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("questions must have unique, non-empty question_id values")
    union = _UnionFind(ids)
    seen: dict[tuple[str, str, str], str] = {}
    for row in rows:
        question_id = str(row["question_id"])
        project = normalize_identifier(row.get("project"))
        keys = {
            (project, "query", _normalized_query(row.get("query"))),
            (
                project,
                "source_question_id",
                normalize_identifier(row.get("source_question_id")),
            ),
            (
                project,
                "source_url",
                normalize_url(row.get("source_url"), keep_fragment=False),
            ),
            (
                project,
                "accepted_answer_url",
                normalize_url(row.get("accepted_answer_url"), keep_fragment=True),
            ),
        }
        for key in keys:
            if not key[2]:
                continue
            previous = seen.setdefault(key, question_id)
            union.union(previous, question_id)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[union.find(str(row["question_id"]))].append(row)
    output = []
    for group in grouped.values():
        projects = {normalize_identifier(row.get("project")) for row in group}
        if len(projects) != 1:
            raise ValueError("a duplicate group crosses project boundaries")
        output.append(sorted(group, key=lambda row: str(row["question_id"])))
    return output


def _apportion(total: int) -> dict[str, int]:
    raw = {split: total * RATIOS[split] for split in SPLITS}
    counts = {split: math.floor(raw[split]) for split in SPLITS}
    remaining = total - sum(counts.values())
    tie_priority = {"final_test": 0, "calibration_validation": 1, "calibration_train": 2}
    order = sorted(
        SPLITS,
        key=lambda split: (-(raw[split] - counts[split]), tie_priority[split]),
    )
    for split in order[:remaining]:
        counts[split] += 1
    return counts


def _project_quotas(project_counts: dict[str, int]) -> dict[str, dict[str, int]]:
    global_targets = _apportion(sum(project_counts.values()))
    quotas = {
        project: {
            split: math.floor(count * RATIOS[split]) for split in SPLITS
        }
        for project, count in project_counts.items()
    }
    row_deficits = {
        project: project_counts[project] - sum(quotas[project].values())
        for project in project_counts
    }
    column_deficits = {
        split: global_targets[split]
        - sum(quotas[project][split] for project in project_counts)
        for split in SPLITS
    }
    split_priority = {
        "calibration_validation": 0,
        "final_test": 1,
        "calibration_train": 2,
    }
    while sum(row_deficits.values()):
        candidates = []
        for project, row_remaining in row_deficits.items():
            if row_remaining <= 0:
                continue
            for split, column_remaining in column_deficits.items():
                if column_remaining <= 0:
                    continue
                fraction = project_counts[project] * RATIOS[split] - quotas[project][split]
                candidates.append(
                    (-fraction, split_priority[split], project, split)
                )
        if not candidates:
            raise ValueError("could not apportion project-stratified split quotas")
        _, _, project, split = min(candidates)
        quotas[project][split] += 1
        row_deficits[project] -= 1
        column_deficits[split] -= 1
    if any(column_deficits.values()):
        raise ValueError("project quotas do not match global split targets")
    return quotas


def _assign_project_groups(
    groups: list[list[dict[str, Any]]],
    quotas: dict[str, int],
    *,
    seed: int,
    project: str,
) -> dict[str, str]:
    ordered = sorted(
        groups,
        key=lambda group: _stable_hash(
            project + "\0" + "\0".join(str(row["question_id"]) for row in group),
            seed,
        ),
    )
    prefix = [0]
    for group in ordered:
        prefix.append(prefix[-1] + len(group))
    decisions: dict[tuple[int, int, int], str] = {}

    @lru_cache(maxsize=None)
    def feasible(index: int, train_count: int, validation_count: int) -> bool:
        test_count = prefix[index] - train_count - validation_count
        if (
            train_count > quotas["calibration_train"]
            or validation_count > quotas["calibration_validation"]
            or test_count > quotas["final_test"]
        ):
            return False
        if index == len(ordered):
            return (
                train_count == quotas["calibration_train"]
                and validation_count == quotas["calibration_validation"]
                and test_count == quotas["final_test"]
            )
        size = len(ordered[index])
        digest = int(
            _stable_hash(
                "\0".join(str(row["question_id"]) for row in ordered[index]), seed
            )[:8],
            16,
        )
        choices = list(SPLITS)
        choices = choices[digest % len(choices) :] + choices[: digest % len(choices)]
        for split in choices:
            next_train = train_count + (
                size if split == "calibration_train" else 0
            )
            next_validation = validation_count + (
                size if split == "calibration_validation" else 0
            )
            if feasible(index + 1, next_train, next_validation):
                decisions[(index, train_count, validation_count)] = split
                return True
        return False

    if not feasible(0, 0, 0):
        sizes = sorted((len(group) for group in ordered), reverse=True)
        raise ValueError(
            f"duplicate-group sizes cannot satisfy exact quotas for {project}: "
            f"quotas={quotas} group_sizes={sizes}"
        )
    assignments: dict[str, str] = {}
    train_count = validation_count = 0
    for index, group in enumerate(ordered):
        split = decisions[(index, train_count, validation_count)]
        for row in group:
            assignments[str(row["question_id"])] = split
        if split == "calibration_train":
            train_count += len(group)
        elif split == "calibration_validation":
            validation_count += len(group)
    return assignments


def build_splits(rows: list[dict[str, Any]], seed: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    groups = _groups(rows)
    project_counts = Counter(str(row.get("project") or "") for row in rows)
    quotas = _project_quotas(dict(sorted(project_counts.items())))
    groups_by_project: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for group in groups:
        groups_by_project[str(group[0].get("project") or "")].append(group)
    assignment: dict[str, str] = {}
    for project, project_groups in sorted(groups_by_project.items()):
        assignment.update(
            _assign_project_groups(
                project_groups, quotas[project], seed=seed, project=project
            )
        )
    output = {split: [] for split in SPLITS}
    for row in rows:
        split = assignment[str(row["question_id"])]
        output[split].append(
            {
                "question_id": str(row["question_id"]),
                "project": str(row.get("project") or ""),
                "source_split": str(row.get("split") or ""),
            }
        )
    for split in SPLITS:
        output[split].sort(key=lambda row: str(row["question_id"]))
    report = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "ratios": RATIOS,
        "questions": len(rows),
        "groups": len(groups),
        "multi_record_groups": sum(len(group) > 1 for group in groups),
        "counts": {split: len(output[split]) for split in SPLITS},
        "project_counts": {
            project: {
                split: sum(row["project"] == project for row in output[split])
                for split in SPLITS
            }
            for project in sorted(project_counts)
        },
        "grouping_keys": [
            "project + normalized exact question",
            "project + normalized source_question_id",
            "project + normalized source_url",
            "project + normalized accepted_answer_url",
        ],
    }
    return output, report


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = _jsonl(args.questions)
    splits, report = build_splits(rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.update(
        {
            "name": "docsqa-weak-supervision-v1",
            "questions_path": str(args.questions),
            "questions_sha256": _sha256(args.questions),
            "partitions": {
                split: [str(row["question_id"]) for row in splits[split]]
                for split in SPLITS
            },
        }
    )
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("evaluation/dataset/evaluation_data/normalized/questions.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "evaluation/dataset/evaluation_data/normalized/weak_supervision_split.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
