"""Frozen GitHub Docs question loader for Microsoft SkillOpt."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skillopt.datasets.base import SplitDataLoader


def normalize_item(raw: dict[str, Any]) -> dict[str, Any]:
    item_id = str(raw.get("question_id") or raw.get("id") or "").strip()
    question = str(raw.get("query") or raw.get("question") or "").strip()
    qrels = [str(value) for value in raw.get("qrel_ids") or [] if str(value)]
    if not item_id or not question or not qrels:
        raise ValueError("GitHub Docs item requires question_id, query, and qrel_ids")
    return {
        **raw,
        "id": item_id,
        "question": question,
        "qrel_ids": qrels,
        "task_type": str(raw.get("evidence_category") or "github_docs"),
        "intent_category": str(raw.get("intent_category") or "unknown"),
        "evidence_structure": str(raw.get("evidence_structure") or "unknown"),
        "qrel_count": int(raw.get("qrel_count") or len(set(qrels))),
    }


class GitHubDocsDshDataLoader(SplitDataLoader):
    """Load the pre-materialized train/val/test JSON arrays."""

    def load_split_items(self, split_path: str) -> list[dict[str, Any]]:
        files = sorted(Path(split_path).glob("*.json"))
        if len(files) != 1:
            raise FileNotFoundError(
                f"expected exactly one JSON file in {split_path}, found {len(files)}"
            )
        data = json.loads(files[0].read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"expected JSON array in {files[0]}")
        items = [normalize_item(row) for row in data]
        ids = [item["id"] for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate question IDs in {files[0]}")
        return items
