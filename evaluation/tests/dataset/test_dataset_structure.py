from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from dataset.scripts.verify import verify_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "evaluation/dataset"


def test_dataset_root_has_exactly_four_directories_and_no_files() -> None:
    assert {path.name for path in DATASET_ROOT.iterdir()} == {
        "docs",
        "templates",
        "scripts",
        "evaluation_data",
    }
    assert all(path.is_dir() for path in DATASET_ROOT.iterdir())


def test_templates_are_data_contracts_and_graph_schema_is_plugin_owned() -> None:
    expected = {
        "public_sources.json",
        "source_config.schema.json",
        "corpus.schema.json",
        "question.schema.json",
        "manifest.schema.json",
    }
    for name in expected:
        json.loads((DATASET_ROOT / "templates" / name).read_text(encoding="utf-8"))
    assert not list((DATASET_ROOT / "evaluation_data").rglob("graph_schema.json"))
    graph_schema = PROJECT_ROOT / "dsh_plugin/plugin/graph_schema.json"
    assert json.loads(graph_schema.read_text(encoding="utf-8"))["owner"] == "dsh-docsqa-plugin"


def test_frozen_discussion_source_manifest_reproduces_candidate_counts() -> None:
    rows = [
        json.loads(line)
        for line in (DATASET_ROOT / "templates/discussion_sources.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert len(rows) == 798
    assert Counter(str(row["dataset"]) for row in rows) == {
        "github_docs": 328,
        "prisma": 213,
        "supabase": 90,
        "tailwind": 167,
    }
    assert len({str(row["source_url"]) for row in rows}) == len(rows)
    assert Counter(str(row["benchmark_split"]) for row in rows) == {
        "test": 625,
        "validation": 118,
        "train": 55,
    }


def test_combined_evaluation_data_passes_integrity_checks() -> None:
    combined = DATASET_ROOT / "evaluation_data" / "combined"
    if not (combined / "manifest.json").is_file():
        pytest.skip("generated combined dataset is not present in this checkout")
    assert verify_dataset(combined) == {
        "documents": 5392,
        "questions": 556,
        "qrels": 627,
    }
