from __future__ import annotations

import importlib.util
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dataset" / "github_docs_kb_benchmark"
DATA = PACKAGE / "data"


def _load_evaluator():
    path = PACKAGE / "benchmark" / "evaluate_retrieval.py"
    spec = importlib.util.spec_from_file_location("packaged_github_docs_evaluator", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_packaged_dataset_counts_and_factual_labels() -> None:
    questions = [
        json.loads(line)
        for line in (DATA / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(questions) == 328
    assert sum(row["qrel_count"] for row in questions) == 421
    assert Counter(row["qrel_count"] for row in questions) == {
        1: 263,
        2: 46,
        3: 14,
        4: 2,
        5: 2,
        6: 1,
    }
    assert all("graph_opportunity" not in row for row in questions)
    assert all(row["qrel_count"] == len(set(row["qrel_ids"])) for row in questions)
    assert {row["evidence_structure"] for row in questions} == {"single", "linked", "dispersed"}


def test_packaged_splits_are_disjoint_and_cover_questions() -> None:
    split_ids: dict[str, set[str]] = {}
    expected_counts = {"train": 55, "validation": 27, "test": 246}
    for name, expected in expected_counts.items():
        rows = json.loads((DATA / "splits" / f"{name}.json").read_text(encoding="utf-8"))
        assert len(rows) == expected
        assert all("graph_opportunity" not in row for row in rows)
        split_ids[name] = {str(row["question_id"]) for row in rows}
    assert split_ids["train"].isdisjoint(split_ids["validation"])
    assert split_ids["train"].isdisjoint(split_ids["test"])
    assert split_ids["validation"].isdisjoint(split_ids["test"])
    assert len(set().union(*split_ids.values())) == 328


def test_manifest_and_corpus_counts() -> None:
    manifest = json.loads((DATA / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["documents"] == 3740
    assert manifest["questions"] == 328
    assert manifest["qrels"] == 421
    assert manifest["oracle_features_excluded"] == ["graph_opportunity"]
    with (DATA / "corpus.jsonl").open(encoding="utf-8") as handle:
        assert sum(1 for line in handle if line.strip()) == 3740


def test_standalone_evaluator_resolves_url_and_scores(tmp_path: Path) -> None:
    evaluator = _load_evaluator()
    question = {
        "question_id": "43",
        "qrel_ids": ["/graphql/guides/using-the-graphql-api-for-discussions"],
        "qrel_count": 1,
        "intent_category": "other_product_question",
        "evidence_category": "single_page_direct",
        "evidence_structure": "single",
    }
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(json.dumps([question]), encoding="utf-8")
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps({
            "question_id": "43",
            "ranked_ids": [
                "https://docs.github.com/en/graphql/guides/using-the-graphql-api-for-discussions"
            ],
            "latency_ms": 12.5,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }) + "\n",
        encoding="utf-8",
    )
    report = evaluator.evaluate(
        questions_path,
        DATA / "corpus.jsonl",
        predictions_path,
    )
    assert report["overall"]["Hit@10"] == 1.0
    assert report["overall"]["Recall@10"] == 1.0
    assert report["overall"]["nDCG@10"] == 1.0
    assert report["overall"]["p50_latency_ms"] == 12.5
    assert report["overall"]["mean_total_tokens"] == 15.0
