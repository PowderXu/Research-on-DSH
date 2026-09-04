from __future__ import annotations

import json
from pathlib import Path

from dataset_analysis.audit_integrity import (
    LEXICAL_JACCARD_THRESHOLD,
    audit,
    canonical_split,
    normalize_identifier,
    normalize_url,
    write_report,
)


def _write_questions(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(
    *,
    question_id: str,
    source_question_id: str,
    split: str,
    query: str,
    source_url: str,
    accepted_answer_url: str,
    project: str = "Docs",
) -> dict:
    return {
        "question_id": question_id,
        "source_question_id": source_question_id,
        "project": project,
        "split": split,
        "query": query,
        "source_url": source_url,
        "accepted_answer_url": accepted_answer_url,
    }


def test_narrow_identifier_url_and_split_normalization() -> None:
    assert canonical_split(" TEST ") == "development"
    assert normalize_identifier(" Docs :: １２ ") == "docs::12"
    assert (
        normalize_url("HTTPS://EXAMPLE.COM:443/a/", keep_fragment=False)
        == "https://example.com/a"
    )
    assert (
        normalize_url("https://example.com/a/#Answer", keep_fragment=True)
        == "https://example.com/a#Answer"
    )


def test_audit_finds_exact_normalized_and_cross_split_overlap(tmp_path: Path) -> None:
    dataset = tmp_path / "normalized"
    dataset.mkdir()
    common_query = "How can I configure this exact documentation setting?"
    _write_questions(
        dataset / "questions.jsonl",
        [
            _row(
                question_id="Docs::1",
                source_question_id="1",
                split="train",
                query=common_query,
                source_url="HTTPS://EXAMPLE.COM:443/questions/1/",
                accepted_answer_url="https://example.com/questions/1/#answer-1",
            ),
            _row(
                question_id=" docs::１ ",
                source_question_id=" １ ",
                split="test",
                query=common_query,
                source_url="https://example.com/questions/1",
                accepted_answer_url="https://EXAMPLE.com/questions/1#answer-1",
            ),
        ],
    )

    report = audit(dataset)

    assert report["input"]["split_counts"] == {"development": 1, "train": 1}
    for key in (
        "exact_question",
        "normalized_question_id",
        "normalized_source_question_id",
        "normalized_source_url",
        "normalized_accepted_answer_url",
    ):
        assert report["duplicates"][key]["group_count"] == 1
        assert report["duplicates"][key]["cross_split_group_count"] == 1
    pair = report["split_overlap"]["pairwise"][0]
    assert pair["splits"] == ["development", "train"]
    assert pair["exact_question"] == 1


def test_lexical_near_duplicates_are_candidates_and_exclude_raw_exact_pairs(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "normalized"
    dataset.mkdir()
    base = "one two three four five six seven eight nine ten eleven twelve"
    _write_questions(
        dataset / "questions.jsonl",
        [
            _row(
                question_id="docs::1",
                source_question_id="1",
                split="train",
                query=base,
                source_url="https://example.com/1",
                accepted_answer_url="https://example.com/1#a",
            ),
            _row(
                question_id="docs::2",
                source_question_id="2",
                split="validation",
                query=base + " thirteen",
                source_url="https://example.com/2",
                accepted_answer_url="https://example.com/2#a",
            ),
            _row(
                question_id="docs::3",
                source_question_id="3",
                split="test",
                query=base,
                source_url="https://example.com/3",
                accepted_answer_url="https://example.com/3#a",
            ),
            _row(
                question_id="docs::4",
                source_question_id="4",
                split="test",
                query="entirely unrelated vocabulary about another technical issue",
                source_url="https://example.com/4",
                accepted_answer_url="https://example.com/4#a",
            ),
        ],
    )

    report = audit(dataset)
    lexical = report["lexical_near_duplicate_diagnostic"]

    assert LEXICAL_JACCARD_THRESHOLD == 0.80
    assert lexical["candidate_pair_count"] == 2
    assert lexical["cross_split_candidate_pair_count"] == 2
    assert {
        (pair["left_question_id"], pair["right_question_id"])
        for pair in lexical["pairs"]
    } == {("docs::1", "docs::2"), ("docs::2", "docs::3")}
    assert "not a semantic-duplicate judgment" in report["method"][
        "lexical_near_duplicate_diagnostic"
    ]["label"]


def test_report_writes_stable_json_and_markdown_warning(tmp_path: Path) -> None:
    dataset = tmp_path / "normalized"
    dataset.mkdir()
    _write_questions(
        dataset / "questions.jsonl",
        [
            _row(
                question_id="docs::1",
                source_question_id="1",
                split="test",
                query="How do I configure the local documentation tool correctly?",
                source_url="https://example.com/1",
                accepted_answer_url="https://example.com/1#a",
            )
        ],
    )
    output = tmp_path / "results"

    write_report(audit(dataset), output)

    persisted = json.loads((output / "report.json").read_text(encoding="utf-8"))
    markdown = (output / "REPORT.md").read_text(encoding="utf-8")
    assert persisted["schema_version"] == "dataset-integrity-v1"
    assert persisted["input"]["question_count"] == 1
    assert "does **not** establish semantic-duplicate absence" in markdown
    assert "development=1" in markdown
