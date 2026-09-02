from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_analysis.build_aspects import (
    DEFAULT_RUBRIC,
    _canonicalize_aspects,
    _resumable_results,
    load_rubric,
    select_source_rows,
    validate_aspects,
)


def _context() -> dict:
    return {
        "requirements": [
            {"requirement_id": "R1", "requirement": "Resolve the request", "critical": True}
        ],
        "claims": [{"claim_id": "C1", "claim": "A supported resolution"}],
        "evidence": [
            {
                "evidence_id": "doc::one#section",
                "doc_id": "doc::one",
                "kind": "local_document_section",
                "text": "Supporting text",
            }
        ],
    }


def _aspect() -> dict:
    return {
        "aspect_id": "custom",
        "description": "Provide the supported resolution",
        "requirement_ids": ["R1"],
        "claim_ids": ["C1"],
        "evidence_ids": ["doc::one#section"],
        "importance": 5,
        "critical": True,
        "rationale": "This resolves the critical requirement.",
    }


def test_production_aspect_rubric_is_domain_neutral() -> None:
    rubric, digest = load_rubric(DEFAULT_RUBRIC)
    serialized = json.dumps(rubric).casefold()
    for forbidden in (
        "prisma",
        "supabase",
        "tailwind",
        "codeql",
        "dependabot",
        "question_overrides",
        "project_overrides",
        "case_overrides",
    ):
        assert forbidden not in serialized
    assert len(digest) == 64


def test_rubric_loader_rejects_case_overrides(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_RUBRIC.read_text(encoding="utf-8"))
    payload["question_overrides"] = {"example": {"aspects": []}}
    path = tmp_path / "rubric.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden override"):
        load_rubric(path)


def test_validate_aspects_requires_critical_requirement_coverage() -> None:
    aspect = _aspect()
    aspect["critical"] = False
    assert validate_aspects([aspect], _context()) == [
        "critical requirements lack a critical aspect: ['R1']"
    ]


def test_validate_aspects_rejects_unknown_mappings() -> None:
    aspect = _aspect()
    aspect["claim_ids"] = ["C404"]
    aspect["evidence_ids"] = ["missing"]
    errors = validate_aspects([aspect], _context())
    assert "aspect[1] has unknown claims: ['C404']" in errors
    assert "aspect[1] has unknown evidence: ['missing']" in errors


def test_canonicalize_aspects_adds_weights_and_retrieval_docs() -> None:
    first = _aspect()
    second = {
        **_aspect(),
        "aspect_id": "anything",
        "description": "Provide a supporting qualification",
        "importance": 3,
        "critical": False,
    }
    rows = _canonicalize_aspects([first, second], _context())
    assert [row["aspect_id"] for row in rows] == ["a1", "a2"]
    assert [row["weight"] for row in rows] == [0.625, 0.375]
    assert rows[0]["retrieval_doc_ids"] == ["doc::one"]


def _source_row(question_id: str, split: str = "test") -> dict:
    return {"question_id": question_id, "split": split, "status": "accepted"}


def test_select_source_rows_uses_manifest_independently_of_legacy_split(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "weak-supervision.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "docsqa-weak-supervision-splits-v1",
                "name": "test",
                "partitions": {
                    "calibration_train": ["q1", "q2"],
                    "calibration_validation": ["q3"],
                    "final_test": ["q4"],
                },
            }
        ),
        encoding="utf-8",
    )
    rows = [
        _source_row("q1", "test"),
        _source_row("q2", "validation"),
        _source_row("q3", "train"),
        _source_row("q4", "test"),
        {"question_id": "rejected", "split": "test", "status": "rejected"},
    ]
    selected, metadata = select_source_rows(
        rows,
        split="all",
        question_ids=[],
        limit=None,
        question_manifest=manifest,
        manifest_partition="calibration_validation",
    )
    assert [row["question_id"] for row in selected] == ["q3"]
    assert metadata["selection_mode"] == "weak_supervision_manifest"
    assert metadata["manifest_partition"] == "calibration_validation"
    assert metadata["selected_question_count"] == 1


def test_select_source_rows_rejects_ambiguous_manifest_options(tmp_path: Path) -> None:
    manifest = tmp_path / "weak-supervision.json"
    with pytest.raises(ValueError, match="cannot be combined"):
        select_source_rows(
            [_source_row("q1")],
            split="test",
            question_ids=[],
            limit=None,
            question_manifest=manifest,
            manifest_partition="calibration_train",
        )
    with pytest.raises(ValueError, match="requires --question-manifest"):
        select_source_rows(
            [_source_row("q1")],
            split="all",
            question_ids=[],
            limit=None,
            question_manifest=None,
            manifest_partition="calibration_train",
        )


def test_select_source_rows_rejects_duplicate_normalization_ids() -> None:
    with pytest.raises(ValueError, match="unique, non-empty"):
        select_source_rows(
            [_source_row("q1"), _source_row("q1", "validation")],
            split="all",
            question_ids=[],
            limit=None,
            question_manifest=None,
            manifest_partition="all",
        )


def test_resume_retries_errors_but_keeps_terminal_rows(tmp_path: Path) -> None:
    path = tmp_path / "per_question.jsonl"
    rows = [
        {"question_id": "retry", "status": "error"},
        {"question_id": "accepted", "status": "accepted"},
        {"question_id": "rejected", "status": "rejected"},
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    assert set(_resumable_results(path)) == {"accepted", "rejected"}
