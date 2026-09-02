from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from dataset_analysis.build_aspects import DEFAULT_RUBRIC
from dataset_analysis.calibrate_aspect_judge import (
    CalibrationReview,
    _immutable_contract,
    _require_case_generation_rubric,
    _resumable_calibration_cases,
    _select_records,
    _validate_question_selection_args,
    _validate_report_only_rows,
    _validate_resume_contract,
    build_weak_supervision_gate,
    classification_report,
    load_reused_cases,
    load_seed_cases,
    reference_review_passed,
    run,
)
from dataset_analysis.question_partitions import EXPECTED_SCHEMA
from dsh_plugin.agent_eval.aspect_judge import load_judge_rubric


GENERIC_RUBRIC = DEFAULT_RUBRIC.with_name("aspect_judge_generic_v1.json")


def test_classification_report_balanced_metrics() -> None:
    rows = [
        {"expected_label": "complete", "predicted_label": "complete"},
        {"expected_label": "partial", "predicted_label": "partial"},
        {"expected_label": "incorrect", "predicted_label": "incorrect"},
    ]
    report = classification_report(rows)
    assert report["accuracy"] == 1.0
    assert report["macro_f1"] == 1.0
    assert report["complete_recall"] == 1.0
    assert report["incorrect_false_complete_rate"] == 0.0


def test_classification_report_exposes_false_acceptance() -> None:
    rows = [
        {"expected_label": "complete", "predicted_label": "complete"},
        {"expected_label": "partial", "predicted_label": "complete"},
        {"expected_label": "incorrect", "predicted_label": "complete"},
    ]
    report = classification_report(rows)
    assert report["accuracy"] == 1 / 3
    assert report["complete_recall"] == 1.0
    assert report["partial_recall"] == 0.0
    assert report["incorrect_false_complete_rate"] == 1.0


def test_weak_supervision_gate_accepts_balanced_discrimination() -> None:
    metrics = classification_report(
        [
            {"expected_label": "complete", "predicted_label": "complete"},
            {"expected_label": "partial", "predicted_label": "partial"},
            {"expected_label": "incorrect", "predicted_label": "incorrect"},
        ]
    )
    gate = build_weak_supervision_gate(
        metrics=metrics,
        independent_source_review_rate=1.0,
        balanced_case_rate=1.0,
        min_source_answer_complete_recall=0.8,
        min_partial_recall=0.8,
        min_incorrect_recall=0.8,
        max_incorrect_false_complete_rate=0.05,
        min_macro_f1=0.8,
        min_balanced_case_rate=0.98,
    )
    assert gate["passed"]
    assert gate["criteria"]["source_answer_complete_recall"]["passed"]


def test_weak_supervision_gate_rejects_always_complete_judge() -> None:
    metrics = classification_report(
        [
            {"expected_label": "complete", "predicted_label": "complete"},
            {"expected_label": "partial", "predicted_label": "complete"},
            {"expected_label": "incorrect", "predicted_label": "complete"},
        ]
    )
    gate = build_weak_supervision_gate(
        metrics=metrics,
        independent_source_review_rate=1.0,
        balanced_case_rate=1.0,
        min_source_answer_complete_recall=0.8,
        min_partial_recall=0.8,
        min_incorrect_recall=0.8,
        max_incorrect_false_complete_rate=0.05,
        min_macro_f1=0.8,
        min_balanced_case_rate=0.98,
    )
    assert not gate["passed"]
    assert not gate["criteria"]["partial_control_recall"]["passed"]
    assert not gate["criteria"]["incorrect_control_recall"]["passed"]
    assert not gate["criteria"]["incorrect_false_complete_rate"]["passed"]


def test_report_only_validation_requires_frozen_balanced_rows() -> None:
    records = [{"question_id": "q1"}]
    cases = [{"question_id": "q1", "status": "accepted"}]
    grade_rows = [
        {
            "question_id": "q1",
            "candidate": label,
            "expected_label": label,
            "predicted_label": label,
        }
        for label in ("complete", "partial", "incorrect")
    ]
    _validate_report_only_rows(
        records, cases, grade_rows, [{"question_id": "q1"}]
    )
    grade_rows[1]["expected_label"] = "complete"
    with pytest.raises(ValueError, match="expected labels are not frozen"):
        _validate_report_only_rows(
            records, cases, grade_rows, [{"question_id": "q1"}]
        )


def test_resume_contract_rejects_uncontracted_or_changed_artifacts(
    tmp_path: Path,
) -> None:
    expected = {"rubric_sha256": "a", "model": "m"}
    (tmp_path / "per_candidate.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no run_contract"):
        _validate_resume_contract(tmp_path, expected)
    (tmp_path / "run_contract.json").write_text(
        json.dumps({"immutable": {"rubric_sha256": "b", "model": "m"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contract mismatch"):
        _validate_resume_contract(tmp_path, expected)
    (tmp_path / "run_contract.json").write_text(
        json.dumps({"immutable": expected}), encoding="utf-8"
    )
    _validate_resume_contract(tmp_path, expected)
    (tmp_path / "run_contract.json").write_text(
        json.dumps(
            {
                "immutable": expected,
                "artifacts": {"per_candidate_sha256": "not-the-real-hash"},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        _validate_resume_contract(tmp_path, expected)
    real_hash = hashlib.sha256(
        (tmp_path / "per_candidate.jsonl").read_bytes()
    ).hexdigest()
    (tmp_path / "run_contract.json").write_text(
        json.dumps(
            {
                "immutable": expected,
                "artifacts": {"per_candidate_sha256": real_hash},
            }
        ),
        encoding="utf-8",
    )
    _validate_resume_contract(tmp_path, expected)


def test_resume_accepts_pre_generation_control_contract_only_for_legacy_split(
    tmp_path: Path,
) -> None:
    legacy = {"rubric_sha256": "a", "model": "m", "seed": 42}
    expected = {
        **legacy,
        "case_generation": {
            "mode": "generated_cases",
            "attempts": 3,
            "max_variant_repairs": 1,
            "seed": 42,
            "request_timeout_seconds": 120.0,
        },
    }
    (tmp_path / "run_contract.json").write_text(
        json.dumps({"immutable": legacy}), encoding="utf-8"
    )
    _validate_resume_contract(tmp_path, expected)

    manifest_expected = {
        **expected,
        "question_selection": {"selection_mode": "weak_supervision_manifest"},
    }
    with pytest.raises(ValueError, match="contract mismatch"):
        _validate_resume_contract(tmp_path, manifest_expected)

    nondefault_expected = {
        **expected,
        "case_generation": {**expected["case_generation"], "attempts": 4},
    }
    with pytest.raises(ValueError, match="contract mismatch"):
        _validate_resume_contract(tmp_path, nondefault_expected)


def test_reference_review_is_counted_even_if_variant_case_is_rejected() -> None:
    case = {
        "status": "rejected",
        "history": [
            {
                "review": {
                    "candidates": [
                        {
                            "candidate": "complete",
                            "actual_label": "complete",
                            "valid_for_calibration": True,
                        },
                        {
                            "candidate": "partial",
                            "actual_label": "complete",
                            "valid_for_calibration": False,
                        },
                    ]
                }
            }
        ],
    }
    assert reference_review_passed(case)


def test_calibration_review_schema_has_no_free_form_object() -> None:
    def visit(value: object) -> None:
        if isinstance(value, dict):
            additional = value.get("additionalProperties")
            assert not isinstance(additional, dict)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(CalibrationReview.model_json_schema())


def test_generic_baseline_rubric_is_domain_neutral_and_judge_only() -> None:
    rubric, digest = load_judge_rubric(GENERIC_RUBRIC)
    assert len(digest) == 64
    assert "judge_system" in rubric
    for field in (
        "candidate_system",
        "candidate_rules",
        "candidate_reviewer_system",
        "candidate_reviewer_rules",
    ):
        assert field not in rubric
    serialized = json.dumps(rubric).casefold()
    for forbidden in ("github", "prisma", "supabase", "tailwind", "question_id"):
        assert forbidden not in serialized


def test_judge_only_rubric_requires_reused_cases() -> None:
    rubric, _ = load_judge_rubric(GENERIC_RUBRIC)
    with pytest.raises(ValueError, match="--reuse-cases-from"):
        _require_case_generation_rubric(rubric)


def test_reused_cases_must_exactly_match_selected_records(tmp_path: Path) -> None:
    source = tmp_path / "calibration_cases.jsonl"
    source.write_text(
        json.dumps({"question_id": "q1", "status": "accepted"}) + "\n",
        encoding="utf-8",
    )
    cases, resolved, digest = load_reused_cases(
        source, [{"question_id": "q1"}]
    )
    assert set(cases) == {"q1"}
    assert resolved == source
    assert len(digest) == 64
    with pytest.raises(ValueError, match="do not exactly match"):
        load_reused_cases(source, [{"question_id": "q1"}, {"question_id": "q2"}])


def test_reused_cases_reject_duplicate_question_ids(tmp_path: Path) -> None:
    source = tmp_path / "calibration_cases.jsonl"
    row = json.dumps({"question_id": "q1", "status": "accepted"})
    source.write_text(f"{row}\n{row}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate reused calibration"):
        load_reused_cases(source, [{"question_id": "q1"}])


def _write_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": EXPECTED_SCHEMA,
                "name": "unit-test-split",
                "partitions": {
                    "calibration_train": ["q2"],
                    "calibration_validation": ["q3"],
                    "final_test": ["q1"],
                },
            }
        ),
        encoding="utf-8",
    )


def _selection_args(manifest: Path, **overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "question_manifest": manifest,
        "manifest_partition": "calibration_validation",
        "split": [],
        "limit": None,
        "seed": 42,
        "rubric_role": "candidate",
        "reuse_cases_from": None,
        "seed_cases_from": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_manifest_selection_ignores_legacy_physical_split(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest)
    records, metadata = _select_records(
        _selection_args(manifest),
        [
            {"question_id": "q1", "split": "train"},
            {"question_id": "q2", "split": "validation"},
            {"question_id": "q3", "split": "train"},
        ],
    )
    assert [row["question_id"] for row in records] == ["q3"]
    assert metadata["manifest_partition"] == "calibration_validation"
    assert len(metadata["manifest_sha256"]) == 64
    assert len(metadata["manifest_canonical_sha256"]) == 64
    assert len(metadata["selected_question_ids_sha256"]) == 64


def test_legacy_physical_split_selection_is_preserved() -> None:
    records, metadata = _select_records(
        argparse.Namespace(
            question_manifest=None,
            manifest_partition=None,
            split=["validation"],
            limit=None,
            seed=42,
            rubric_role="frozen",
            reuse_cases_from=None,
            seed_cases_from=None,
        ),
        [
            {"question_id": "q1", "split": "train"},
            {"question_id": "q2", "split": "validation"},
        ],
    )
    assert [row["question_id"] for row in records] == ["q2"]
    assert metadata["selection_mode"] == "legacy_physical_split"


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"split": ["validation"]}, "--split"),
        ({"limit": 1}, "--limit"),
        (
            {"manifest_partition": "final_test", "rubric_role": "candidate"},
            "rubric-role frozen",
        ),
        ({"question_manifest": None}, "requires --question-manifest"),
    ],
)
def test_manifest_selection_rejects_unsafe_combinations(
    tmp_path: Path, overrides: dict[str, object], match: str
) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest)
    with pytest.raises(ValueError, match=match):
        _validate_question_selection_args(_selection_args(manifest, **overrides))


def test_final_test_accepts_only_unlimited_frozen_run(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest)
    args = _selection_args(
        manifest, manifest_partition="final_test", rubric_role="frozen"
    )
    _validate_question_selection_args(args)
    args.limit = 1
    with pytest.raises(ValueError, match="--limit"):
        _validate_question_selection_args(args)


def test_manifest_provenance_enters_immutable_contract(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest)
    aspects = tmp_path / "aspects.jsonl"
    aspects.write_text("{}\n", encoding="utf-8")
    args = _selection_args(manifest)
    args.aspects = aspects
    args.model = "model"
    args.reasoning_effort = "medium"
    args.attempts = 2
    args.max_variant_repairs = 4
    args.request_timeout = 90.0
    args.min_source_answer_complete_recall = 0.8
    args.min_partial_recall = 0.8
    args.min_incorrect_recall = 0.8
    args.max_incorrect_false_complete_rate = 0.05
    args.min_macro_f1 = 0.8
    args.min_balanced_case_rate = 0.98
    records, metadata = _select_records(
        args,
        [
            {"question_id": "q1", "split": "train"},
            {"question_id": "q2", "split": "validation"},
            {"question_id": "q3", "split": "train"},
        ],
    )
    contract = _immutable_contract(
        args,
        rubric={"rubric_version": "v1"},
        rubric_sha256="a" * 64,
        records=records,
        selection_metadata=metadata,
        reused_case_sha256=None,
        seeded_case_provenance=None,
    )
    selection = contract["question_selection"]
    assert selection["manifest_partition"] == "calibration_validation"
    assert selection["manifest_sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()
    assert selection["selected_question_ids_sha256"] == contract[
        "selected_question_ids_sha256"
    ]
    assert contract["case_generation"] == {
        "mode": "generated_cases",
        "attempts": 2,
        "max_variant_repairs": 4,
        "seed": 42,
        "request_timeout_seconds": 90.0,
    }
    for field, value in (("attempts", 3), ("max_variant_repairs", 3)):
        changed_args = argparse.Namespace(**vars(args))
        setattr(changed_args, field, value)
        changed_contract = _immutable_contract(
            changed_args,
            rubric={"rubric_version": "v1"},
            rubric_sha256="a" * 64,
            records=records,
            selection_metadata=metadata,
            reused_case_sha256=None,
            seeded_case_provenance=None,
        )
        assert changed_contract != contract
    args.seed_cases_from = tmp_path / "seed.jsonl"
    seed_provenance = {
        "source": str(args.seed_cases_from),
        "source_sha256": "b" * 64,
        "source_row_count": 2,
        "source_status_counts": {"accepted": 1, "rejected": 1},
        "accepted_seeded_question_count": 1,
        "accepted_seeded_question_ids_sha256": "c" * 64,
    }
    seeded_contract = _immutable_contract(
        args,
        rubric={"rubric_version": "v1"},
        rubric_sha256="a" * 64,
        records=records,
        selection_metadata=metadata,
        reused_case_sha256=None,
        seeded_case_provenance=seed_provenance,
    )
    assert seeded_contract["case_mode"] == "seeded_fixed_cases_plus_generated"
    assert seeded_contract["seeded_case_provenance"] == seed_provenance


def test_seed_cases_allow_subset_but_reject_unknown_and_duplicates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "seed.jsonl"
    source.write_text(
        json.dumps({"question_id": "q1", "status": "accepted"}) + "\n",
        encoding="utf-8",
    )
    cases, resolved, digest, provenance = load_seed_cases(
        source, [{"question_id": "q1"}, {"question_id": "q2"}]
    )
    assert set(cases) == {"q1"}
    assert resolved == source
    assert len(digest) == 64
    assert provenance["source_row_count"] == 1
    assert provenance["source_status_counts"] == {"accepted": 1}
    assert provenance["accepted_seeded_question_count"] == 1
    assert len(provenance["accepted_seeded_question_ids_sha256"]) == 64
    with pytest.raises(ValueError, match="outside the selected partition"):
        load_seed_cases(source, [{"question_id": "q2"}])
    row = json.dumps({"question_id": "q1", "status": "accepted"})
    source.write_text(f"{row}\n{row}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate seed calibration"):
        load_seed_cases(source, [{"question_id": "q1"}])


def test_seed_cases_regenerate_rejected_error_and_missing_status_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "seed.jsonl"
    rows = [
        {"question_id": "accepted", "status": "accepted"},
        {"question_id": "rejected", "status": "rejected"},
        {"question_id": "error", "status": "error"},
        {"question_id": "missing"},
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    cases, _, _, provenance = load_seed_cases(
        source, [{"question_id": row["question_id"]} for row in rows]
    )
    assert set(cases) == {"accepted"}
    assert provenance == {
        "source_row_count": 4,
        "source_status_counts": {
            "accepted": 1,
            "error": 1,
            "missing": 1,
            "rejected": 1,
        },
        "accepted_seeded_question_count": 1,
        "accepted_seeded_question_ids_sha256": provenance[
            "accepted_seeded_question_ids_sha256"
        ],
    }


def test_resume_retries_case_generation_errors(tmp_path: Path) -> None:
    source = tmp_path / "calibration_cases.jsonl"
    rows = [
        {"question_id": "retry", "status": "error"},
        {"question_id": "accepted", "status": "accepted"},
        {"question_id": "rejected", "status": "rejected"},
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    assert set(_resumable_calibration_cases(source)) == {"accepted", "rejected"}


def test_seed_and_exact_reuse_are_mutually_exclusive(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _validate_question_selection_args(
            _selection_args(
                manifest,
                reuse_cases_from=tmp_path / "exact.jsonl",
                seed_cases_from=tmp_path / "seed.jsonl",
            )
        )


def test_report_and_contract_record_manifest_provenance(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest)
    aspects = tmp_path / "aspects.jsonl"
    aspects.write_text(
        "\n".join(
            json.dumps(
                {
                    "question_id": question_id,
                    "split": split,
                    "project": "project",
                }
            )
            for question_id, split in (
                ("q1", "train"),
                ("q2", "validation"),
                ("q3", "train"),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "run"
    output.mkdir()
    (output / "calibration_cases.jsonl").write_text(
        json.dumps(
            {
                "question_id": "q3",
                "project": "project",
                "status": "accepted",
                "history": [
                    {
                        "review": {
                            "candidates": [
                                {
                                    "candidate": "complete",
                                    "actual_label": "complete",
                                    "valid_for_calibration": True,
                                }
                            ]
                        }
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "per_candidate.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "question_id": "q3",
                    "project": "project",
                    "candidate": label,
                    "expected_label": label,
                    "predicted_label": label,
                }
            )
            for label in ("complete", "partial", "incorrect")
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "judge_calls.jsonl").write_text(
        json.dumps({"question_id": "q3"}) + "\n", encoding="utf-8"
    )
    args = _selection_args(manifest)
    args.aspects = aspects
    args.rubric = DEFAULT_RUBRIC
    args.output_dir = output
    args.model = "gpt-5.6-luna"
    args.reasoning_effort = "medium"
    args.workers = 1
    args.attempts = 1
    args.max_variant_repairs = 0
    args.request_timeout = 1.0
    args.resume = False
    args.report_only = True
    args.min_source_answer_complete_recall = 0.8
    args.min_partial_recall = 0.8
    args.min_incorrect_recall = 0.8
    args.max_incorrect_false_complete_rate = 0.05
    args.min_macro_f1 = 0.8
    args.min_balanced_case_rate = 0.98
    report = run(args)
    assert report["question_selection"]["manifest_partition"] == (
        "calibration_validation"
    )
    contract = json.loads((output / "run_contract.json").read_text(encoding="utf-8"))
    assert contract["immutable"]["question_selection"] == report[
        "question_selection"
    ]
    assert report["case_generation"] == {
        "mode": "generated_cases",
        "attempts": 1,
        "max_variant_repairs": 0,
        "seed": 42,
        "request_timeout_seconds": 1.0,
    }
    assert contract["immutable"]["case_generation"] == report["case_generation"]
