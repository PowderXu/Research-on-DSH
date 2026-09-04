from __future__ import annotations

from rule_optimization.adapter import (
    AnswerAspect,
    AspectConstruction,
    SourceSupport,
    _extract_json_object,
    _split_project_groups,
    bind_evidence_ids_from_claims,
    build_frozen_aspect_record,
    score_construction,
)


def _item() -> dict:
    return {
        "id": "project::1",
        "project": "project",
        "requirements": [
            {"requirement_id": "R1", "critical": True},
            {"requirement_id": "R2", "critical": False},
        ],
        "claims": [{"claim_id": "C1"}, {"claim_id": "C2"}],
        "evidence": [
            {"evidence_id": "accepted_answer", "kind": "accepted_answer"},
            {
                "evidence_id": "doc#section",
                "kind": "local_document_section",
                "doc_id": "project::/doc",
                "local_path": "docs/project/doc.md",
            },
        ],
    }


def test_score_construction_uses_deterministic_weighted_coverage() -> None:
    construction = AspectConstruction(
        aspects=[
            AnswerAspect(
                aspect_id="a1",
                description="Give the main answer",
                requirement_ids=["R1"],
                claim_ids=["C1"],
                evidence_ids=["accepted_answer"],
                importance=5,
                critical=True,
                rationale="Primary requirement",
            ),
            AnswerAspect(
                aspect_id="a2",
                description="Give the supporting procedure",
                requirement_ids=["R2"],
                claim_ids=["C2"],
                evidence_ids=["doc#section"],
                importance=3,
                critical=False,
                rationale="Useful procedure",
            ),
        ],
        source_support=[
            SourceSupport(aspect_id="a1", support="full", explanation="Present"),
            SourceSupport(aspect_id="a2", support="partial", explanation="Incomplete"),
        ],
        coverage_summary="Main answer is complete; procedure is partial.",
    )
    hard, soft, errors = score_construction(construction, _item(), threshold=0.80)
    assert hard == 1
    assert soft == 0.8125
    assert errors == []


def test_score_rejects_unknown_ids_and_missing_critical_mapping() -> None:
    construction = AspectConstruction(
        aspects=[
            AnswerAspect(
                aspect_id="a1",
                description="Unsupported aspect",
                requirement_ids=["R2"],
                claim_ids=["missing"],
                evidence_ids=["missing"],
                importance=5,
                critical=False,
                rationale="Invalid mappings",
            )
        ],
        source_support=[
            SourceSupport(aspect_id="a1", support="full", explanation="Claimed")
        ],
        coverage_summary="Invalid",
    )
    hard, soft, errors = score_construction(construction, _item(), threshold=0.80)
    assert hard == 0
    assert soft == 0.0
    assert any("unknown claim" in error for error in errors)
    assert any("critical requirements" in error for error in errors)


def test_split_is_deterministic_project_stratified_and_group_safe() -> None:
    rows = []
    for project in ("a", "b"):
        for index in range(10):
            rows.append(
                {
                    "id": f"{project}::{index}",
                    "project": project,
                    "source_url": f"https://example.test/{project}/{index}",
                }
            )
    duplicate = dict(rows[0])
    duplicate["id"] = "a::duplicate"
    rows.append(duplicate)
    first = _split_project_groups(rows, 7)
    second = _split_project_groups(rows, 7)
    assert {name: [row["id"] for row in values] for name, values in first.items()} == {
        name: [row["id"] for row in values] for name, values in second.items()
    }
    membership = {
        row["id"]: split for split, values in first.items() for row in values
    }
    assert membership["a::0"] == membership["a::duplicate"]
    for project in ("a", "b"):
        assert all(any(row["project"] == project for row in first[split]) for split in first)


def test_extract_json_accepts_fenced_output() -> None:
    assert _extract_json_object("```json\n{\"aspects\": []}\n```") == {"aspects": []}


def test_frozen_record_contains_weights_and_local_retrieval_targets() -> None:
    construction = AspectConstruction(
        aspects=[
            AnswerAspect(
                aspect_id="a1",
                description="Give the answer and procedure",
                requirement_ids=["R1"],
                claim_ids=["C1"],
                evidence_ids=["accepted_answer", "doc#section"],
                importance=5,
                critical=True,
                rationale="Primary requirement",
            )
        ],
        source_support=[
            SourceSupport(aspect_id="a1", support="full", explanation="Present")
        ],
        coverage_summary="Complete",
    )
    item = _item() | {
        "question": "How?",
        "reference_answer": "Do this.",
        "task_type": "single_page",
    }
    record = build_frozen_aspect_record(item, construction, "abc")
    assert record["rule_sha256"] == "abc"
    assert record["aspects"][0]["weight"] == 1.0
    assert record["aspects"][0]["retrieval_doc_ids"] == ["project::/doc"]


def test_freeze_binds_mistyped_evidence_id_from_mapped_claim() -> None:
    construction = AspectConstruction(
        aspects=[
            AnswerAspect(
                aspect_id="a1",
                description="Give the documented procedure",
                requirement_ids=["R1"],
                claim_ids=["C1"],
                evidence_ids=["project::/wrong#section"],
                importance=5,
                critical=True,
                rationale="Primary requirement",
            )
        ],
        source_support=[
            SourceSupport(aspect_id="a1", support="full", explanation="Present")
        ],
        coverage_summary="Complete",
    )
    item = _item()
    item["claims"] = [{"claim_id": "C1", "evidence_ids": ["doc#section"]}]
    repaired, repairs = bind_evidence_ids_from_claims(construction, item)
    assert repaired.aspects[0].evidence_ids == ["doc#section"]
    assert repairs == [
        {
            "aspect_id": "a1",
            "generated_evidence_ids": ["project::/wrong#section"],
            "bound_evidence_ids": ["doc#section"],
        }
    ]


def test_freeze_does_not_guess_from_historical_link_context() -> None:
    construction = AspectConstruction(
        aspects=[
            AnswerAspect(
                aspect_id="a1",
                description="Unsupported procedure",
                requirement_ids=["R1"],
                claim_ids=["C1"],
                evidence_ids=["missing"],
                importance=5,
                critical=True,
                rationale="Primary requirement",
            )
        ],
        source_support=[
            SourceSupport(aspect_id="a1", support="full", explanation="Present")
        ],
        coverage_summary="Complete",
    )
    item = _item()
    item["claims"] = [{"claim_id": "C1", "evidence_ids": ["historical"]}]
    item["evidence"].append(
        {"evidence_id": "historical", "kind": "historical_link_context"}
    )
    repaired, repairs = bind_evidence_ids_from_claims(construction, item)
    assert repaired.aspects[0].evidence_ids == ["missing"]
    assert repairs == []
