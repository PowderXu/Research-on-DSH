"""Synthetic checks of the public evaluation contract; no model or dataset needed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kbbench.scoring import GitHubDocsSourceResolver, score_ranked_sources

from dsh_plugin.agent_eval.report import METRICS, build_report
from dsh_plugin.agent_eval.runner import run_batch


ARMS = ("fs", "hybrid", "neo4j")
DOC_ID = "/synthetic/example"


@pytest.fixture
def synthetic_project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    data = project / "evaluation/dataset/data"
    data.mkdir(parents=True)
    (data / "questions.jsonl").write_text(
        "".join(
            json.dumps({"question_id": question_id, "qrel_ids": [DOC_ID]}) + "\n"
            for question_id in ("synthetic-q1", "synthetic-q2")
        ),
        encoding="utf-8",
    )
    (data / "corpus.jsonl").write_text(
        json.dumps({"doc_id": DOC_ID, "source_path": "content/synthetic/example.md"})
        + "\n",
        encoding="utf-8",
    )
    for arm in ARMS:
        skill = project / f"dsh_plugin/plugin/skills/{arm}/initial_skill.md"
        skill.parent.mkdir(parents=True)
        skill.write_text("Synthetic test skill.\n", encoding="utf-8")
    return project


def _row(question_id: str = "synthetic-q1", **changes: object) -> dict:
    return {
        "id": question_id,
        "qrel_ids": [DOC_ID],
        "ranked_ids": [DOC_ID],
        "citation_ranked_ids": [DOC_ID],
        "predicted_answer": "Synthetic grounded answer.",
        "agent_ok": True,
        "task_type": "synthetic",
        "evidence_structure": "single_page",
        "usage": {"total": 10},
        "latency_seconds": 2.0,
        **score_ranked_sources([DOC_ID], [DOC_ID]),
        **changes,
    }


def _write_arms(project: Path, rows_by_arm: dict[str, list[dict]]) -> dict[str, Path]:
    roots = {}
    for arm, rows in rows_by_arm.items():
        root = project / "runs" / arm
        root.mkdir(parents=True, exist_ok=True)
        (root / "rollouts.json").write_text(json.dumps(rows), encoding="utf-8")
        roots[arm] = root
    return roots


@pytest.mark.parametrize("citations", [[], None])
def test_explicit_empty_final_sources_do_not_use_visible_hits(
    synthetic_project: Path, citations: list | None
) -> None:
    roots = _write_arms(synthetic_project, {"fs": [_row(citation_ranked_ids=citations)]})
    report, details = build_report(roots, project_root=synthetic_project)
    assert all(report["arms"][0][metric] == 0.0 for metric in METRICS)
    assert all(details[0][f"citation_{metric}"] == 0.0 for metric in METRICS)
    assert details[0]["citation_ranked_ids"] == []
    assert details[0]["visible_ranked_ids"] == [DOC_ID]
    assert details[0]["visible_hit_at_10"] == 1.0


def test_missing_legacy_citation_field_retains_ranked_ids_fallback(
    synthetic_project: Path,
) -> None:
    row = _row()
    del row["citation_ranked_ids"]
    roots = _write_arms(synthetic_project, {"fs": [row]})
    report, details = build_report(roots, project_root=synthetic_project)
    assert report["arms"][0]["hit_at_10"] == 1.0
    assert details[0]["citation_ranked_ids"] == [DOC_ID]


@pytest.mark.parametrize("visible_ids, expected", [([], 0.0), ([DOC_ID], 1.0)])
def test_explicit_visible_sources_determine_diagnostic_metrics(
    synthetic_project: Path, visible_ids: list[str], expected: float
) -> None:
    row = _row(visible_ranked_ids=visible_ids)
    # Deliberately stale primary metrics must not determine visible diagnostics.
    row.update({metric: 1.0 - expected for metric in METRICS})
    roots = _write_arms(synthetic_project, {"fs": [row]})
    report, details = build_report(roots, project_root=synthetic_project)
    assert report["arms"][0]["hit_at_10"] == 1.0
    assert details[0]["visible_ranked_ids"] == visible_ids
    assert all(details[0][f"visible_{metric}"] == expected for metric in METRICS)


@pytest.mark.parametrize("duplicate_arm", ARMS)
def test_duplicate_question_ids_are_rejected_even_when_id_sets_match(
    synthetic_project: Path, duplicate_arm: str
) -> None:
    rows = {arm: [_row()] for arm in ARMS}
    rows[duplicate_arm].append(_row())
    roots = _write_arms(synthetic_project, rows)
    with pytest.raises(ValueError, match=f"duplicate question IDs.*{duplicate_arm}.*synthetic-q1"):
        build_report(roots, project_root=synthetic_project)


def test_unique_paired_rows_can_arrive_in_different_orders(synthetic_project: Path) -> None:
    rows = [_row(), _row("synthetic-q2")]
    roots = _write_arms(synthetic_project, {"fs": rows, "hybrid": list(reversed(rows))})
    report, details = build_report(roots, project_root=synthetic_project)
    assert report["questions"] == 2
    assert len(details) == 4
    assert all(arm["questions"] == 2 for arm in report["arms"])


@pytest.mark.parametrize(
    "changes, failure_rate",
    [({"agent_ok": False}, 0.5), ({"predicted_answer": "  not found\n"}, 0.0)],
)
def test_failed_and_abstained_rows_keep_denominator_costs_and_visible_evidence(
    synthetic_project: Path, changes: dict, failure_rate: float
) -> None:
    rows = [_row(), _row("synthetic-q2", visible_ranked_ids=[DOC_ID], **changes)]
    roots = _write_arms(synthetic_project, {"fs": rows})
    report, details = build_report(roots, project_root=synthetic_project)
    arm = report["arms"][0]
    assert report["questions"] == arm["questions"] == len(details) == 2
    assert all(arm[metric] == 0.5 for metric in METRICS)
    assert all(arm[f"citation_{metric}"] == 0.5 for metric in METRICS)
    assert arm["failure_rate"] == failure_rate
    assert arm["tokens_per_qa"]["total"] == 10
    assert arm["latency_p50_ms"] == 2000
    assert arm["by_evidence_structure"][0]["hit_at_10"] == 0.5
    assert details[1]["citation_ranked_ids"] == [DOC_ID]
    assert details[1]["visible_hit_at_10"] == 1.0
    assert all(details[1][metric] == 0.0 for metric in METRICS)


@pytest.mark.parametrize(
    "changes, expected_gain, expected_ok",
    [
        ({}, 1.0, True),
        ({"return_code": 1}, 0.0, False),
        ({"parse_error": "synthetic parse error"}, 0.0, False),
        ({"skill_loaded": False}, 0.0, False),
        ({"answer": "  not found\n"}, 0.0, True),
        ({"answer": "Explain why a file was not found."}, 1.0, True),
    ],
)
def test_runner_and_report_agree_on_episode_gain(
    synthetic_project: Path, changes: dict, expected_gain: float, expected_ok: bool
) -> None:
    item = {
        "id": "synthetic-q1",
        "question": "What does the synthetic page say?",
        "qrel_ids": [DOC_ID],
        "task_type": "synthetic",
    }

    def fake_runner(_item: dict, _skill: Path, _case: Path) -> dict:
        return {
            "answer": "Synthetic grounded answer.",
            "sources": [DOC_ID],
            "visible_sources": [DOC_ID],
            "return_code": 0,
            "parse_error": "",
            "skill_loaded": True,
            "latency_seconds": 2.0,
            "usage": {"total": 10},
            **changes,
        }

    root = synthetic_project / "runs/fs"
    rows = run_batch(
        items=[item],
        skill_path=synthetic_project / "dsh_plugin/plugin/skills/fs/initial_skill.md",
        out_root=str(root),
        arm="fs",
        resolver=GitHubDocsSourceResolver(synthetic_project / "evaluation/dataset/data/corpus.jsonl"),
        runner=fake_runner,
    )
    assert rows[0]["agent_ok"] is expected_ok
    for metric in (*METRICS, "hard", "soft"):
        assert rows[0][metric] == expected_gain
    persisted = json.loads((root / "rollouts.json").read_text(encoding="utf-8"))
    assert persisted[0]["hit_at_10"] == expected_gain
    report, details = build_report({"fs": root}, project_root=synthetic_project)
    assert all(report["arms"][0][metric] == expected_gain for metric in METRICS)
    assert details[0]["visible_hit_at_10"] == 1.0
    assert details[0]["citation_ranked_ids"] == [DOC_ID]
