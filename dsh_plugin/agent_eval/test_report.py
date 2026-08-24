from __future__ import annotations

import json
from pathlib import Path

import pytest

from dsh_plugin.agent_eval.report import build_report, percentile


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _row(item_id: str, score: float, *, expand: bool = False) -> dict:
    tools = ["skill", "techdocs_search"]
    if expand:
        tools.append("techdocs_expand")
    return {
        "id": item_id,
        "task_type": "multi_page_linked",
        "intent_category": "troubleshooting",
        "evidence_structure": "linked",
        "qrel_count": 1,
        "qrel_ids": ["/docs/example"],
        "ranked_ids": ["/docs/example"],
        "citation_ranked_ids": ["/docs/example"] if score else ["/docs/other"],
        "tool_sequence": tools,
        "latency_seconds": 2.0,
        "usage": {
            "input_fresh": 10,
            "input_cached": 20,
            "input_total": 30,
            "output": 5,
            "total": 35,
        },
        "actual_models": ["test-model"],
        "agent_ok": True,
        "recall_at_1": score,
        "recall_at_5": score,
        "recall_at_10": score,
        "recall_at_20": score,
        "hit_at_1": score,
        "hit_at_5": score,
        "hit_at_10": score,
        "hit_at_20": score,
        "ndcg_at_10": score,
        "all_support_at_10": score,
    }


def _write_run(root: Path, rows: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "rollouts.json").write_text(json.dumps(rows), encoding="utf-8")


def test_percentile_uses_linear_interpolation() -> None:
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert percentile([0, 10], 0.95) == pytest.approx(9.5)


def test_report_requires_paired_ids_and_keeps_graph_use_explicit(tmp_path: Path) -> None:
    roots = {arm: tmp_path / arm for arm in ("fs", "hybrid", "neo4j")}
    for arm, root in roots.items():
        missed_final_source = _row("q2", 0.0)
        for key in (
            "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20",
            "hit_at_1", "hit_at_5", "hit_at_10", "hit_at_20",
            "ndcg_at_10", "all_support_at_10",
        ):
            missed_final_source[key] = 1.0
        _write_run(
            root,
            [_row("q1", 1.0, expand=arm == "neo4j"), missed_final_source],
        )
    report, per_query = build_report(roots, project_root=PROJECT_ROOT)
    assert report["questions"] == 2
    assert len(per_query) == 6
    by_arm = {row["arm"]: row for row in report["arms"]}
    assert by_arm["hybrid"]["hit_at_10"] == 0.5
    assert by_arm["hybrid"]["visible_hit_at_10"] == 1.0
    assert by_arm["hybrid"]["tokens_per_qa"]["total"] == 35
    assert by_arm["neo4j"]["graph_application_rate"] == 0.5
    assert "by_graph_opportunity" not in by_arm["neo4j"]
    assert by_arm["neo4j"]["by_evidence_structure"][0]["evidence_structure"] == "linked"

    # A missing paired question must fail rather than producing an unfair table.
    _write_run(roots["fs"], [_row("different", 1.0), _row("q2", 0.0)])
    with pytest.raises(ValueError, match="identical question IDs"):
        build_report(roots, project_root=PROJECT_ROOT)
