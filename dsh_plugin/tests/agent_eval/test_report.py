from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from dsh_plugin.agent_eval.report import (
    METRICS,
    _attach_primary_metrics,
    build_report,
    current_runtime_identity,
    load_arm_rollouts,
    percentile,
)
from dsh_plugin.agent_eval.runner import _repo_file_bundle_identity
from dsh_plugin.backend.retrieval_policy import retrieval_contract


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _test_runtime_identity(arm: str) -> dict:
    runtime_digest = hashlib.sha256(f"runtime:{arm}".encode()).hexdigest()
    plugin_digest = hashlib.sha256(f"plugin:{arm}".encode()).hexdigest()
    return {
        "runtime": {
            "sha256": runtime_digest,
            "files": [{"path": f"runtime-{arm}", "sha256": runtime_digest}],
        },
        "compiled_plugin": {
            "sha256": plugin_digest,
            "files": [{"path": f"plugin-{arm}", "sha256": plugin_digest}],
        },
    }


@pytest.fixture(autouse=True)
def _stub_checked_out_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dsh_plugin.agent_eval.report.current_runtime_identity",
        lambda _project_root, arm: _test_runtime_identity(arm),
    )


def _row(item_id: str, score: float, *, expand: bool = False) -> dict:
    tools = ["skill", "docsqa_search"]
    if expand:
        tools.append("docsqa_expand")
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
        "actual_models": ["gpt-5.6-luna"],
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


def _write_run(root: Path, arm: str, rows: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    skill = PROJECT_ROOT / f"dsh_plugin/plugin/skills/{arm}/initial_skill.md"
    prepared = []
    for source in rows:
        row = dict(source)
        row["arm"] = arm
        row["skill_path"] = str(skill)
        row["skill_sha256"] = hashlib.sha256(skill.read_bytes()).hexdigest()
        row["evaluation_contract"] = {
            "retrieval": retrieval_contract(arm),
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
            "corpus": {"sha256": "c" * 64},
            "split": {"sha256": "s" * 64},
            "scoring": {"sha256": "g" * 64},
            "dependencies": {"sha256": "d" * 64},
            "provider": {"model": "gpt-5.6-luna"},
            **_test_runtime_identity(arm),
        }
        prepared.append(row)
        calls = []
        results = []
        for index, tool in enumerate(row["tool_sequence"]):
            call_id = f"{row['id']}-{index}"
            calls.append(
                {"type": "tool/call", "data": {"callId": call_id, "name": tool}}
            )
            results.append(
                {
                    "type": "tool/result",
                    "data": {
                        "message": {
                            "content": [
                                {
                                    "type": "tool-result",
                                    "toolCallId": call_id,
                                    "isError": tool in row.get("failed_tools", []),
                                }
                            ]
                        }
                    },
                }
            )
        conversation = root / "predictions" / row["id"] / "conversation.json"
        conversation.parent.mkdir(parents=True, exist_ok=True)
        conversation.write_text(json.dumps([*calls, *results]), encoding="utf-8")
    (root / "rollouts.json").write_text(json.dumps(prepared), encoding="utf-8")


def test_percentile_uses_linear_interpolation() -> None:
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert percentile([0, 10], 0.95) == pytest.approx(9.5)


def test_load_arm_rollouts_uses_embedded_outcomes_without_prediction_tree(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "id": "q1",
            "tool_outcomes": [
                {"call_id": "c1", "name": "docsqa_search", "successful": True},
                {"call_id": "c2", "name": "docsqa_expand", "is_error": True},
            ],
        }
    ]
    path = tmp_path / "rollouts.json"
    path.write_text(json.dumps(rows), encoding="utf-8")

    loaded = load_arm_rollouts(path)

    assert loaded[0]["persisted_tool_outcomes"] == [
        {"call_id": "c1", "tool": "docsqa_search", "success": True},
        {"call_id": "c2", "tool": "docsqa_expand", "success": False},
    ]


def test_artifact_replay_uses_recorded_identity_without_current_runtime(
    tmp_path: Path,
) -> None:
    roots = {arm: tmp_path / arm for arm in ("fs", "hybrid", "neo4j")}
    shared = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cpu",
        "corpus": {"sha256": "c" * 64},
        "split": {"sha256": "s" * 64},
        "scoring": {"sha256": "g" * 64},
        "dependencies": {"sha256": "d" * 64},
        "provider": {"model": "gpt-5.6-luna"},
    }
    for arm, root in roots.items():
        root.mkdir(parents=True)
        base = _row("q1", 1.0, expand=arm == "neo4j")
        row = {
            **base,
            "arm": arm,
            "skill_path": f"/historical/dsh_plugin/plugin/skills/{arm}/initial_skill.md",
            "skill_sha256": arm[0] * 64,
            "evaluation_contract": {
                "retrieval": retrieval_contract(arm),
                **shared,
                "runtime": {
                    "sha256": "r" * 64,
                    "files": [{"path": "historical-runtime", "sha256": "r" * 64}],
                },
                "compiled_plugin": {
                    "sha256": "p" * 64,
                    "files": [{"path": "historical-plugin", "sha256": "p" * 64}],
                },
            },
            "tool_outcomes": [
                {"name": tool, "successful": True}
                for tool in base["tool_sequence"]
            ],
        }
        (root / "rollouts.json").write_text(json.dumps([row]), encoding="utf-8")

    report, per_query = build_report(
        roots,
        project_root=tmp_path / "checkout-without-runtime",
        expected_question_ids=["q1"],
        small_trial=True,
        artifact_replay=True,
    )

    assert report["artifact_replay"] is True
    assert report["questions"] == 1
    assert len(per_query) == 3
    assert {arm["skill_sha256"] for arm in report["arms"]} == {
        "f" * 64,
        "h" * 64,
        "n" * 64,
    }
    assert any("artifact replay" in value.lower() for value in report["limitations"])


def test_attach_primary_metrics_scores_visible_ranking_independently() -> None:
    row = {
        "id": "q1",
        "qrel_ids": ["/docs/example"],
        "ranked_ids": ["/docs/example"],
        "citation_ranked_ids": ["/docs/example"],
        "visible_ranked_ids": ["/docs/other"],
        **{metric: 1.0 for metric in METRICS},
    }

    _attach_primary_metrics([row], {})

    assert row["hit_at_10"] == 1.0
    assert row["citation_hit_at_10"] == 1.0
    assert row["visible_hit_at_10"] == 0.0
    assert row["visible_ranked_ids"] == ["/docs/other"]

    # An explicitly empty visible ranking means that no tool evidence was
    # visible; it must not fall back to a successful final citation ranking.
    row["visible_ranked_ids"] = []
    _attach_primary_metrics([row], {})
    assert row["visible_hit_at_10"] == 0.0


def _write_fake_runtime(project_root: Path, arm: str) -> tuple[list[Path], list[Path]]:
    relative_runtime_paths = [
        "dsh_plugin/agent_eval/runner.py",
        "dsh_plugin/backend/http_contract.py",
        "dsh_plugin/backend/retrieval_policy.py",
        "dsh_plugin/backend/service.py",
        "dsh_plugin/plugin/package.json",
        "evaluation/kbbench/plugin_eval.py",
        "evaluation/kbbench/retrieval.py",
        "dsh_plugin/harness/model-openai.patch.yml",
        "dsh_plugin/harness/docsqa_dsh_common.patch.yml",
        f"dsh_plugin/harness/docsqa_{arm}_system.patch.yml",
    ]
    compiled_relative_paths = [
        "dsh_plugin/plugin/lib/index.js",
        "dsh_plugin/plugin/lib/types/index.js",
    ]
    for relative in [*relative_runtime_paths, *compiled_relative_paths]:
        path = project_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"content:{relative}\n", encoding="utf-8")
    compiled = [project_root / relative for relative in compiled_relative_paths]
    runtime = [
        *(project_root / relative for relative in relative_runtime_paths),
        *compiled,
    ]
    return runtime, compiled


def test_runtime_identity_matches_runner_nested_logical_bundle(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first-checkout"
    second_root = tmp_path / "second-checkout"
    first_runtime, first_compiled = _write_fake_runtime(first_root, "hybrid")
    _write_fake_runtime(second_root, "hybrid")

    identity = current_runtime_identity(first_root, "hybrid")
    assert set(identity) == {"runtime", "compiled_plugin"}
    assert identity["runtime"] == _repo_file_bundle_identity(
        first_runtime, first_root
    )
    assert identity["compiled_plugin"] == _repo_file_bundle_identity(
        first_compiled, first_root
    )
    assert any(
        row["path"] == "dsh_plugin/plugin/lib/types/index.js"
        for row in identity["compiled_plugin"]["files"]
    )
    assert all(
        not row["path"].startswith(str(first_root))
        for bundle in identity.values()
        for row in bundle["files"]
    )
    assert identity == current_runtime_identity(second_root, "hybrid")


def test_report_requires_paired_ids_and_keeps_graph_use_visible(tmp_path: Path) -> None:
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
            arm,
            [_row("q1", 1.0, expand=arm == "neo4j"), missed_final_source],
        )
    report, per_query = build_report(
        roots,
        project_root=PROJECT_ROOT,
        expected_question_ids=["q1", "q2"],
        small_trial=True,
    )
    assert report["questions"] == 2
    assert report["complete_matched_development_run"] is False
    assert "reportable" not in report
    assert len(per_query) == 6
    by_arm = {row["arm"]: row for row in report["arms"]}
    assert by_arm["hybrid"]["hit_at_10"] == 0.5
    assert by_arm["hybrid"]["visible_hit_at_10"] == 1.0
    assert by_arm["hybrid"]["tokens_per_qa"]["total"] == 35
    assert by_arm["neo4j"]["graph_application_rate"] == 0.5
    assert "by_graph_opportunity" not in by_arm["neo4j"]
    assert by_arm["neo4j"]["by_evidence_structure"][0]["evidence_structure"] == "linked"

    # A missing paired question must fail rather than producing an unfair table.
    _write_run(roots["fs"], "fs", [_row("different", 1.0), _row("q2", 0.0)])
    with pytest.raises(ValueError, match="explicit expected set"):
        build_report(
            roots,
            project_root=PROJECT_ROOT,
            expected_question_ids=["q1", "q2"],
            small_trial=True,
        )


def test_report_rejects_attempted_but_failed_graph_expansion(tmp_path: Path) -> None:
    roots = {arm: tmp_path / arm for arm in ("fs", "hybrid", "neo4j")}
    for arm, root in roots.items():
        row = _row("q1", 1.0, expand=arm == "neo4j")
        if arm == "neo4j":
            row["failed_tools"] = ["docsqa_expand"]
        _write_run(root, arm, [row])
    report, _ = build_report(
        roots,
        project_root=PROJECT_ROOT,
        expected_question_ids=["q1"],
        small_trial=True,
    )
    graph = next(row for row in report["arms"] if row["arm"] == "neo4j")
    assert graph["graph_expansion_attempt_rate"] == 1.0
    assert graph["graph_application_rate"] == 0.0
    assert graph["graph_expansion_successes_per_qa"] == 0.0


def test_report_rejects_duplicate_ids_and_stale_contracts(tmp_path: Path) -> None:
    roots = {arm: tmp_path / arm for arm in ("fs", "hybrid", "neo4j")}
    for arm, root in roots.items():
        rows = [_row("q1", 1.0)]
        if arm == "fs":
            rows.append(_row("q1", 1.0))
        _write_run(root, arm, rows)
    with pytest.raises(ValueError, match="duplicate rollout IDs"):
        build_report(
            roots,
            project_root=PROJECT_ROOT,
            expected_question_ids=["q1"],
            small_trial=True,
        )

    _write_run(roots["fs"], "fs", [_row("q1", 1.0)])
    rollout_path = roots["hybrid"] / "rollouts.json"
    rows = json.loads(rollout_path.read_text(encoding="utf-8"))[:1]
    rows[0]["evaluation_contract"]["runtime"]["sha256"] = "stale"
    rollout_path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime mismatch"):
        build_report(
            roots,
            project_root=PROJECT_ROOT,
            expected_question_ids=["q1"],
            small_trial=True,
        )

    _write_run(roots["hybrid"], "hybrid", [_row("q1", 1.0)])
    rows = json.loads(rollout_path.read_text(encoding="utf-8"))
    rows[0]["evaluation_contract"]["compiled_plugin"]["sha256"] = "stale"
    rollout_path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="compiled_plugin mismatch"):
        build_report(
            roots,
            project_root=PROJECT_ROOT,
            expected_question_ids=["q1"],
            small_trial=True,
        )


def test_report_requires_exact_arms_model_and_explicit_pilot_flag(tmp_path: Path) -> None:
    roots = {arm: tmp_path / arm for arm in ("fs", "hybrid", "neo4j")}
    for arm, root in roots.items():
        _write_run(root, arm, [_row("q1", 1.0)])

    with pytest.raises(ValueError, match="requires exactly 361 expected IDs"):
        build_report(
            roots,
            project_root=PROJECT_ROOT,
            expected_question_ids=["q1"],
        )
    with pytest.raises(ValueError, match="arms must be exactly"):
        build_report(
            {"fs": roots["fs"], "hybrid": roots["hybrid"]},
            project_root=PROJECT_ROOT,
            expected_question_ids=["q1"],
            small_trial=True,
        )

    rollout_path = roots["fs"] / "rollouts.json"
    rows = json.loads(rollout_path.read_text(encoding="utf-8"))
    rows[0]["actual_models"] = ["another-model"]
    rollout_path.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="require actual_models"):
        build_report(
            roots,
            project_root=PROJECT_ROOT,
            expected_question_ids=["q1"],
            small_trial=True,
        )
