from __future__ import annotations

import json
from pathlib import Path

from kbbench.scoring import GitHubDocsSourceResolver

from dsh_plugin.agent_eval.runner import (
    DshCommandConfig,
    _dsh_invocation,
    load_split_items,
    parse_agent_json,
    run_batch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "evaluation/dataset/data"


def test_load_split_items_uses_canonical_split_without_optimizer_copy() -> None:
    items = load_split_items(DATA / "splits/validation.json", limit=2)
    assert len(items) == 2
    assert all(item["id"] == str(item["question_id"]) for item in items)
    assert all(item["question"] == item["query"] for item in items)
    assert all(item["task_type"] == item["evidence_category"] for item in items)


def test_direct_rollout_scores_and_persists_one_result(tmp_path: Path) -> None:
    item = load_split_items(DATA / "splits/train.json", limit=1)[0]
    resolver = GitHubDocsSourceResolver(DATA / "corpus.jsonl")

    def fake_runner(_item: dict, _skill: Path, _case: Path) -> dict:
        return {
            "answer": "grounded",
            "sources": [item["qrel_ids"][0]],
            "visible_sources": [item["qrel_ids"][0]],
            "return_code": 0,
            "parse_error": "",
            "execution_error": "",
            "skill_loaded": True,
            "model_steps": 1,
            "latency_seconds": 0.01,
            "usage": {"total": 10},
            "tool_sequence": ["skill", "techdocs_search"],
            "actual_models": ["fake"],
            "conversation": [],
        }

    rows = run_batch(
        items=[item],
        skill_path=PROJECT_ROOT / "dsh_plugin/plugin/skills/hybrid/initial_skill.md",
        out_root=str(tmp_path),
        arm="hybrid",
        resolver=resolver,
        runner=fake_runner,
    )
    assert rows[0]["hit_at_10"] == 1.0
    assert json.loads((tmp_path / "rollouts.json").read_text())[0]["id"] == item["id"]
    assert not (tmp_path / "candidate_skills").exists()


def test_agent_json_and_command_contract(tmp_path: Path) -> None:
    assert parse_agent_json('```json\n{"answer":"ok","sources":[]}\n```')["answer"] == "ok"
    files = {
        name: tmp_path / name
        for name in ("dsh", "home", "workspace", "model.yml", "common.yml", "arm.yml")
    }
    for path in files.values():
        path.mkdir() if path.name in {"home", "workspace"} else path.write_text("")
    candidate = tmp_path / "candidate.yml"
    candidate.write_text("")
    config = DshCommandConfig(
        arm="fs",
        dsh_binary=files["dsh"],
        dsh_home=files["home"],
        workspace=files["workspace"],
        model_patch=files["model.yml"],
        common_patch=files["common.yml"],
        arm_patch=files["arm.yml"],
    )
    command = _dsh_invocation(
        config,
        {"id": "q1", "question": "How?"},
        candidate,
        None,
    )
    assert command.count("--patch") == 4
    assert command[-1].count("Question:\nHow?") == 1
