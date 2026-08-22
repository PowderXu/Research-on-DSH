from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILLOPT_ROOT = PROJECT_ROOT / "vendor" / "SkillOpt"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SKILLOPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILLOPT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from kbbench.skillopt_github_docs.dataloader import GitHubDocsDshDataLoader
from kbbench.skillopt_github_docs.adapter import GitHubDocsDshAdapter
from kbbench.skillopt_github_docs.credentials import load_openai_key_from_configured_env
from kbbench.skillopt_github_docs.rollout import (
    DshCommandConfig,
    _dsh_gateway_url,
    _dsh_invocation,
    _write_dsh_gateway_patch,
    run_batch,
)
from kbbench.skillopt_github_docs.scorer import (
    GitHubDocsSourceResolver,
    score_ranked_sources,
)
from kbbench.skillopt_github_docs.validation import (
    SkillCandidateError,
    load_leakage_markers,
    validate_skill_candidate,
)
from materialize_github_docs_skillopt_split import build_split


SPLIT_DIR = PROJECT_ROOT / "evaluation" / "skillopt" / "github_docs_v2_split"
CORPUS = PROJECT_ROOT / "evaluation" / "github_docs_v2" / "corpus.jsonl"


def test_frozen_split_preserves_original_test_and_category_coverage() -> None:
    rows = [
        json.loads(line)
        for line in (PROJECT_ROOT / "evaluation/github_docs_v2/questions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    split = build_split(rows, seed=20260821, train_size=55)
    assert {name: len(items) for name, items in split.items()} == {
        "train": 55,
        "val": 27,
        "test": 246,
    }
    source_test_ids = {
        str(row["question_id"]) for row in rows if row["split"] == "test"
    }
    assert {str(row["question_id"]) for row in split["test"]} == source_test_ids
    for name in ("train", "val"):
        categories = {row["evidence_category"] for row in split[name]}
        assert "multi_page_linked" in categories
        assert "single_page_composed_anchor" in categories


def test_skillopt_loader_reads_frozen_arrays() -> None:
    loader = GitHubDocsDshDataLoader(
        split_dir=str(SPLIT_DIR), split_mode="split_dir", seed=20260821
    )
    loader.setup({"split_dir": str(SPLIT_DIR), "split_mode": "split_dir"})
    assert loader.get_train_size() == 55
    assert len(loader.val_items) == 27
    assert len(loader.test_items) == 246
    assert all(item["qrel_ids"] and item["task_type"] for item in loader.train_items)


def test_adapter_can_select_frozen_ids_without_rewriting_the_split() -> None:
    adapter = GitHubDocsDshAdapter(
        split_dir=str(SPLIT_DIR),
        split_mode="split_dir",
        arm="neo4j",
        corpus_path=str(CORPUS),
        eval_item_ids="26686",
        start_backend=False,
        runner=lambda *_args, **_kwargs: {},
    )
    adapter.setup({"split_dir": str(SPLIT_DIR), "split_mode": "split_dir"})
    assert [item["id"] for item in adapter.dataloader.train_items] == ["26686"]
    assert adapter.dataloader.val_items == []
    assert adapter.dataloader.test_items == []


def test_source_resolver_accepts_doc_ids_urls_and_repo_paths() -> None:
    resolver = GitHubDocsSourceResolver(CORPUS)
    doc_id = "/graphql/guides/using-the-graphql-api-for-discussions"
    assert resolver.resolve(doc_id) == doc_id
    assert (
        resolver.resolve(
            "https://docs.github.com/en/graphql/guides/using-the-graphql-api-for-discussions#x"
        )
        == doc_id
    )
    assert (
        resolver.resolve(
            "content/graphql/guides/using-the-graphql-api-for-discussions.md"
        )
        == doc_id
    )


def test_ir_score_rewards_rank_and_complete_multi_page_coverage() -> None:
    metrics = score_ranked_sources(["/b", "/a"], ["/a", "/b"])
    assert metrics["hit_at_1"] == 1.0
    assert metrics["recall_at_1"] == 0.5
    assert metrics["recall_at_10"] == 1.0
    assert metrics["all_support_at_10"] == 1.0
    assert metrics["ndcg_at_10"] == pytest.approx(1.0)


def test_candidate_validator_blocks_contract_edits_and_benchmark_leakage() -> None:
    skill = (PROJECT_ROOT / "dsh-techdocs-plugin/skills/hybrid/initial_skill.md").read_text(
        encoding="utf-8"
    )
    markers = load_leakage_markers(SPLIT_DIR)
    parsed = validate_skill_candidate(
        skill,
        arm="hybrid",
        expected_name="github-docs-hybrid",
        leakage_markers=markers,
    )
    assert parsed["metadata"]["arm"] == "hybrid"
    with pytest.raises(SkillCandidateError, match="tool name"):
        validate_skill_candidate(
            skill.replace("`techdocs_fetch`", "the fetch operation"),
            arm="hybrid",
            expected_name="github-docs-hybrid",
            leakage_markers=markers,
        )
    leaked = skill + "\n" + sorted(markers["all_qrel_ids"])[0]
    with pytest.raises(SkillCandidateError, match="document IDs"):
        validate_skill_candidate(
            leaked,
            arm="hybrid",
            expected_name="github-docs-hybrid",
            leakage_markers=markers,
        )


def test_rollout_persists_skillopt_conversation_without_exposing_qrels(
    tmp_path: Path,
) -> None:
    items = json.loads((SPLIT_DIR / "train/items.json").read_text(encoding="utf-8"))
    item = dict(items[0])
    item.update(
        id=str(item["question_id"]),
        question=item["query"],
        task_type=item["evidence_category"],
    )
    resolver = GitHubDocsSourceResolver(CORPUS)
    source_path = next(
        json.loads(line)["source_path"]
        for line in CORPUS.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["doc_id"] == item["qrel_ids"][0]
    )

    def fake_runner(_item: dict, _skill_path: Path, _case_dir: Path) -> dict:
        return {
            "answer": "grounded",
            "sources": [source_path],
            "latency_seconds": 0.01,
            "return_code": 0,
            "execution_error": "",
            "parse_error": "",
            "session_file": None,
            "conversation": [{"type": "tool/call", "data": "techdocs_search"}],
            "usage": {"total": 10},
            "actual_models": ["fake"],
            "tool_sequence": ["skill", "techdocs_search"],
            "model_steps": 1,
            "skill_loaded": True,
        }

    skill = (PROJECT_ROOT / "dsh-techdocs-plugin/skills/hybrid/initial_skill.md").read_text(
        encoding="utf-8"
    )
    rows = run_batch(
        items=[item],
        skill_content=skill,
        out_root=str(tmp_path),
        arm="hybrid",
        resolver=resolver,
        runner=fake_runner,
        expected_name="github-docs-hybrid",
        leakage_markers=load_leakage_markers(SPLIT_DIR),
    )
    assert rows[0]["hard"] == 1.0
    assert rows[0]["ranked_ids"] == [item["qrel_ids"][0]]
    conversation = tmp_path / "predictions" / item["id"] / "conversation.json"
    assert conversation.exists()
    user_prompt = (tmp_path / "predictions" / item["id"] / "target_user_prompt.txt").read_text()
    assert item["qrel_ids"][0] not in user_prompt


def test_credential_bridge_parses_env_as_data(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("UNRELATED=x\nexport OPENAI_API_KEY='test-secret'\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("KBBENCH_OPENAI_ENV_FILE", str(env_file))
    assert load_openai_key_from_configured_env()
    assert __import__("os").environ["OPENAI_API_KEY"] == "test-secret"
    assert __import__("os").environ["AZURE_OPENAI_API_KEY"] == "test-secret"


def test_dsh_invocation_uses_default_provider_unless_gateway_is_configured(
    tmp_path: Path,
) -> None:
    config = DshCommandConfig(
        arm="hybrid",
        dsh_binary=tmp_path / "dsh",
        dsh_home=tmp_path / "dsh-home",
        workspace=tmp_path / "workspace",
        model_patch=tmp_path / "model.patch.yml",
        common_patch=tmp_path / "common.patch.yml",
        arm_patch=tmp_path / "arm.patch.yml",
    )
    item = {"id": "q1", "question": "question"}
    candidate = tmp_path / "candidate.patch.yml"
    config.model_patch.write_text(
        """- id: agent-default-model
  config:
    provider: openai
    model: gpt-test
- id: llm-pi-ai
  config:
    providers:
      openai:
        apiKeyEnv: OPENAI_API_KEY
        models:
          - id: gpt-test
            contextWindow: 1000
            maxTokens: 100
""",
        encoding="utf-8",
    )

    assert _dsh_gateway_url({}) == ""
    assert _write_dsh_gateway_patch(tmp_path, "", config.model_patch) is None
    default_command = _dsh_invocation(config, item, candidate, None)
    assert "provider-model.patch.yml" not in " ".join(default_command)

    gateway_url = "https://gateway.example/v1"
    assert _dsh_gateway_url({"OPENAI_BASE_URL": gateway_url}) == gateway_url
    assert (
        _dsh_gateway_url(
            {
                "OPENAI_BASE_URL": "https://fallback.example/v1",
                "DSH_OPENAI_BASE_URL": gateway_url,
            }
        )
        == gateway_url
    )
    gateway_patch = _write_dsh_gateway_patch(
        tmp_path, gateway_url, config.model_patch
    )
    assert gateway_patch is not None
    gateway_document = __import__("yaml").safe_load(
        gateway_patch.read_text(encoding="utf-8")
    )
    provider = next(row for row in gateway_document if row["id"] == "llm-pi-ai")
    openai = provider["config"]["providers"]["openai"]
    assert openai["baseURL"] == gateway_url
    assert openai["apiKeyEnv"] == "OPENAI_API_KEY"
    assert openai["models"][0]["id"] == "gpt-test"
    custom_command = _dsh_invocation(config, item, candidate, gateway_patch)
    assert str(gateway_patch.resolve()) in custom_command
    assert str(config.model_patch.resolve()) not in custom_command
