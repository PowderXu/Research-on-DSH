from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from kbbench.scoring import GitHubDocsSourceResolver

from dsh_plugin.agent_eval.runner import (
    DshCommandConfig,
    DshCommandRunner,
    _dsh_invocation,
    _files_sha256,
    _stable_service_identity,
    _trace_summary,
    _validate_agent_json,
    _visible_sources,
    load_split_items,
    parse_agent_json,
    preflight_corpus_workspace,
    run_batch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _test_item_and_resolver(
    tmp_path: Path,
) -> tuple[dict, GitHubDocsSourceResolver]:
    doc_id = "github-docs::/example"
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        json.dumps(
            {
                "doc_id": doc_id,
                "project": "github-docs",
                "source_doc_id": "/example",
                "source_path": "github-docs/content/example.md",
                "repository_source_path": "content/example.md",
                "route": doc_id,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    item = {
        "id": "github-docs::q1",
        "question_id": "github-docs::q1",
        "query": "How does the example work?",
        "question": "How does the example work?",
        "evidence_category": "single_page",
        "task_type": "single_page",
        "dataset": "github-docs",
        "qrel_ids": [doc_id],
        "qrel_count": 1,
        "evidence_structure": "single",
    }
    return item, GitHubDocsSourceResolver(corpus_path)


def test_load_split_items_uses_canonical_split_without_optimizer_copy(
    tmp_path: Path,
) -> None:
    split = tmp_path / "validation.json"
    split.write_text(
        json.dumps(
            [
                {
                    "question_id": f"q{index}",
                    "query": f"question {index}",
                    "evidence_category": "single_page",
                }
                for index in range(3)
            ]
        ),
        encoding="utf-8",
    )
    items = load_split_items(split, limit=2)
    assert len(items) == 2
    assert all(item["id"] == str(item["question_id"]) for item in items)
    assert all(item["question"] == item["query"] for item in items)
    assert all(item["task_type"] == item["evidence_category"] for item in items)


def test_trace_summary_requires_successful_skill_result() -> None:
    failed = [
        {
            "type": "tool/call",
            "data": {"name": "skill", "callId": "skill-1"},
        },
        {
            "type": "tool/result",
            "data": {
                "message": {
                    "content": [
                        {
                            "type": "tool-result",
                            "toolCallId": "skill-1",
                            "isError": True,
                        }
                    ]
                }
            },
        },
    ]
    successful = [
        failed[0],
        {
            "type": "tool/result",
            "data": {
                "message": {
                    "content": [
                        {
                            "type": "tool-result",
                            "toolCallId": "skill-1",
                            "isError": False,
                        }
                    ]
                }
            },
        },
    ]

    assert _trace_summary(failed)["skill_loaded"] is False
    assert _trace_summary(successful)["skill_loaded"] is True


def _tool_events(
    name: str,
    call_id: str,
    arguments: dict,
    text: str,
    *,
    is_error: bool = False,
    meta: dict | None = None,
) -> list[dict]:
    return [
        {
            "type": "tool/call",
            "data": {
                "name": name,
                "callId": call_id,
                "arguments": json.dumps(arguments),
            },
        },
        {
            "type": "tool/result",
            "data": {
                "message": {
                    "source": {"callId": call_id},
                    "content": [
                        {
                            "type": "tool-result",
                            "toolCallId": call_id,
                            "isError": is_error,
                            "content": [{"type": "text", "text": text}],
                        }
                    ],
                },
                "meta": meta or {},
            },
        },
    ]


def test_trace_matches_arguments_results_and_exact_arm_skill() -> None:
    events = [
        *_tool_events(
            "skill",
            "skill-1",
            {"name": "docsqa-neo4j"},
            '<skill_content name="docsqa-neo4j">instructions</skill_content>',
        ),
        *_tool_events(
            "docsqa_search",
            "search-1",
            {"query": "release notes", "limit": 10},
            '<docsqa-evidence query-id="q"><x>\nURI: '
            "viking://resources/docsqa/github-docs::/rest/releases\n</x>",
        ),
    ]

    summary = _trace_summary(events, expected_arm="neo4j")

    assert summary["expected_skill_loaded"] is True
    assert summary["retrieval_succeeded"] is True
    assert summary["tool_outcomes"][1]["call_id"] == "search-1"
    assert summary["tool_outcomes"][1]["arguments"] == {
        "query": "release notes",
        "limit": 10,
    }
    assert summary["tool_outcomes"][1]["result_matched"] is True
    assert summary["tool_outcomes"][1]["successful"] is True
    assert "docsqa_expand" not in summary["tool_sequence"]


def test_trace_rejects_wrong_skill_and_accepts_linked_grep_plus_read() -> None:
    wrong_skill = _tool_events(
        "skill",
        "skill-1",
        {"name": "docsqa-hybrid"},
        '<skill_content name="docsqa-hybrid">instructions</skill_content>',
    )
    positive_grep = _tool_events(
        "grep",
        "grep-1",
        {"pattern": "release", "path": "."},
        "Found 1 match",
        meta={
            "shape": "matches",
            "total": 1,
            "files": [{"path": "github-docs/content/release.md"}],
        },
    )
    positive_read = _tool_events(
        "read",
        "read-1",
        {"file_path": "github-docs/content/release.md"},
        "<path>github-docs/content/release.md</path>\nevidence",
        meta={"lines": [{"number": 1, "text": "evidence"}]},
    )

    without_read = _trace_summary(
        [*wrong_skill, *positive_grep], expected_arm="fs"
    )
    with_read = _trace_summary(
        [
            *_tool_events(
                "skill",
                "skill-2",
                {"name": "docsqa-fs"},
                '<skill_content name="docsqa-fs">instructions</skill_content>',
            ),
            *positive_grep,
            *positive_read,
        ],
        expected_arm="fs",
    )

    assert without_read["expected_skill_loaded"] is False
    assert without_read["retrieval_succeeded"] is False
    assert with_read["expected_skill_loaded"] is True
    assert with_read["retrieval_succeeded"] is True
    assert with_read["fs_retrieval_links"] == [
        {
            "discovery_call_id": "grep-1",
            "discovery_tool": "grep",
            "discovered_path": "github-docs/content/release.md",
            "read_call_id": "read-1",
            "read_path": "github-docs/content/release.md",
        }
    ]


def test_trace_accepts_linked_glob_plus_read() -> None:
    skill = _tool_events(
        "skill",
        "skill-1",
        {"name": "docsqa-fs"},
        '<skill_content name="docsqa-fs">instructions</skill_content>',
    )
    glob = _tool_events(
        "glob",
        "glob-1",
        {"pattern": "**/*release*.md"},
        "./github-docs/content/releases/release-notes.md",
        meta={
            "shape": "paths",
            "total": 1,
            "paths": ["./github-docs/content/releases/release-notes.md"],
        },
    )
    read = _tool_events(
        "read",
        "read-1",
        {"file_path": "github-docs/content/releases/release-notes.md"},
        "<path>github-docs/content/releases/release-notes.md</path>\nevidence",
        meta={"lines": [{"number": 1, "text": "evidence"}]},
    )

    summary = _trace_summary([*skill, *glob, *read], expected_arm="fs")

    assert summary["retrieval_succeeded"] is True
    assert summary["successful_retrieval_tools"] == ["glob", "read"]
    assert summary["fs_retrieval_links"][0]["discovery_tool"] == "glob"
    assert summary["tool_outcomes"][1]["discovered_path_count"] == 1
    assert summary["tool_outcomes"][1]["discovered_paths"] == [
        "github-docs/content/releases/release-notes.md"
    ]


def test_trace_rejects_unlinked_or_late_fs_discovery() -> None:
    skill = _tool_events(
        "skill",
        "skill-1",
        {"name": "docsqa-fs"},
        '<skill_content name="docsqa-fs">instructions</skill_content>',
    )
    glob = _tool_events(
        "glob",
        "glob-1",
        {"pattern": "**/*.md"},
        "github-docs/content/a.md",
        meta={"shape": "paths", "total": 1, "paths": ["github-docs/content/a.md"]},
    )
    unrelated_read = _tool_events(
        "read",
        "read-1",
        {"file_path": "github-docs/content/b.md"},
        "<path>github-docs/content/b.md</path>\nevidence",
        meta={"lines": [{"number": 1, "text": "evidence"}]},
    )

    unrelated = _trace_summary(
        [*skill, *glob, *unrelated_read], expected_arm="fs"
    )
    discovery_after_read = _trace_summary(
        [*skill, *unrelated_read, *glob], expected_arm="fs"
    )

    assert unrelated["retrieval_succeeded"] is False
    assert unrelated["fs_retrieval_links"] == []
    assert discovery_after_read["retrieval_succeeded"] is False


def test_visible_sources_preserves_namespaced_viking_uri() -> None:
    uri = (
        "viking://resources/docsqa/github-docs::/repositories/releases/notes"
        "#Generate-release-notes"
    )
    events = _tool_events(
        "docsqa_search",
        "search-1",
        {"query": "notes"},
        f'<docsqa-evidence query-id="q">\nURI: {uri}\n</docsqa-evidence>',
    )

    assert _visible_sources(events) == [uri]


def test_agent_output_schema_is_strict_and_records_invalid_sources() -> None:
    value, errors, invalid = _validate_agent_json(
        '{"answer":"ok","sources":["a","a",3],"extra":true}'
    )
    assert value == {"answer": "ok", "sources": ["a"]}
    assert "unexpected_keys:extra" in errors
    assert "invalid_sources" in errors
    assert [row["reason"] for row in invalid] == ["duplicate", "not_string"]

    _value, too_many_errors, _invalid = _validate_agent_json(
        json.dumps({"answer": "ok", "sources": [str(i) for i in range(11)]})
    )
    assert "too_many_sources" in too_many_errors


def test_logical_file_hash_ignores_host_absolute_root(tmp_path: Path) -> None:
    first_root = tmp_path / "one"
    second_root = tmp_path / "two"
    first_root.mkdir()
    second_root.mkdir()
    (first_root / "runtime.js").write_text("export const version = 1;\n")
    (second_root / "runtime.js").write_text("export const version = 1;\n")

    assert _files_sha256(
        [first_root / "runtime.js"], logical_root=first_root
    ) == _files_sha256([second_root / "runtime.js"], logical_root=second_root)


def test_service_identity_keeps_graph_snapshot_but_drops_runtime_noise() -> None:
    identity = _stable_service_identity(
        {
            "status": "ok",
            "arm": "neo4j",
            "revision": "sha256:corpus",
            "documents": 10,
            "chunks": 20,
            "method": "hybrid + graph",
            "retrievalContract": {"final_result_limit": 10},
            "graphSnapshot": {"snapshot_sha256": "graph"},
            "buildSeconds": 12.5,
            "dataRoot": "/host/private/path",
        }
    )

    assert identity["graph_snapshot"] == {"snapshot_sha256": "graph"}
    assert identity["retrieval_contract"] == {"final_result_limit": 10}
    assert "buildSeconds" not in identity
    assert "dataRoot" not in identity


def test_direct_rollout_scores_and_persists_one_result(tmp_path: Path) -> None:
    item, resolver = _test_item_and_resolver(tmp_path)

    def fake_runner(_item: dict, _skill: Path, _case: Path) -> dict:
        return {
            "answer": "grounded",
            "sources": [item["qrel_ids"][0]],
            "visible_sources": [item["qrel_ids"][0]],
            "return_code": 0,
            "parse_error": "",
            "execution_error": "",
            "skill_loaded": True,
            "expected_skill_loaded": True,
            "retrieval_succeeded": True,
            "model_steps": 1,
            "latency_seconds": 0.01,
            "usage": {"total": 10},
            "tool_sequence": ["skill", "docsqa_search"],
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


def test_direct_rollout_resume_reuses_completed_result(tmp_path: Path) -> None:
    item, resolver = _test_item_and_resolver(tmp_path)
    calls = 0

    def fake_runner(_item: dict, _skill: Path, _case: Path) -> dict:
        nonlocal calls
        calls += 1
        return {
            "answer": "grounded",
            "sources": [item["qrel_ids"][0]],
            "visible_sources": [item["qrel_ids"][0]],
            "return_code": 0,
            "parse_error": "",
            "execution_error": "",
            "skill_loaded": True,
            "expected_skill_loaded": True,
            "retrieval_succeeded": True,
            "model_steps": 1,
            "latency_seconds": 0.01,
            "usage": {"total": 10},
            "tool_sequence": ["skill", "docsqa_search"],
            "actual_models": ["fake"],
            "conversation": [],
        }

    arguments = {
        "items": [item],
        "skill_path": PROJECT_ROOT / "dsh_plugin/plugin/skills/hybrid/initial_skill.md",
        "out_root": str(tmp_path),
        "arm": "hybrid",
        "resolver": resolver,
        "runner": fake_runner,
    }
    first = run_batch(**arguments)
    second = run_batch(**arguments, resume=True)

    assert calls == 1
    assert second == first


def test_direct_rollout_resume_retries_changed_retrieval_contract(
    tmp_path: Path,
) -> None:
    item, resolver = _test_item_and_resolver(tmp_path)
    calls = 0

    def fake_runner(_item: dict, _skill: Path, _case: Path) -> dict:
        nonlocal calls
        calls += 1
        return {
            "answer": "grounded",
            "sources": [item["qrel_ids"][0]],
            "visible_sources": [item["qrel_ids"][0]],
            "return_code": 0,
            "parse_error": "",
            "execution_error": "",
            "skill_loaded": True,
            "expected_skill_loaded": True,
            "retrieval_succeeded": True,
            "model_steps": 1,
            "latency_seconds": 0.01,
            "usage": {"total": 10},
            "tool_sequence": ["skill", "docsqa_search"],
            "actual_models": ["fake"],
            "conversation": [],
        }

    arguments = {
        "items": [item],
        "skill_path": PROJECT_ROOT / "dsh_plugin/plugin/skills/hybrid/initial_skill.md",
        "out_root": str(tmp_path),
        "arm": "hybrid",
        "resolver": resolver,
        "runner": fake_runner,
    }
    run_batch(**arguments, evaluation_contract={"retrieval_depth": 350})
    run_batch(
        **arguments,
        evaluation_contract={"retrieval_depth": 50},
        resume=True,
    )

    assert calls == 2


def test_direct_rollout_resume_retries_changed_qrels(tmp_path: Path) -> None:
    item, resolver = _test_item_and_resolver(tmp_path)
    changed = {**item, "qrel_ids": ["github-docs::/definitely-different"]}
    calls = 0

    def fake_runner(_item: dict, _skill: Path, _case: Path) -> dict:
        nonlocal calls
        calls += 1
        return {
            "answer": "grounded",
            "sources": [item["qrel_ids"][0]],
            "visible_sources": [item["qrel_ids"][0]],
            "return_code": 0,
            "parse_error": "",
            "execution_error": "",
            "skill_loaded": True,
            "expected_skill_loaded": True,
            "retrieval_succeeded": True,
            "model_steps": 1,
            "latency_seconds": 0.01,
            "usage": {"total": 10},
            "tool_sequence": ["skill", "docsqa_search"],
            "actual_models": ["fake"],
            "conversation": [],
        }

    arguments = {
        "skill_path": PROJECT_ROOT / "dsh_plugin/plugin/skills/hybrid/initial_skill.md",
        "out_root": str(tmp_path),
        "arm": "hybrid",
        "resolver": resolver,
        "runner": fake_runner,
    }
    run_batch(items=[item], **arguments)
    run_batch(items=[changed], **arguments, resume=True)

    assert calls == 2


def test_direct_rollout_rejects_and_records_unresolved_sources(tmp_path: Path) -> None:
    item, resolver = _test_item_and_resolver(tmp_path)

    def fake_runner(_item: dict, _skill: Path, _case: Path) -> dict:
        return {
            "answer": "unsupported citation",
            "sources": ["github-docs::/not-in-the-pinned-corpus"],
            "visible_sources": [],
            "return_code": 0,
            "parse_error": "",
            "execution_error": "",
            "skill_loaded": True,
            "expected_skill_loaded": True,
            "retrieval_succeeded": True,
            "model_steps": 1,
            "latency_seconds": 0.01,
            "usage": {"total": 10},
            "tool_sequence": ["skill", "docsqa_search"],
            "actual_models": ["fake"],
            "conversation": [],
            "backend_events": [{"operation": "search", "ranked_ids": []}],
            "provider_mode": "default",
        }

    rows = run_batch(
        items=[item],
        skill_path=PROJECT_ROOT / "dsh_plugin/plugin/skills/hybrid/initial_skill.md",
        out_root=str(tmp_path),
        arm="hybrid",
        resolver=resolver,
        runner=fake_runner,
    )

    assert rows[0]["agent_ok"] is False
    assert rows[0]["hard"] == 0
    assert rows[0]["unresolved_sources"] == [
        "github-docs::/not-in-the-pinned-corpus"
    ]
    assert rows[0]["fail_reason"] == "unresolved_sources"
    assert rows[0]["backend_events"][0]["operation"] == "search"
    assert rows[0]["provider_mode"] == "default"


def test_direct_rollout_resume_retries_failed_result(tmp_path: Path) -> None:
    item, resolver = _test_item_and_resolver(tmp_path)
    calls = 0

    def fake_runner(_item: dict, _skill: Path, _case: Path) -> dict:
        nonlocal calls
        calls += 1
        success = calls > 1
        return {
            "answer": "grounded" if success else "",
            "sources": [item["qrel_ids"][0]] if success else [],
            "visible_sources": [],
            "return_code": 0 if success else 1,
            "parse_error": "",
            "execution_error": "" if success else "transient",
            "skill_loaded": success,
            "expected_skill_loaded": success,
            "retrieval_succeeded": success,
            "model_steps": 1,
            "latency_seconds": 0.01,
            "usage": {"total": 10},
            "tool_sequence": ["skill"] if success else [],
            "actual_models": ["fake"],
            "conversation": [],
        }

    arguments = {
        "items": [item],
        "skill_path": PROJECT_ROOT / "dsh_plugin/plugin/skills/hybrid/initial_skill.md",
        "out_root": str(tmp_path),
        "arm": "hybrid",
        "resolver": resolver,
        "runner": fake_runner,
    }
    first = run_batch(**arguments)
    second = run_batch(**arguments, resume=True)

    assert first[0]["agent_ok"] is False
    assert second[0]["agent_ok"] is True
    assert calls == 2


def test_agent_json_and_command_contract(tmp_path: Path) -> None:
    assert parse_agent_json('{"answer":"ok","sources":[]}')["answer"] == "ok"
    with pytest.raises(ValueError, match="invalid_json"):
        parse_agent_json('```json\n{"answer":"ok","sources":[]}\n```')
    files = {
        name: tmp_path / name
        for name in ("dsh", "home", "workspace", "model.yml", "common.yml", "arm.yml", "corpus.jsonl")
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
        corpus_path=files["corpus.jsonl"],
    )
    command = _dsh_invocation(
        config,
        {"id": "q1", "question": "How?"},
        candidate,
        None,
    )
    assert command.count("--patch") == 4
    assert command[-1].count("Question:\nHow?") == 1
    assert "(fs," not in command[-1]


def _preflight_config(tmp_path: Path) -> DshCommandConfig:
    from dsh_plugin.backend.corpus_workspace import materialize_corpus_workspace

    corpus = [{"source_path": "project/nested/guide.md", "rendered_text": "Guide evidence."}]
    workspace = tmp_path / "documents"
    materialize_corpus_workspace(corpus, workspace)
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(json.dumps(corpus[0]) + "\n", encoding="utf-8")
    return DshCommandConfig(
        arm="fs", dsh_binary=corpus_path, dsh_home=tmp_path,
        workspace=workspace, model_patch=corpus_path, common_patch=corpus_path,
        arm_patch=corpus_path, corpus_path=corpus_path,
    )


def test_preflight_passes_only_corpus_content_to_official_tools(tmp_path: Path, monkeypatch) -> None:
    config = _preflight_config(tmp_path)
    requests = []

    def official_check(command, **kwargs):
        requests.append((command, json.loads(kwargs["input"])))
        return subprocess.CompletedProcess(command, 0, json.dumps({
            "ok": True, "expected_document_count": 1, "searchable_document_count": 1,
            "glob_document_count": 1, "probe_count": 1,
        }), "")

    monkeypatch.setattr("dsh_plugin.agent_eval.runner.subprocess.run", official_check)
    result = preflight_corpus_workspace(config)
    assert result["document_count"] == 1
    assert result["protocol"] == "exact-corpus-official-fs-v1"
    assert len(requests) == 1
    assert requests[0][0][-1].endswith("scripts/preflight_fs.ts")
    assert requests[0][1] == {
        "workspace": str(config.workspace.resolve()),
        "documents": [{"path": "project/nested/guide.md", "text": "Guide evidence."}],
    }


@pytest.mark.parametrize("payload", [
    {"ok": False},
    {"ok": True, "expected_document_count": 1, "searchable_document_count": 0,
     "glob_document_count": 1, "probe_count": 1},
    {"ok": True, "expected_document_count": 1, "searchable_document_count": 1,
     "glob_document_count": 1, "probe_count": 0},
])
def test_preflight_fails_closed_on_tool_visibility_failure(tmp_path: Path, monkeypatch, payload) -> None:
    config = _preflight_config(tmp_path)
    monkeypatch.setattr(
        "dsh_plugin.agent_eval.runner.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, json.dumps(payload), ""),
    )
    with pytest.raises(ValueError, match="preflight"):
        preflight_corpus_workspace(config)


def test_runner_rejects_workspace_drift_before_any_subprocess(tmp_path: Path, monkeypatch) -> None:
    config = _preflight_config(tmp_path)
    (config.workspace / "answers.json").write_text("hidden answer data", encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("no tools or model should run after workspace validation fails")

    monkeypatch.setattr("dsh_plugin.agent_eval.runner.subprocess.run", forbidden)
    with pytest.raises(ValueError, match="extra file"):
        DshCommandRunner(config)


def test_preflight_rejects_post_probe_content_change(tmp_path: Path, monkeypatch) -> None:
    config = _preflight_config(tmp_path)

    def changed(command, **kwargs):
        (config.workspace / "project/nested/guide.md").write_text("Changed", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, json.dumps({
            "ok": True, "expected_document_count": 1, "searchable_document_count": 1,
            "glob_document_count": 1, "probe_count": 1,
        }), "")

    monkeypatch.setattr("dsh_plugin.agent_eval.runner.subprocess.run", changed)
    with pytest.raises(ValueError, match="content mismatch"):
        preflight_corpus_workspace(config)
