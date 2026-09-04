from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import dsh_plugin.agent_eval.aspect_judge as aspect_judge_module

from dsh_plugin.agent_eval.aspect_judge import (
    AspectBatchJudgment,
    AspectScore,
    CandidateAspectJudgment,
    ClaimIssue,
    build_judge_prompt,
    compute_candidate_score,
    evaluate,
    load_judge_rubric,
    paired_bootstrap_deltas,
    replay_artifact_report,
    run_judgments,
    summarize,
    summarize_by,
    _usage_totals,
    _jsonl,
    _assert_mixed_inputs_unchanged,
    _capture_mixed_inputs,
    _validate_fs_preflight,
    _validate_question_linkage,
    _unique_index,
    _load_cached_call,
    _write_cached_call,
)


class _FakeResponses:
    def __init__(self, owner: "_FakeClient") -> None:
        self.owner = owner

    def parse(self, **kwargs):
        payload = json.loads(kwargs["input"][0]["content"])
        question_id = str(payload["question_id"])
        with self.owner.lock:
            self.owner.call_ids.append(question_id)
            self.owner.active += 1
            self.owner.max_active = max(self.owner.max_active, self.owner.active)
        try:
            time.sleep(self.owner.delay_seconds)
            if question_id in self.owner.fail_on:
                raise RuntimeError(f"fake failure for {question_id}")
            candidate_rows = []
            labels = [row["candidate"] for row in payload["candidates"]]
            if question_id in self.owner.invalid_on:
                labels = labels[:1]
            for label in labels:
                candidate_rows.append(
                    CandidateAspectJudgment(
                        candidate=label,
                        aspects=[
                            AspectScore(
                                aspect_id=str(aspect["aspect_id"]),
                                coverage=1.0,
                                support="full",
                                evidence_ids=list(aspect.get("evidence_ids") or []),
                                explanation="fake supported score",
                            )
                            for aspect in payload["frozen_aspects"]
                        ],
                        claim_issues=[],
                        invalid_citations=[],
                        overall_quality=5,
                        explanation="fake complete answer",
                    )
                )
            return SimpleNamespace(
                output_parsed=AspectBatchJudgment(candidates=candidate_rows),
                usage=SimpleNamespace(
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                    input_tokens_details=SimpleNamespace(cached_tokens=0),
                ),
            )
        finally:
            with self.owner.lock:
                self.owner.active -= 1


class _FakeClient:
    def __init__(
        self,
        *,
        delay_seconds: float = 0.01,
        fail_on: set[str] | None = None,
        invalid_on: set[str] | None = None,
    ) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.call_ids: list[str] = []
        self.delay_seconds = delay_seconds
        self.fail_on = fail_on or set()
        self.invalid_on = invalid_on or set()
        self.responses = _FakeResponses(self)


def _record() -> dict:
    return {
        "question_id": "q1",
        "question": "How is the task completed?",
        "reference_answer": "Complete the core operation and preserve the qualification.",
        "aspects": [
            {
                "aspect_id": "a1",
                "description": "Complete the core operation",
                "importance": 5,
                "weight": 0.625,
                "critical": True,
                "requirement_ids": ["R1"],
                "evidence_ids": ["d1#s1"],
                "evidence": [
                    {
                        "evidence_id": "d1#s1",
                        "doc_id": "d1",
                        "kind": "local_document_section",
                        "text": "Core operation evidence",
                    }
                ],
            },
            {
                "aspect_id": "a2",
                "description": "Preserve the qualification",
                "importance": 3,
                "weight": 0.375,
                "critical": False,
                "requirement_ids": ["R1"],
                "evidence_ids": ["d2#s1"],
                "evidence": [
                    {
                        "evidence_id": "d2#s1",
                        "doc_id": "d2",
                        "kind": "local_document_section",
                        "text": "Qualification evidence",
                    }
                ],
            },
        ],
    }


def _judgment_inputs(
    question_ids: list[str],
) -> tuple[
    dict[str, dict],
    dict[str, dict[str, dict]],
    dict[str, dict],
]:
    aspects: dict[str, dict] = {}
    for question_id in question_ids:
        record = json.loads(json.dumps(_record()))
        record["question_id"] = question_id
        record["question"] = f"Question {question_id}"
        aspects[question_id] = record
    by_arm = {
        arm: {
            question_id: {
                "answer": f"{arm} answer for {question_id}",
                "ranked_ids": ["d1", "d2"],
                "agent_ok": True,
                "usage": {"total": 20},
                "latency_seconds": 0.5,
            }
            for question_id in question_ids
        }
        # Deliberately not ARM_ORDER: returned rows must not inherit dict order.
        for arm in ("neo4j", "fs", "hybrid")
    }
    corpus = {
        "d1": {"doc_id": "d1", "title": "One", "rendered_text": "First"},
        "d2": {"doc_id": "d2", "title": "Two", "rendered_text": "Second"},
    }
    return aspects, by_arm, corpus


def _judgment(a1: tuple[float, str], a2: tuple[float, str]) -> CandidateAspectJudgment:
    return CandidateAspectJudgment(
        candidate="A",
        aspects=[
            AspectScore(
                aspect_id="a1", coverage=a1[0], support=a1[1], evidence_ids=["d1#s1"], explanation=""
            ),
            AspectScore(
                aspect_id="a2", coverage=a2[0], support=a2[1], evidence_ids=["d2#s1"], explanation=""
            ),
        ],
        claim_issues=[],
        invalid_citations=[],
        overall_quality=4,
        explanation="",
    )


def test_complete_outcome_requires_all_critical_aspects() -> None:
    result = compute_candidate_score(_record(), _judgment((1.0, "full"), (0.5, "partial")))
    assert result["outcome"] == "complete"
    assert result["weighted_aspect_coverage"] == 0.8125
    assert result["critical_aspect_success"] == 1.0


def test_missing_critical_aspect_is_partial() -> None:
    result = compute_candidate_score(_record(), _judgment((0.5, "partial"), (1.0, "full")))
    assert result["outcome"] == "partial"
    assert result["weighted_aspect_coverage"] == 0.6875


def test_contradicted_critical_aspect_is_incorrect() -> None:
    result = compute_candidate_score(_record(), _judgment((0.0, "contradicted"), (1.0, "full")))
    assert result["outcome"] == "incorrect"


def test_minor_claim_issue_does_not_make_complete_core_incorrect() -> None:
    judgment = _judgment((1.0, "full"), (1.0, "full"))
    judgment.claim_issues = [
        ClaimIssue(
            issue_type="unsupported",
            severity="minor",
            claim="A peripheral explanation",
            explanation="It does not alter the resolution.",
        )
    ]
    result = compute_candidate_score(_record(), judgment)
    assert result["outcome"] == "complete"
    assert result["material_unsupported_claims"] == []


def test_material_claim_issue_makes_answer_incorrect() -> None:
    judgment = _judgment((1.0, "full"), (1.0, "full"))
    judgment.claim_issues = [
        ClaimIssue(
            issue_type="contradicted",
            severity="material",
            claim="A resolution-changing contradiction",
            explanation="It changes what the user should do.",
        )
    ]
    result = compute_candidate_score(_record(), judgment)
    assert result["outcome"] == "incorrect"
    assert result["material_contradictions"] == [
        "A resolution-changing contradiction"
    ]


def test_agent_failure_is_zero_and_incorrect() -> None:
    result = compute_candidate_score(
        _record(), _judgment((1.0, "full"), (1.0, "full")), agent_ok=False
    )
    assert result["outcome"] == "incorrect"
    assert result["weighted_aspect_coverage"] == 0.0


def test_weighted_aspect_coverage_includes_every_frozen_aspect() -> None:
    record = _record()
    record["aspects"][1]["retrieval_doc_ids"] = []
    record["aspects"][0]["retrieval_doc_ids"] = ["d1"]

    result = compute_candidate_score(
        record, _judgment((1.0, "full"), (0.0, "absent"))
    )

    assert result["weighted_aspect_coverage"] == 0.625
    assert result["document_unsupported_aspect_count"] == 1


def test_question_without_document_supported_critical_aspect_still_has_wac() -> None:
    record = _record()
    record["aspects"][0]["retrieval_doc_ids"] = []
    record["aspects"][1]["retrieval_doc_ids"] = ["d2"]

    result = compute_candidate_score(
        record, _judgment((1.0, "full"), (1.0, "full"))
    )

    assert result["weighted_aspect_coverage"] == 1.0
    assert result["document_supported_critical_aspect_count"] == 0
    assert result["document_unsupported_critical_aspect_count"] == 1


def test_prompt_uses_anonymous_candidates_and_frozen_aspects() -> None:
    rubric, _ = load_judge_rubric()
    payload, aliases = build_judge_prompt(
        _record(),
        {
            "first_system": {"answer": "one", "ranked_ids": ["d1"]},
            "second_system": {"answer": "two", "ranked_ids": ["d2"]},
        },
        {
            "d1": {"doc_id": "d1", "title": "One", "rendered_text": "First"},
            "d2": {"doc_id": "d2", "title": "Two", "rendered_text": "Second"},
        },
        rubric,
    )
    assert set(aliases.values()) == {"A", "B"}
    assert [row["aspect_id"] for row in payload["frozen_aspects"]] == ["a1", "a2"]
    assert {row["candidate"] for row in payload["candidates"]} == {"A", "B"}
    assert "first_system" not in str(payload)


def test_judge_call_cache_is_exactly_keyed(tmp_path) -> None:
    path = tmp_path / "call.json"
    judgment = AspectBatchJudgment(candidates=[_judgment((1.0, "full"), (0.5, "partial"))])
    _write_cached_call(
        path,
        question_id="q1",
        rubric_sha256="rubric",
        model="gpt-5.6-luna",
        reasoning_effort="medium",
        prompt_sha256="prompt",
        aliases={"fs": "A"},
        judgment=judgment,
        meta={"latency_seconds": 1.0, "usage": {"input_tokens": 2, "output_tokens": 3}},
    )

    cached = _load_cached_call(
        path,
        question_id="q1",
        rubric_sha256="rubric",
        model="gpt-5.6-luna",
        reasoning_effort="medium",
        prompt_sha256="prompt",
    )
    assert cached is not None
    assert cached[2] == {"fs": "A"}
    assert _load_cached_call(
        path,
        question_id="q1",
        rubric_sha256="rubric",
        model="gpt-5.6-luna",
        reasoning_effort="medium",
        prompt_sha256="changed",
    ) is None


def test_unique_index_rejects_duplicate_aspects(tmp_path) -> None:
    source = tmp_path / "aspects.jsonl"
    with pytest.raises(ValueError, match="duplicate question_id"):
        _unique_index(
            [{"question_id": "q1"}, {"question_id": "q1"}],
            "question_id",
            source=source,
        )


def test_paired_bootstrap_reports_primary_arm_deltas() -> None:
    rows = []
    values = {
        "q1": {"fs": 0.2, "hybrid": 0.4, "neo4j": 0.5},
        "q2": {"fs": 0.6, "hybrid": 0.7, "neo4j": 0.7},
    }
    for question_id, arms in values.items():
        for arm, value in arms.items():
            rows.append(
                {
                    "question_id": question_id,
                    "arm": arm,
                    "weighted_aspect_coverage": value,
                }
            )
    by_contrast = {
        row["contrast"]: row
        for row in paired_bootstrap_deltas(rows, samples=500, seed=7)
    }
    assert by_contrast["hybrid_minus_fs"]["mean_delta"] == pytest.approx(0.15)
    assert by_contrast["neo4j_minus_hybrid"]["mean_delta"] == pytest.approx(0.05)
    assert by_contrast["neo4j_minus_fs"]["paired_questions"] == 2
    assert by_contrast["neo4j_minus_fs"]["ci95"]["method"] == "paired percentile bootstrap"


def test_usage_totals_distinguishes_persisted_from_incremental() -> None:
    calls = [
        {
            "usage": {
                "input_tokens": 10,
                "cached_input_tokens": 2,
                "output_tokens": 3,
                "total_tokens": 13,
            },
            "latency_seconds": 1.0,
            "resumed": True,
        },
        {
            "usage": {
                "input_tokens": 20,
                "cached_input_tokens": 4,
                "output_tokens": 6,
                "total_tokens": 26,
            },
            "latency_seconds": 2.0,
            "resumed": False,
        },
    ]
    persisted = _usage_totals(calls)
    incremental = _usage_totals([row for row in calls if not row["resumed"]])
    assert persisted["total_tokens"] == 39
    assert incremental["total_tokens"] == 26
    assert persisted["judgments"] == 2
    assert incremental["judgments"] == 1


def test_run_judgments_is_bounded_ordered_and_resumable(tmp_path, capsys) -> None:
    ordered_ids = ["q3", "q1", "q2"]
    aspects, by_arm, corpus = _judgment_inputs(ordered_ids)
    rubric, rubric_sha256 = load_judge_rubric()
    client = _FakeClient(delay_seconds=0.02)

    rows, calls = run_judgments(
        ordered_ids,
        aspects=aspects,
        by_arm=by_arm,
        corpus=corpus,
        rubric=rubric,
        rubric_sha256=rubric_sha256,
        output_dir=tmp_path,
        client=client,
        model="gpt-5.6-luna",
        reasoning_effort="medium",
        resume=False,
        workers=2,
    )

    assert sorted(client.call_ids) == sorted(ordered_ids)
    assert client.max_active == 2
    assert [row["question_id"] for row in calls] == ordered_ids
    assert [(row["question_id"], row["arm"]) for row in rows] == [
        (question_id, arm)
        for question_id in ordered_ids
        for arm in ("fs", "hybrid", "neo4j")
    ]
    assert len(list((tmp_path / "call_cache").glob("*.json"))) == 3
    assert not list((tmp_path / "call_cache").glob("*.tmp"))
    assert "aspect judge progress: 3/3 (resumed=0, new=3)" in capsys.readouterr().out

    no_call_client = _FakeClient(fail_on=set(ordered_ids))
    resumed_rows, resumed_calls = run_judgments(
        ordered_ids,
        aspects=aspects,
        by_arm=by_arm,
        corpus=corpus,
        rubric=rubric,
        rubric_sha256=rubric_sha256,
        output_dir=tmp_path,
        client=no_call_client,
        model="gpt-5.6-luna",
        reasoning_effort="medium",
        resume=True,
        workers=3,
    )
    assert no_call_client.call_ids == []
    assert resumed_rows == rows
    assert [row["question_id"] for row in resumed_calls] == ordered_ids
    assert all(row["resumed"] for row in resumed_calls)


def test_run_judgments_fails_closed_on_client_exception(tmp_path, capsys) -> None:
    ordered_ids = ["q1", "q2", "q3"]
    aspects, by_arm, corpus = _judgment_inputs(ordered_ids)
    rubric, rubric_sha256 = load_judge_rubric()

    with pytest.raises(RuntimeError, match="fake failure for q2"):
        run_judgments(
            ordered_ids,
            aspects=aspects,
            by_arm=by_arm,
            corpus=corpus,
            rubric=rubric,
            rubric_sha256=rubric_sha256,
            output_dir=tmp_path,
            client=_FakeClient(delay_seconds=0.01, fail_on={"q2"}),
            model="gpt-5.6-luna",
            reasoning_effort="medium",
            resume=False,
            workers=2,
        )

    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "per_query.jsonl").exists()
    assert not list((tmp_path / "call_cache").glob("*.tmp"))
    assert "aspect judge failed closed" in capsys.readouterr().out


def test_malformed_judgment_is_not_cached(tmp_path) -> None:
    ordered_ids = ["q1"]
    aspects, by_arm, corpus = _judgment_inputs(ordered_ids)
    rubric, rubric_sha256 = load_judge_rubric()

    with pytest.raises(ValueError, match="candidate labels differ"):
        run_judgments(
            ordered_ids,
            aspects=aspects,
            by_arm=by_arm,
            corpus=corpus,
            rubric=rubric,
            rubric_sha256=rubric_sha256,
            output_dir=tmp_path,
            client=_FakeClient(invalid_on={"q1"}),
            model="gpt-5.6-luna",
            reasoning_effort="medium",
            resume=False,
            workers=1,
        )
    assert not list((tmp_path / "call_cache").glob("*.json"))


def test_evaluate_requires_exact_three_arms_before_any_api_call(tmp_path) -> None:
    aspects = tmp_path / "aspects.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    aspects.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    corpus.write_text(
        json.dumps({"doc_id": "d1", "rendered_text": "one"}) + "\n",
        encoding="utf-8",
    )
    # Use the real rubric path; the failure must happen before rollout loading
    # or OpenAI client construction.
    from dsh_plugin.agent_eval.aspect_judge import DEFAULT_RUBRIC

    args = argparse.Namespace(
        rubric=DEFAULT_RUBRIC,
        aspects=aspects,
        corpus=corpus,
        arm=[("fs", tmp_path / "fs.json")],
        small_trial=True,
        question_id=["q1"],
        limit=None,
    )
    with pytest.raises(ValueError, match="arms must be exactly"):
        evaluate(args)


def test_mixed_generation_mode_runs_fresh_and_records_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rubric, rubric_sha = load_judge_rubric()
    record = _record()
    record.update(
        split="test",
        status="accepted",
        project="github-docs",
        rubric_version=rubric["rubric_version"],
        rubric_sha256=rubric_sha,
    )
    aspects_path = tmp_path / "aspects.jsonl"
    aspects_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        "".join(
            json.dumps({"doc_id": doc_id, "title": doc_id, "rendered_text": doc_id}) + "\n"
            for doc_id in ("d1", "d2")
        ),
        encoding="utf-8",
    )
    split_path = tmp_path / "test.json"
    split_path.write_text(json.dumps([{"question_id": "q1"}]), encoding="utf-8")
    workspace = {
        "protocol": "exact-corpus-official-fs-v1",
        "document_count": 2,
        "corpus_sha256": "c" * 64,
        "workspace_sha256": "w" * 64,
        "ignore_sha256": "i" * 64,
    }
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(
        json.dumps(
            {
                **workspace,
                "official_tools": {
                    "ok": True,
                    "expected_document_count": 2,
                    "searchable_document_count": 2,
                    "glob_document_count": 2,
                    "model_calls": 0,
                    "probes": [{"path": "example.md"}],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(aspect_judge_module, "CANONICAL_TEST_SPLIT", split_path)
    monkeypatch.setattr(aspect_judge_module, "EXPECTED_FULL_QUESTIONS", 1)

    _, memory_rows, _ = _judgment_inputs(["q1"])
    paths: dict[str, Path] = {}
    loaded: dict[Path, list[dict]] = {}
    for arm in ("fs", "hybrid", "neo4j"):
        path = tmp_path / f"{arm}.json"
        path.write_text(f"{arm}\n", encoding="utf-8")
        paths[arm] = path
        row = memory_rows[arm]["q1"]
        row.update(
            id="q1",
            arm=arm,
            question=record["question"],
            target_user_prompt=f"Prompt\nQuestion:\n{record['question']}",
            skill_path=f"/{arm}/initial_skill.md",
            skill_sha256=hashlib.sha256(arm.encode()).hexdigest(),
            evaluation_contract={
                "corpus": {
                    "corpus": {"sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest()},
                    "searchable_workspace": workspace,
                },
                "split": {"file": {"sha256": hashlib.sha256(split_path.read_bytes()).hexdigest()}},
                "runtime": {"sha256": "a" * 64},
                "compiled_plugin": {"sha256": "b" * 64},
                "dependencies": {"locked_files": {"sha256": "c" * 64}},
            },
        )
        loaded[path] = [row]
    monkeypatch.setattr(aspect_judge_module, "load_arm_rollouts", lambda path: loaded[path])
    validation: dict[str, object] = {}

    def fake_validate(*args, **kwargs):
        validation.update(kwargs)
        return {"q1"}

    monkeypatch.setattr(aspect_judge_module, "validate_arm_rollouts", fake_validate)
    output_dir = tmp_path / "output"
    args = argparse.Namespace(
        rubric=aspect_judge_module.DEFAULT_RUBRIC,
        aspects=aspects_path,
        corpus=corpus_path,
        arm=[(arm, paths[arm]) for arm in ("fs", "hybrid", "neo4j")],
        output_dir=output_dir,
        model="gpt-5.6-luna",
        reasoning_effort="medium",
        request_timeout=120.0,
        workers=1,
        limit=None,
        question_id=[],
        small_trial=False,
        resume=False,
        mixed_generation_sensitivity=True,
        fs_preflight=preflight_path,
        expected_rollout_sha256=[
            (arm, hashlib.sha256(paths[arm].read_bytes()).hexdigest())
            for arm in ("hybrid", "neo4j")
        ],
    )
    client = _FakeClient(delay_seconds=0)

    report = evaluate(args, client=client)

    assert client.call_ids == ["q1"]
    assert validation["mixed_generation"] is True
    assert report["comparison_design"] == "mixed_generation_sensitivity"
    assert report["confirmatory"] is False
    assert report["fresh_paired_judge_calls"] is True
    assert report["question_linkage"]["exact_frozen_question"] == 1
    assert report["question_linkage"]["appended_image_evidence"] == 0
    assert report["provenance"]["arms"]["fs"]["rollouts_sha256"] == hashlib.sha256(
        paths["fs"].read_bytes()
    ).hexdigest()
    assert all(not row["resumed"] for row in _jsonl(output_dir / "judge_calls.jsonl"))

    corpus_path.write_text(corpus_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    args.output_dir = tmp_path / "changed-corpus-output"
    with pytest.raises(ValueError, match="corpus bytes differ"):
        evaluate(args, client=client)
    assert client.call_ids == ["q1"]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"resume": True}, "fresh paired judge calls"),
        ({"small_trial": True}, "full 361-question test split"),
        ({"question_id": ["q1"]}, "full 361-question test split"),
        ({"limit": 1}, "full 361-question test split"),
    ],
)
def test_mixed_generation_mode_rejects_nonfinal_options(
    tmp_path: Path, changes: dict, message: str
) -> None:
    aspects = tmp_path / "aspects.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    aspects.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    corpus.write_text(json.dumps({"doc_id": "d1"}) + "\n", encoding="utf-8")
    values = {
        "rubric": aspect_judge_module.DEFAULT_RUBRIC,
        "aspects": aspects,
        "corpus": corpus,
        "arm": [(arm, tmp_path / f"{arm}.json") for arm in ("fs", "hybrid", "neo4j")],
        "output_dir": tmp_path / "output",
        "mixed_generation_sensitivity": True,
        "small_trial": False,
        "question_id": [],
        "limit": None,
        "resume": False,
    }
    values.update(changes)
    args = argparse.Namespace(**values)
    with pytest.raises(ValueError, match=message):
        evaluate(args)


def test_mixed_generation_mode_rejects_nonempty_output(tmp_path: Path) -> None:
    aspects = tmp_path / "aspects.jsonl"
    corpus = tmp_path / "corpus.jsonl"
    output = tmp_path / "output"
    output.mkdir()
    (output / "stale.json").write_text("{}", encoding="utf-8")
    aspects.write_text(json.dumps(_record()) + "\n", encoding="utf-8")
    corpus.write_text(json.dumps({"doc_id": "d1"}) + "\n", encoding="utf-8")
    args = argparse.Namespace(
        rubric=aspect_judge_module.DEFAULT_RUBRIC,
        aspects=aspects,
        corpus=corpus,
        arm=[(arm, tmp_path / f"{arm}.json") for arm in ("fs", "hybrid", "neo4j")],
        output_dir=output,
        mixed_generation_sensitivity=True,
        small_trial=False,
        question_id=[],
        limit=None,
        resume=False,
        model="gpt-5.6-luna",
        reasoning_effort="medium",
    )
    with pytest.raises(ValueError, match="new or empty"):
        evaluate(args)


def test_mixed_helpers_bind_inputs_preflight_and_enriched_question(tmp_path: Path) -> None:
    rollout = tmp_path / "rollouts.json"
    aspects = tmp_path / "aspects.jsonl"
    rollout.write_text("[]", encoding="utf-8")
    aspects.write_text("{}\n", encoding="utf-8")
    captured = _capture_mixed_inputs([("fs", rollout)], {"aspects": aspects})
    _assert_mixed_inputs_unchanged(captured)
    rollout.write_text("[{}]", encoding="utf-8")
    with pytest.raises(ValueError, match="changed during evaluation"):
        _assert_mixed_inputs_unchanged(captured)

    workspace = {
        "protocol": "exact-corpus-official-fs-v1",
        "document_count": 1,
        "corpus_sha256": "c" * 64,
        "workspace_sha256": "w" * 64,
        "ignore_sha256": "i" * 64,
    }
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                **workspace,
                "official_tools": {
                    "ok": True,
                    "expected_document_count": 1,
                    "searchable_document_count": 1,
                    "glob_document_count": 1,
                    "model_calls": 0,
                    "probes": [{"path": "a.md"}],
                },
            }
        ),
        encoding="utf-8",
    )
    _validate_fs_preflight(preflight, {"corpus": {"searchable_workspace": workspace}})
    bad = json.loads(preflight.read_text(encoding="utf-8"))
    bad["official_tools"]["glob_document_count"] = 0
    preflight.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="complete official-tool visibility"):
        _validate_fs_preflight(preflight, {"corpus": {"searchable_workspace": workspace}})

    base = "Question text"
    arms = {
        arm: {
            "q1": {
                "question": base + "\n\nImage-derived question evidence:\n- visible text",
                "target_user_prompt": (
                    "Prompt\nQuestion:\n"
                    + base
                    + "\n\nImage-derived question evidence:\n- visible text"
                ),
                "question_has_image": True,
            }
        }
        for arm in ("fs", "hybrid", "neo4j")
    }
    linkage = _validate_question_linkage({"q1": {"question": base}}, arms, ["q1"])
    assert linkage["appended_image_evidence"] == 1
    arms["fs"]["q1"]["question"] = "Different question"
    with pytest.raises(ValueError, match="question linkage mismatch"):
        _validate_question_linkage({"q1": {"question": base}}, arms, ["q1"])


def test_replay_artifact_report_recomputes_without_model_calls(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "replayed"
    source.mkdir()
    rows = []
    for arm, score in (("fs", 0.25), ("hybrid", 0.75), ("neo4j", 1.0)):
        rows.append(
            {
                "question_id": "q1",
                "arm": arm,
                "project": "github-docs",
                "intent_category": "how_to",
                "evidence_structure": "single",
                "qrel_count": 1,
                "question_has_image": False,
                "weighted_aspect_coverage": score,
                "critical_aspect_success": score,
                "outcome": "complete",
                "material_unsupported_claims": [],
                "material_contradictions": [],
                "citation_integrity": True,
                "overall_quality": 5,
                "agent_latency_seconds": 1.0,
                "agent_total_tokens": 10,
            }
        )
    calls = [
        {
            "resumed": False,
            "usage": {
                "input_tokens": 10,
                "cached_input_tokens": 2,
                "output_tokens": 5,
                "total_tokens": 15,
            },
            "latency_seconds": 1.5,
        }
    ]
    report = {
        "benchmark": "test artifact",
        "questions": 1,
        "arms": summarize(rows),
        "paired_primary_metric_deltas": paired_bootstrap_deltas(rows),
        "by_project": summarize_by(rows, "project"),
        "by_intent_category": summarize_by(rows, "intent_category"),
        "by_evidence_structure": summarize_by(rows, "evidence_structure"),
        "by_qrel_count": summarize_by(rows, "qrel_count"),
        "by_question_image": summarize_by(rows, "question_has_image"),
        "judge_usage": {
            "incremental_api_usage": {**_usage_totals(calls), "api_calls": 1},
            "persisted_artifact_usage": _usage_totals(calls),
            "resumed_judgments": 0,
        },
    }
    (source / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (source / "per_query.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (source / "judge_calls.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in calls), encoding="utf-8"
    )

    assert replay_artifact_report(source, output) == report
    assert json.loads((output / "report.json").read_text()) == report
    with pytest.raises(ValueError, match="output must differ"):
        replay_artifact_report(source, source)
