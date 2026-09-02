"""Aggregate matched real-DSH rollouts without re-running retrieval or models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from kbbench.scoring import score_ranked_sources
from dsh_plugin.backend.retrieval_policy import retrieval_contract


METRICS = (
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "recall_at_20",
    "hit_at_1",
    "hit_at_5",
    "hit_at_10",
    "hit_at_20",
    "ndcg_at_10",
    "all_support_at_10",
)
CITATION_METRICS = tuple(f"citation_{metric}" for metric in METRICS)
VISIBLE_METRICS = tuple(f"visible_{metric}" for metric in METRICS)
TOKEN_FIELDS = ("input_fresh", "input_cached", "input_total", "output", "total")
EXPECTED_ARMS = ("fs", "hybrid", "neo4j")
EXPECTED_FULL_QUESTIONS = 361
MATCHED_DEVELOPMENT_MODEL = "gpt-5.6-luna"


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _persisted_tool_outcomes(conversation_path: Path) -> list[dict[str, Any]]:
    """Recover completed tool outcomes from the runner's persisted trajectory."""

    if not conversation_path.is_file():
        raise FileNotFoundError(
            f"persisted conversation is required to validate tool outcomes: {conversation_path}"
        )
    events = json.loads(conversation_path.read_text(encoding="utf-8"))
    if not isinstance(events, list):
        raise ValueError(f"expected conversation array in {conversation_path}")
    calls: dict[str, str] = {}
    outcomes: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        raw_data = event.get("data") or {}
        if isinstance(raw_data, str):
            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                continue
        elif isinstance(raw_data, dict):
            data = raw_data
        else:
            continue
        if event.get("type") == "tool/call":
            call_id = str(data.get("callId") or "")
            name = str(data.get("name") or "")
            if call_id and name:
                calls[call_id] = name
        elif event.get("type") == "tool/result":
            message = data.get("message") or {}
            for block in message.get("content") or []:
                if not isinstance(block, dict) or block.get("type") != "tool-result":
                    continue
                call_id = str(block.get("toolCallId") or "")
                outcomes.append(
                    {
                        "call_id": call_id,
                        "tool": calls.get(call_id, "unknown"),
                        "success": not bool(block.get("isError")),
                    }
                )
    return outcomes


def _embedded_tool_outcomes(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Read the compact tool outcomes persisted in a consolidated rollout."""

    raw_outcomes = row.get("tool_outcomes")
    if not isinstance(raw_outcomes, list):
        raise ValueError(
            f"rollout {row.get('id') or row.get('question_id')} lacks tool_outcomes"
        )
    outcomes: list[dict[str, Any]] = []
    for raw in raw_outcomes:
        if not isinstance(raw, dict):
            raise ValueError("embedded tool outcomes must be objects")
        tool = str(raw.get("tool") or raw.get("name") or "unknown")
        success = raw.get("success")
        if success is None:
            success = raw.get("successful")
        if success is None:
            success = not bool(raw.get("is_error"))
        outcomes.append(
            {
                "call_id": str(raw.get("call_id") or ""),
                "tool": tool,
                "success": bool(success),
            }
        )
    return outcomes


def load_arm_rollouts(run_root: Path) -> list[dict[str, Any]]:
    """Load a direct DSH rollout file or directory."""

    if run_root.is_file():
        data = json.loads(run_root.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"expected rollout array in {run_root}")
        base = run_root.parent
        rows = data
    else:
        direct = run_root / "rollouts.json"
        if not direct.is_file():
            raise FileNotFoundError(f"no rollouts.json found in {run_root}")
        data = json.loads(direct.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"expected rollout array in {direct}")
        base = run_root
        rows = data
    loaded: list[dict[str, Any]] = []
    for source in rows:
        if not isinstance(source, dict):
            raise ValueError(f"rollout rows must be objects in {run_root}")
        row = dict(source)
        question_id = str(row.get("id") or row.get("question_id") or "")
        if not question_id:
            raise ValueError(f"rollout lacks an ID in {run_root}")
        conversation = base / "predictions" / question_id / "conversation.json"
        row["persisted_tool_outcomes"] = (
            _persisted_tool_outcomes(conversation)
            if conversation.is_file()
            else _embedded_tool_outcomes(row)
        )
        loaded.append(row)
    return loaded


def load_expected_question_ids(path: Path) -> list[str]:
    """Load an explicit expected set from a split JSON or aspects JSONL."""

    if path.suffix == ".jsonl":
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError(f"expected an array or JSONL records in {path}")
    ids = [str(row.get("question_id") or row.get("id") or "") for row in values]
    if any(not value for value in ids):
        raise ValueError(f"expected-ID source contains a row without question_id/id: {path}")
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise ValueError(f"expected-ID source contains duplicate IDs: {duplicates}")
    return ids


def _logical_relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"runtime file is outside logical root: {path}") from error


def _files_sha256(paths: Iterable[Path], *, logical_root: Path) -> str:
    """Hash the same logical paths and bytes persisted by the runner."""

    digest = hashlib.sha256()
    resolved = sorted(
        (value.resolve() for value in paths),
        key=lambda path: _logical_relative_path(path, logical_root),
    )
    for path in resolved:
        digest.update(_logical_relative_path(path, logical_root).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _file_identity(path: Path, logical_root: Path) -> dict[str, str]:
    return {
        "path": _logical_relative_path(path, logical_root),
        "sha256": hashlib.sha256(path.resolve().read_bytes()).hexdigest(),
    }


def _repo_file_bundle_identity(
    paths: Iterable[Path], project_root: Path
) -> dict[str, Any]:
    """Mirror ``runner._repo_file_bundle_identity`` exactly."""

    files = sorted(
        (path.resolve() for path in paths),
        key=lambda path: _logical_relative_path(path, project_root),
    )
    return {
        "sha256": _files_sha256(files, logical_root=project_root),
        "files": [_file_identity(path, project_root) for path in files],
    }


def current_runtime_identity(project_root: Path, arm: str) -> dict[str, Any]:
    """Reproduce the runtime identity stored by ``agent_eval.runner``."""

    compiled = sorted((project_root / "dsh_plugin/plugin/lib").rglob("*.js"))
    if not compiled:
        raise FileNotFoundError(
            "compiled DSH plugin runtime is missing; run npm run build in dsh_plugin/plugin"
        )
    paths = [
        project_root / "dsh_plugin/agent_eval/runner.py",
        project_root / "dsh_plugin/backend/http_contract.py",
        project_root / "dsh_plugin/backend/retrieval_policy.py",
        project_root / "dsh_plugin/backend/service.py",
        project_root / "dsh_plugin/plugin/package.json",
        project_root / "evaluation/kbbench/plugin_eval.py",
        project_root / "evaluation/kbbench/retrieval.py",
        project_root / "dsh_plugin/harness/model-openai.patch.yml",
        project_root / "dsh_plugin/harness/docsqa_dsh_common.patch.yml",
        project_root / f"dsh_plugin/harness/docsqa_{arm}_system.patch.yml",
        *compiled,
    ]
    return {
        "runtime": _repo_file_bundle_identity(paths, project_root),
        "compiled_plugin": _repo_file_bundle_identity(compiled, project_root),
    }


def _questions_by_id(project_root: Path) -> dict[str, dict[str, Any]]:
    path = project_root / "evaluation/dataset/evaluation_data/combined/questions.jsonl"
    if not path.is_file():
        # Current rollouts persist qrels and evidence metadata directly. The
        # source package is therefore optional when replaying a frozen artifact.
        return {}
    return {
        str(row["question_id"]): row
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }


def _attach_primary_metrics(
    rows: list[dict[str, Any]], questions: dict[str, dict[str, Any]]
) -> None:
    """Use the agent's final ordered sources as the primary retrieval ranking.

    Older rollout files stored first-visible tool results in ``ranked_ids`` and
    final answer citations in ``citation_ranked_ids``. Preserve and score the
    former as a diagnostic, then normalize all primary metric fields to final
    citations. Current rollouts persist ``visible_ranked_ids`` explicitly; an
    explicit empty list means no visible evidence and must not use a fallback.
    """

    for row in rows:
        question = questions.get(str(row.get("id"))) or {}
        # Current rollouts are self-contained and their frozen normalized qrels
        # take precedence. The source package is only a compatibility fallback
        # for legacy rollouts that did not persist qrels.
        qrels = row.get("qrel_ids") or question.get("qrel_ids") or []
        legacy_ranked = row.get("ranked_ids") or []
        final_ranked = row.get("citation_ranked_ids") or legacy_ranked
        if "visible_ranked_ids" in row and row.get("visible_ranked_ids") is not None:
            visible_ranked = row.get("visible_ranked_ids") or []
        else:
            visible_ranked = legacy_ranked
        scored = score_ranked_sources(final_ranked, qrels)
        visible_scored = score_ranked_sources(visible_ranked, qrels)
        for metric in METRICS:
            row[metric] = float(scored.get(metric, 0.0))
            row[f"citation_{metric}"] = float(scored.get(metric, 0.0))
            row[f"visible_{metric}"] = float(visible_scored.get(metric, 0.0))
        row["visible_ranked_ids"] = list(visible_ranked)
        row["ranked_ids"] = list(final_ranked)
        row["citation_ranked_ids"] = list(final_ranked)
        row["evidence_structure"] = row.get("evidence_structure") or question.get(
            "evidence_structure", "unknown"
        )
        row["qrel_count"] = int(
            row.get("qrel_count") or question.get("qrel_count") or len(set(qrels))
        )
        row["qrel_count_group"] = "1" if row["qrel_count"] == 1 else "2+"


def validate_arm_rollouts(
    by_arm: dict[str, list[dict[str, Any]]],
    *,
    project_root: Path,
    expected_question_ids: Iterable[str],
    arm_skills: dict[str, Path] | None = None,
    small_trial: bool = False,
    artifact_replay: bool = False,
) -> set[str]:
    """Validate either a current run or an explicitly frozen artifact replay.

    Current-run validation binds every rollout to the checked-out skill and
    compiled runtime. Artifact replay instead verifies the identities recorded
    inside the rollouts and never represents them as current-source results.
    """

    actual_arms = set(by_arm)
    if actual_arms != set(EXPECTED_ARMS):
        raise ValueError(
            f"arms must be exactly {list(EXPECTED_ARMS)}; actual={sorted(actual_arms)}"
        )
    expected_list = [str(value) for value in expected_question_ids]
    duplicate_expected = sorted(
        {value for value in expected_list if expected_list.count(value) > 1}
    )
    if duplicate_expected:
        raise ValueError(f"expected question IDs contain duplicates: {duplicate_expected}")
    expected = set(expected_list)
    if not expected:
        raise ValueError("expected question IDs cannot be empty")
    if not small_trial and len(expected) != EXPECTED_FULL_QUESTIONS:
        raise ValueError(
            f"complete matched development run requires exactly {EXPECTED_FULL_QUESTIONS} expected IDs; "
            f"got {len(expected)} (use explicit small_trial=True only for a pilot)"
        )

    shared_contract_fields: dict[str, tuple[str, str]] = {}
    for arm in EXPECTED_ARMS:
        rows = by_arm[arm]
        ids = [str(row.get("id") or row.get("question_id") or "") for row in rows]
        duplicate_ids = sorted({value for value in ids if ids.count(value) > 1})
        if duplicate_ids:
            raise ValueError(f"{arm} contains duplicate rollout IDs: {duplicate_ids}")
        actual_ids = set(ids)
        if actual_ids != expected:
            missing = sorted(expected - actual_ids)
            unexpected = sorted(actual_ids - expected)
            raise ValueError(
                f"{arm} rollout IDs do not equal the explicit expected set; "
                f"missing={missing} unexpected={unexpected}"
            )
        skill_path = (
            (arm_skills or {}).get(arm)
            or project_root / f"dsh_plugin/plugin/skills/{arm}/initial_skill.md"
        )
        current_skill_sha = (
            None
            if artifact_replay
            else hashlib.sha256(skill_path.read_bytes()).hexdigest()
        )
        current_runtime = (
            None if artifact_replay else current_runtime_identity(project_root, arm)
        )
        contracts: list[dict[str, Any]] = []
        recorded_skill_shas: set[str] = set()
        recorded_skill_paths: set[str] = set()
        for row in rows:
            if str(row.get("arm") or "") != arm:
                raise ValueError(
                    f"rollout {row.get('id')} is stored under {arm} but declares arm={row.get('arm')!r}"
                )
            recorded_skill_sha = str(row.get("skill_sha256") or "")
            if len(recorded_skill_sha) != 64:
                raise ValueError(
                    f"rollout lacks a recorded skill SHA-256: {arm}/{row.get('id')}"
                )
            recorded_skill_shas.add(recorded_skill_sha)
            recorded_skill_paths.add(str(row.get("skill_path") or ""))
            if not artifact_replay and recorded_skill_sha != current_skill_sha:
                raise ValueError(
                    f"stored/current skill mismatch for {arm} rollout {row.get('id')}"
                )
            models = [str(value) for value in row.get("actual_models") or []]
            if models != [MATCHED_DEVELOPMENT_MODEL]:
                raise ValueError(
                    f"complete matched development rollouts require actual_models=['{MATCHED_DEVELOPMENT_MODEL}']; "
                    f"{arm}/{row.get('id')} has {models}"
                )
            contract = row.get("evaluation_contract")
            if not isinstance(contract, dict):
                raise ValueError(
                    f"rollout lacks a persisted evaluation contract: {arm}/{row.get('id')}"
                )
            contracts.append(contract)
            if contract.get("retrieval") != retrieval_contract(arm):
                raise ValueError(
                    f"stored/current retrieval contract mismatch for {arm}/{row.get('id')}"
                )
            if artifact_replay:
                for field in ("runtime", "compiled_plugin"):
                    identity = contract.get(field)
                    if not isinstance(identity, dict):
                        raise ValueError(
                            f"artifact contract lacks {field}: {arm}/{row.get('id')}"
                        )
                    digest = str(identity.get("sha256") or "")
                    files = identity.get("files")
                    if len(digest) != 64 or not isinstance(files, list) or not files:
                        raise ValueError(
                            f"artifact contract has invalid {field} identity: "
                            f"{arm}/{row.get('id')}"
                        )
            else:
                assert current_runtime is not None
                for field, current_value in current_runtime.items():
                    if contract.get(field) != current_value:
                        raise ValueError(
                            f"stored/current {field} mismatch for {arm}/{row.get('id')}"
                        )
            for field in (
                "embedding_model",
                "device",
                "corpus",
                "split",
                "scoring",
                "dependencies",
                "provider",
            ):
                raw_value = contract.get(field)
                if raw_value is None or raw_value == "":
                    raise ValueError(
                        f"evaluation contract lacks {field}: {arm}/{row.get('id')}"
                    )
                value = (
                    str(raw_value or "")
                    if field in {"embedding_model", "device"}
                    else json.dumps(raw_value, sort_keys=True, separators=(",", ":"))
                )
                previous = shared_contract_fields.setdefault(field, (arm, value))
                if previous[1] != value:
                    raise ValueError(
                        f"evaluation contract {field} differs by arm: "
                        f"{previous[0]}={previous[1]!r}, {arm}={value!r}"
                    )
        if len(recorded_skill_shas) != 1:
            raise ValueError(f"recorded skill identity varies within the {arm} arm")
        if len(recorded_skill_paths) != 1:
            raise ValueError(f"recorded skill path varies within the {arm} arm")
        canonical_contract = json.dumps(contracts[0], sort_keys=True)
        if any(json.dumps(value, sort_keys=True) != canonical_contract for value in contracts[1:]):
            raise ValueError(f"evaluation contract varies within the {arm} arm")
    return expected


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty rollout set")
    result: dict[str, Any] = {"questions": len(rows)}
    for metric in METRICS:
        result[metric] = statistics.fmean(float(row.get(metric, 0.0)) for row in rows)
    for metric in CITATION_METRICS:
        result[metric] = statistics.fmean(float(row.get(metric, 0.0)) for row in rows)
    for metric in VISIBLE_METRICS:
        result[metric] = statistics.fmean(float(row.get(metric, 0.0)) for row in rows)
    latencies = [float(row.get("latency_seconds", 0.0)) for row in rows]
    result["latency_p50_ms"] = percentile(latencies, 0.50) * 1000.0
    result["latency_p95_ms"] = percentile(latencies, 0.95) * 1000.0
    result["tokens_per_qa"] = {
        field: statistics.fmean(
            float((row.get("usage") or {}).get(field, 0.0)) for row in rows
        )
        for field in TOKEN_FIELDS
    }
    result["tool_calls_per_qa"] = statistics.fmean(
        len(row.get("tool_sequence") or []) for row in rows
    )
    result["failure_rate"] = statistics.fmean(
        0.0 if row.get("agent_ok") else 1.0 for row in rows
    )
    result["graph_expansion_attempt_rate"] = statistics.fmean(
        1.0 if "docsqa_expand" in (row.get("tool_sequence") or []) else 0.0
        for row in rows
    )
    successful_expansions = [
        sum(
            outcome.get("tool") == "docsqa_expand" and bool(outcome.get("success"))
            for outcome in row.get("persisted_tool_outcomes") or []
        )
        for row in rows
    ]
    result["graph_application_rate"] = statistics.fmean(
        1.0 if count else 0.0 for count in successful_expansions
    )
    result["graph_expansion_successes_per_qa"] = statistics.fmean(successful_expansions)
    result["models"] = sorted(
        {
            str(model)
            for row in rows
            for model in (row.get("actual_models") or [])
            if str(model)
        }
    )
    return result


def grouped(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if isinstance(value, bool):
            key = "true" if value else "false"
        else:
            key = str(value or "unknown")
        groups[key].append(row)
    return [{field: key, **summarize(groups[key])} for key in sorted(groups)]


def build_report(
    arm_roots: dict[str, Path],
    *,
    project_root: Path,
    arm_skills: dict[str, Path] | None = None,
    expected_question_ids: Iterable[str],
    small_trial: bool = False,
    artifact_replay: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_arm = {arm: load_arm_rollouts(root) for arm, root in arm_roots.items()}
    expected_ids = validate_arm_rollouts(
        by_arm,
        project_root=project_root,
        expected_question_ids=expected_question_ids,
        arm_skills=arm_skills,
        small_trial=small_trial,
        artifact_replay=artifact_replay,
    )
    questions = _questions_by_id(project_root)
    for rows in by_arm.values():
        _attach_primary_metrics(rows, questions)
    per_query: list[dict[str, Any]] = []
    arm_reports: list[dict[str, Any]] = []
    for arm, rows in by_arm.items():
        skill_path = (
            (arm_skills or {}).get(arm)
            or project_root / f"dsh_plugin/plugin/skills/{arm}/initial_skill.md"
        )
        if artifact_replay:
            recorded_path = str(rows[0].get("skill_path") or skill_path)
            normalized_path = recorded_path.replace("\\", "/")
            skill_marker = "dsh_plugin/plugin/skills/"
            if skill_marker in normalized_path:
                skill_label = skill_marker + normalized_path.split(skill_marker, 1)[1]
            else:
                try:
                    skill_label = str(
                        Path(recorded_path).resolve().relative_to(project_root)
                    )
                except ValueError:
                    skill_label = Path(recorded_path).name
            skill_sha = str(rows[0]["skill_sha256"])
        else:
            skill_label = str(skill_path.relative_to(project_root))
            skill_sha = hashlib.sha256(skill_path.read_bytes()).hexdigest()
        arm_reports.append(
            {
                "arm": arm,
                "skill": skill_label,
                "skill_sha256": skill_sha,
                "evaluation_contract": rows[0]["evaluation_contract"],
                **summarize(rows),
                "by_evidence_category": grouped(rows, "task_type"),
                "by_project": grouped(rows, "project"),
                "by_intent_category": grouped(rows, "intent_category"),
                "by_evidence_structure": grouped(rows, "evidence_structure"),
                "by_qrel_count": grouped(rows, "qrel_count_group"),
                "by_question_image": grouped(rows, "question_has_image"),
            }
        )
        for row in rows:
            per_query.append(
                {
                    "arm": arm,
                    "question_id": str(row["id"]),
                    "project": row.get("project"),
                    "evidence_category": row.get("task_type"),
                    "intent_category": row.get("intent_category"),
                    "evidence_structure": row.get("evidence_structure"),
                    "qrel_count": row.get("qrel_count"),
                    "question_has_image": bool(row.get("question_has_image")),
                    "ranked_ids": row.get("ranked_ids") or [],
                    "citation_ranked_ids": row.get("citation_ranked_ids") or [],
                    "visible_ranked_ids": row.get("visible_ranked_ids") or [],
                    "tool_sequence": row.get("tool_sequence") or [],
                    "successful_tool_sequence": [
                        outcome.get("tool")
                        for outcome in row.get("persisted_tool_outcomes") or []
                        if outcome.get("success")
                    ],
                    "latency_seconds": row.get("latency_seconds", 0.0),
                    "usage": row.get("usage") or {},
                    "agent_ok": bool(row.get("agent_ok")),
                    **{metric: row.get(metric, 0.0) for metric in METRICS},
                    **{f"visible_{metric}": row.get(f"visible_{metric}", 0.0) for metric in METRICS},
                    **{metric: row.get(metric, 0.0) for metric in CITATION_METRICS},
                }
            )

    model_sets = {arm["arm"]: tuple(arm["models"]) for arm in arm_reports}
    limitations = [
        "Accepted-answer citation qrels are incomplete and may drift relative to the pinned current documentation corpus.",
        "Latency is end-to-end agent latency; provider-only latency was not recorded in these trajectories.",
        "Primary retrieval metrics rank the agent's final ordered sources. First-visible tool documents are retained only as diagnostic fields.",
    ]
    if small_trial:
        limitations.append(
            f"Only {len(expected_ids)} explicitly selected paired questions are included; treat this as an engineering pilot, not a complete matched development run."
        )
    if artifact_replay:
        limitations.append(
            "This report is an artifact replay: persisted skill/runtime identities were validated, but they were not required to match the current checkout."
        )
    evidence_structures = {
        str(row.get("evidence_structure") or "unknown")
        for rows in by_arm.values()
        for row in rows
    }
    if "linked" not in evidence_structures:
        limitations.append(
            "The sample contains no linked multi-page question, so it cannot measure graph-traversal advantage."
        )
    neo4j_rows = by_arm.get("neo4j") or []
    if neo4j_rows and not any(
        any(
            outcome.get("tool") == "docsqa_expand" and outcome.get("success")
            for outcome in row.get("persisted_tool_outcomes") or []
        )
        for row in neo4j_rows
    ):
        limitations.append(
            "The Neo4j arm has no successful persisted docsqa_expand outcome, so its result does not test graph traversal."
        )
    report = {
        "benchmark": "DocsQA DSH agent performance",
        "evaluation_layer": "DSH agent + arm-specific skill + loaded retrieval plugins",
        "sample_kind": "paired DSH agent evaluation",
        "questions": len(expected_ids),
        "question_ids": sorted(expected_ids),
        "arms": arm_reports,
        "skill_optimization_used": False,
        "models_by_arm": model_sets,
        "model_mismatch": False,
        "complete_matched_development_run": not small_trial,
        "artifact_replay": artifact_replay,
        "limitations": limitations,
    }
    return report, sorted(per_query, key=lambda row: (row["question_id"], row["arm"]))


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# DocsQA DSH-agent performance report",
        "",
        f"This report contains **{report['questions']} identical real-model questions per DSH arm**.",
        "",
        "## Overall",
        "",
        "| Arm | Recall@10 | Hit@10 | nDCG@10 | Visible Hit@10 (diagnostic) | Visible nDCG@10 (diagnostic) | p50 latency | Tokens / QA | Tool calls / QA | Failures | Graph use |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in report["arms"]:
        lines.append(
            f"| {arm['arm']} | {arm['recall_at_10']:.3f} | {arm['hit_at_10']:.3f} | "
            f"{arm['ndcg_at_10']:.3f} | {arm['visible_hit_at_10']:.3f} | "
            f"{arm['visible_ndcg_at_10']:.3f} | {arm['latency_p50_ms'] / 1000:.2f} s | "
            f"{arm['tokens_per_qa']['total']:,.0f} | {arm['tool_calls_per_qa']:.1f} | "
            f"{arm['failure_rate']:.3f} | {arm['graph_application_rate']:.3f} |"
        )
    for group_field, title in (
        ("by_evidence_category", "By evidence category"),
        ("by_intent_category", "By intent category"),
        ("by_evidence_structure", "By evidence structure"),
        ("by_qrel_count", "By qrel count"),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Arm | Category | N | Recall@10 | Hit@10 | nDCG@10 | p50 latency | Tokens / QA |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for arm in report["arms"]:
            for group in arm[group_field]:
                category_key = {
                    "by_evidence_category": "task_type",
                    "by_intent_category": "intent_category",
                    "by_evidence_structure": "evidence_structure",
                    "by_qrel_count": "qrel_count_group",
                }[group_field]
                lines.append(
                    f"| {arm['arm']} | {group[category_key]} | {group['questions']} | "
                    f"{group['recall_at_10']:.3f} | {group['hit_at_10']:.3f} | "
                    f"{group['ndcg_at_10']:.3f} | {group['latency_p50_ms'] / 1000:.2f} s | "
                    f"{group['tokens_per_qa']['total']:,.0f} |"
                )
    lines.extend(["", "## Interpretation boundary", ""])
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def write_report(
    report: dict[str, Any], per_query: list[dict[str, Any]], out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "per_query.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in per_query),
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(render_markdown(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fs", type=Path, required=True)
    parser.add_argument("--hybrid", type=Path, required=True)
    parser.add_argument("--neo4j", type=Path, required=True)
    parser.add_argument(
        "--expected-ids",
        type=Path,
        required=True,
        help="Frozen split JSON or aspects JSONL defining the exact expected question IDs.",
    )
    parser.add_argument(
        "--small-trial",
        action="store_true",
        help="Explicitly label a non-361 expected-ID subset as an engineering pilot rather than a complete matched development run.",
    )
    parser.add_argument(
        "--artifact-replay",
        action="store_true",
        help=(
            "Recompute a historical report from self-contained rollouts while "
            "validating their persisted identities instead of the current checkout."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    arm_roots = {"fs": args.fs, "hybrid": args.hybrid, "neo4j": args.neo4j}
    report, per_query = build_report(
        arm_roots,
        project_root=project_root,
        expected_question_ids=load_expected_question_ids(args.expected_ids),
        small_trial=args.small_trial,
        artifact_replay=args.artifact_replay,
    )
    write_report(report, per_query, args.out_dir)
    print(render_markdown(report))


if __name__ == "__main__":
    main()
