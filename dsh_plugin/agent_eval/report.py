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


def load_arm_rollouts(run_root: Path) -> list[dict[str, Any]]:
    """Load a direct DSH rollout file or directory."""

    if run_root.is_file():
        data = json.loads(run_root.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"expected rollout array in {run_root}")
        return data
    direct = run_root / "rollouts.json"
    if direct.is_file():
        data = json.loads(direct.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"expected rollout array in {direct}")
        return data
    raise FileNotFoundError(f"no rollouts.json found in {run_root}")


def _questions_by_id(project_root: Path) -> dict[str, dict[str, Any]]:
    path = project_root / "evaluation/dataset/data/questions.jsonl"
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
    final answer citations in ``citation_ranked_ids``. Preserve the former as a
    diagnostic, then normalize all primary metric fields to final citations.
    """

    for row in rows:
        for metric in METRICS:
            row[f"visible_{metric}"] = float(row.get(metric, 0.0))
        question = questions.get(str(row.get("id"))) or {}
        qrels = question.get("qrel_ids") or row.get("qrel_ids") or []
        final_ranked = row.get("citation_ranked_ids") or row.get("ranked_ids") or []
        scored = score_ranked_sources(
            final_ranked, qrels
        )
        for metric in METRICS:
            row[metric] = float(scored.get(metric, 0.0))
            row[f"citation_{metric}"] = float(scored.get(metric, 0.0))
        row["visible_ranked_ids"] = row.get("visible_ranked_ids") or row.get("ranked_ids") or []
        row["ranked_ids"] = list(final_ranked)
        row["citation_ranked_ids"] = list(final_ranked)
        row["evidence_structure"] = question.get(
            "evidence_structure", row.get("evidence_structure", "unknown")
        )
        row["qrel_count"] = int(
            question.get("qrel_count") or row.get("qrel_count") or len(set(qrels))
        )
        row["qrel_count_group"] = "1" if row["qrel_count"] == 1 else "2+"


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
    result["graph_application_rate"] = statistics.fmean(
        1.0 if "docsqa_expand" in (row.get("tool_sequence") or []) else 0.0
        for row in rows
    )
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
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_arm = {arm: load_arm_rollouts(root) for arm, root in arm_roots.items()}
    questions = _questions_by_id(project_root)
    for rows in by_arm.values():
        _attach_primary_metrics(rows, questions)
    id_sets = {arm: {str(row["id"]) for row in rows} for arm, rows in by_arm.items()}
    first_ids = next(iter(id_sets.values()))
    mismatched = {arm: sorted(ids) for arm, ids in id_sets.items() if ids != first_ids}
    if mismatched:
        raise ValueError(f"arms do not contain identical question IDs: {mismatched}")

    per_query: list[dict[str, Any]] = []
    arm_reports: list[dict[str, Any]] = []
    for arm, rows in by_arm.items():
        skill_path = (
            (arm_skills or {}).get(arm)
            or project_root / f"dsh_plugin/plugin/skills/{arm}/initial_skill.md"
        )
        skill_sha = hashlib.sha256(skill_path.read_bytes()).hexdigest()
        arm_reports.append(
            {
                "arm": arm,
                "skill": str(skill_path.relative_to(project_root)),
                "skill_sha256": skill_sha,
                **summarize(rows),
                "by_evidence_category": grouped(rows, "task_type"),
                "by_intent_category": grouped(rows, "intent_category"),
                "by_evidence_structure": grouped(rows, "evidence_structure"),
                "by_qrel_count": grouped(rows, "qrel_count_group"),
            }
        )
        for row in rows:
            per_query.append(
                {
                    "arm": arm,
                    "question_id": str(row["id"]),
                    "evidence_category": row.get("task_type"),
                    "intent_category": row.get("intent_category"),
                    "evidence_structure": row.get("evidence_structure"),
                    "qrel_count": row.get("qrel_count"),
                    "ranked_ids": row.get("ranked_ids") or [],
                    "citation_ranked_ids": row.get("citation_ranked_ids") or [],
                    "visible_ranked_ids": row.get("visible_ranked_ids") or [],
                    "tool_sequence": row.get("tool_sequence") or [],
                    "latency_seconds": row.get("latency_seconds", 0.0),
                    "usage": row.get("usage") or {},
                    "agent_ok": bool(row.get("agent_ok")),
                    **{metric: row.get(metric, 0.0) for metric in METRICS},
                    **{f"visible_{metric}": row.get(f"visible_{metric}", 0.0) for metric in METRICS},
                    **{metric: row.get(metric, 0.0) for metric in CITATION_METRICS},
                }
            )

    model_sets = {arm["arm"]: tuple(arm["models"]) for arm in arm_reports}
    model_mismatch = len(set(model_sets.values())) > 1
    limitations = [
        "Accepted-answer citation qrels are incomplete and may drift relative to the pinned current documentation corpus.",
        "Latency is end-to-end agent latency; provider-only latency was not recorded in these trajectories.",
        "Primary retrieval metrics rank the agent's final ordered sources. First-visible tool documents are retained only as diagnostic fields.",
    ]
    if len(first_ids) < 30:
        limitations.append(
            f"Only {len(first_ids)} paired questions are included; treat this as an engineering pilot, not a final ranking."
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
        "docsqa_expand" in (row.get("tool_sequence") or []) for row in neo4j_rows
    ):
        limitations.append(
            "The Neo4j arm made no docsqa_expand calls, so its result does not test graph traversal."
        )
    if model_mismatch:
        limitations.append(
            "Requested/actual models differ by arm; accuracy and efficiency differences are descriptive system results, not a harness-only causal estimate."
        )
    report = {
        "benchmark": "GitHub Docs DSH agent performance",
        "evaluation_layer": "DSH agent + arm-specific skill + loaded retrieval plugins",
        "sample_kind": "paired DSH agent evaluation",
        "questions": len(first_ids),
        "question_ids": sorted(first_ids),
        "arms": arm_reports,
        "skill_optimization_used": False,
        "models_by_arm": model_sets,
        "model_mismatch": model_mismatch,
        "limitations": limitations,
    }
    return report, sorted(per_query, key=lambda row: (row["question_id"], row["arm"]))


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# GitHub Docs DSH-agent performance report",
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
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    arm_roots = {"fs": args.fs, "hybrid": args.hybrid, "neo4j": args.neo4j}
    report, per_query = build_report(
        arm_roots,
        project_root=project_root,
    )
    write_report(report, per_query, args.out_dir)
    print(render_markdown(report))


if __name__ == "__main__":
    main()
