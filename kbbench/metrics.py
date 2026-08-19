from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def first_relevant_rank(ranked_ids: list[str], relevant_ids: frozenset[str]) -> int | None:
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return rank
    return None


def per_query_metrics(ranked_ids: list[str], relevant_ids: frozenset[str]) -> dict[str, float | int | None]:
    rank = first_relevant_rank(ranked_ids, relevant_ids)
    metrics: dict[str, float] = {}
    for cutoff in (1, 5, 10, 20):
        found = sum(doc_id in relevant_ids for doc_id in ranked_ids[:cutoff])
        metrics[f"recall@{cutoff}"] = found / max(1, len(relevant_ids))
    metrics["mrr@10"] = 1.0 / rank if rank is not None and rank <= 10 else 0.0
    metrics["ndcg@10"] = 1.0 / math.log2(rank + 1) if rank is not None and rank <= 10 else 0.0
    metrics["rank"] = rank
    return metrics


def summarize(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)
    summaries: dict[str, dict[str, float]] = {}
    metric_names = ("recall@1", "recall@5", "recall@10", "recall@20", "mrr@10", "ndcg@10")
    for method, method_rows in by_method.items():
        summary = {
            metric: float(np.mean([float(row[metric]) for row in method_rows]))
            for metric in metric_names
        }
        latencies = np.asarray([float(row["latency_ms"]) for row in method_rows])
        summary.update(
            {
                "queries": float(len(method_rows)),
                "latency_mean_ms": float(latencies.mean()),
                "latency_p50_ms": float(np.percentile(latencies, 50)),
                "latency_p95_ms": float(np.percentile(latencies, 95)),
            }
        )
        summaries[method] = summary
    return summaries


def paired_bootstrap(
    rows: list[dict[str, object]],
    metric: str = "ndcg@10",
    samples: int = 2_000,
    seed: int = 17,
) -> list[dict[str, object]]:
    methods = sorted({str(row["method"]) for row in rows})
    lookup = {(str(row["method"]), str(row["query_id"])): float(row[metric]) for row in rows}
    query_ids = sorted({str(row["query_id"]) for row in rows})
    rng = np.random.default_rng(seed)
    comparisons: list[dict[str, object]] = []
    for left_index, left in enumerate(methods):
        for right in methods[left_index + 1 :]:
            paired_ids = [query_id for query_id in query_ids if (left, query_id) in lookup and (right, query_id) in lookup]
            if not paired_ids:
                continue
            deltas = np.asarray([lookup[(right, query_id)] - lookup[(left, query_id)] for query_id in paired_ids])
            bootstrap = np.empty(samples, dtype=np.float64)
            for sample_index in range(samples):
                draw = rng.integers(0, len(deltas), size=len(deltas))
                bootstrap[sample_index] = deltas[draw].mean()
            comparisons.append(
                {
                    "left": left,
                    "right": right,
                    "metric": metric,
                    "right_minus_left": float(deltas.mean()),
                    "ci95_low": float(np.percentile(bootstrap, 2.5)),
                    "ci95_high": float(np.percentile(bootstrap, 97.5)),
                    "paired_queries": len(paired_ids),
                }
            )
    return comparisons
