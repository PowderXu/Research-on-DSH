# Existing results

## Retrieval-only held-out result

The deterministic v2 run evaluated all 246 test questions. It is useful for KB
architecture selection, but it is not a DSH-versus-Codex agent comparison.

| Method | Recall@10 | Hit@10 | nDCG@10 | p50 retrieval |
|---|---:|---:|---:|---:|
| BM25 | 0.381 | 0.415 | 0.243 | 3.80 ms |
| HNSW | 0.470 | 0.524 | 0.306 | 17.33 ms |
| BM25 + HNSW RRF | 0.493 | 0.537 | 0.337 | 17.34 ms |
| Hybrid + conditional explicit-link expansion | 0.503 | 0.545 | 0.349 | 19.42 ms |
| Hybrid + always-on explicit-link expansion | 0.518 | 0.561 | 0.355 | 19.49 ms |
| Pure routing + BM25 | 0.288 | 0.321 | 0.196 | 3.83 ms |

On this dataset, hybrid retrieval is the strong foundation; explicit-link
expansion adds a small benefit at about two milliseconds of median local
retrieval time. Pure routing is substantially worse. The always-on result does
not justify always-on production graph use by itself: the graph is inexpensive
and deterministic here, and the benchmark has sparse citation qrels.

Machine-readable evidence is in
`results/reference/retrieval_v2/report.json` and `per_query.jsonl`. The stored
report was normalized to factual evidence-structure and qrel-count slices.

## Four-arm real-agent pilot

The imported paired pilot has only five single-qrel questions. Its primary
metrics now consistently score each agent's final ordered sources.

| Arm | Hit@10 | nDCG@10 | p50 end-to-end | Tokens / QA |
|---|---:|---:|---:|---:|
| DSH filesystem | 0.600 | 0.377 | 35.01 s | 193,230 |
| DSH hybrid | 0.600 | 0.312 | 29.82 s | 36,793 |
| DSH Neo4j | 0.200 | 0.126 | 18.41 s | 15,367 |
| Codex + FastCtx | 0.200 | 0.200 | 34.95 s | 146,444 |

Do not rank the systems from this table. The sample is tiny, contains no linked
multi-page case, the Neo4j agent made no expansion call, and Codex used
`gpt-5.4-mini` while the DSH arms used `gpt-5-mini`. It demonstrates that the
four paths execute and that final citations can differ sharply from visible
search results.

The normalized report is
`results/reference/agent_pilot5/normalized/report.md`. Older raw reports and
rollouts are retained for provenance; their legacy visible-result fields are
not the benchmark's primary score.

## Optimization evidence

`results/optimization/{fs,hybrid,neo4j}` contains the SkillOpt inputs,
trajectories, candidate skills, and selected artifacts used by the pilot.
`results/diagnostics` preserves failed Codex gates and earlier development
analysis. These are engineering records, not held-out performance claims.
