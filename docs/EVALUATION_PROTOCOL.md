# Evaluation protocol

## Layer 1: plugin/retrieval performance

This no-model evaluation measures the three retrieval implementations before
answer generation:

1. DSH filesystem search with a fixed, bounded exact-term planner;
2. BM25 + HNSW with reciprocal-rank fusion;
3. the same hybrid seeds plus bounded Neo4j structural expansion.

The runner uses the same backend classes as the native DSH service and the same
packaged ripgrep binary as the DSH filesystem tool. It does not ask an LLM to
choose queries, reorder pages, or write an answer.

Hold corpus revision, question IDs, top-k, embedding model, candidate depth,
warm-up, hardware, and thread count fixed. Report Recall@5/10/20, Hit@1/5/10,
nDCG@10, AllSupport@10, p50/p95 warm retrieval latency, index build time, graph
ingest time, and graph/storage statistics. Publish every factual slice.

## Layer 2: integrated DSH-agent performance

Run all arms through the same pinned DSH headless profile. Hold question order,
requested model, actual model, answer schema, timeout, non-KB plugin inventory,
machine, and answer budget fixed. Only the retrieval tool capability and its
matching skill may vary.

The primary ranking is the agent's final ordered `sources` array. The first
documents visible in tool results are retained as diagnostics only. A failed,
malformed, or abstained episode stays in the denominator with zero gain.

Report retrieval metrics plus fresh/cached/output tokens, p50/p95 end-to-end
latency, tool calls, model steps, failure rate, actual model ID, and graph-tool
application rate. If models differ, the result is descriptive and cannot support
a causal plugin comparison.

## Graph construction and expansion

The graph is built before queries from corpus-only structure:

- page and chunk nodes;
- explicit contextual Markdown links;
- route membership;
- shared reusable-content references;
- exact code-shaped identifiers found in documentation.

At query time, the graph remains static but traversal is dynamic: expansion
starts from current hybrid seeds and selects a bounded set of neighbors whose
chunks match the query. No evaluation label tells the agent whether to expand.

## Leakage and reporting rules

- Tune skills or parameters only on train/validation.
- Do not inspect per-question test failures before freezing a final system.
- Use identical paired question IDs for all arms.
- Preserve zeroes for failures instead of dropping them.
- Do not pool plugin latency with end-to-end agent latency.
- Do not infer relevance from missing qrels.
- Report small pilots as engineering checks, not system rankings.
