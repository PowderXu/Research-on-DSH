# Evaluation protocol

## Two separate layers

The retrieval layer compares ranked canonical page IDs before answer
generation. The agent layer compares the complete DSH or Codex episode. Do not
pool their latency or token numbers.

### Retrieval layer

Hold corpus, split, chunking, embedding model, candidate depth, top-k, warm-up,
thread count, and hardware fixed. Compare:

1. filesystem search;
2. BM25 + HNSW with RRF;
3. the same hybrid seeds plus Neo4j graph expansion.

Report Recall@5/10/20, Hit@1/5/10, nDCG@10, AllSupport@10, p50/p95 retrieval
latency, index construction time, and storage. Report every factual question
slice even when a slice is small.

### Agent layer

Run the same question IDs in paired order with the same requested model,
non-KB plugin inventory, system/user answer schema, timeout, and answer budget.
The filesystem, hybrid, Neo4j, and FastCtx arms may have different
retrieval-specific skills because their tools differ; those skills are frozen
before the test split is opened.

The primary ranking is the final ordered `sources` list. Tool-result order is
logged as `visible_ranked_ids` only to diagnose whether the agent discarded or
reordered good evidence. A failed, malformed, or abstained episode remains in
the denominator with zero retrieval gain.

Report retrieval metrics plus total input/cached/output tokens, p50/p95
end-to-end latency, tool calls, model steps, failure rate, and actual model ID.
If actual models differ, the report is descriptive system evidence and not a
causal harness comparison.

## Graph policy

The graph is built before queries from deterministic repository structure:

- Page and Section nodes;
- route/index hierarchy;
- explicit Markdown links with source section, anchor text, target anchor, and
  local context;
- reusable-content includes and version/tool/platform metadata when available.

Query-time graph expansion is still dynamic: it starts from retrieved seeds and
traverses only a bounded set of relevant structural edges. The frozen graph and
dynamic expansion are different stages.

The benchmark supplies no graph-use oracle. Systems must decide whether to
expand from the query and retrieved evidence. Results are sliced afterward by
factual qrel connectivity (`single`, `linked`, or `dispersed`) and qrel count.

## Frozen data and tuning

- Train (55): skill or retrieval-policy optimization.
- Validation (27): architecture and parameter selection.
- Test (246): final reporting only.

The current imported five-question real-agent run is an engineering pilot, not
a final benchmark. A publishable four-arm comparison still requires a paired
held-out run with the same actual model and enough observations for uncertainty
estimation.
