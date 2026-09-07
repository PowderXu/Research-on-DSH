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

## Executable checks of the agent evaluation contract

The research question determines what receives credit: the primary metrics ask
whether the **final answer cites the reference pages**, not whether those pages
appeared somewhere in a tool trace. Likewise, a paired mean should give each
question one contribution per arm, including unsuccessful episodes. Otherwise,
reporting choices can change the apparent comparison without improving an agent.

The following synthetic counterexamples make these distinctions testable. Page A
is the only reference page; none of these examples is a benchmark measurement.

| Case | Expected behavior | Why it matters |
|---|---|---|
| A is visible, but the explicit final citation list is empty | Primary Hit@10 = 0; visible Hit@10 = 1 | Exposure to evidence is not final evidence selection. |
| A is cited, but `agent_ok` is false | All primary gains = 0; retain the row, attempted citations, cost, and visible evidence | A partially failed episode must not receive successful-answer credit. |
| The answer is exactly `NOT FOUND`, even with a citation to A | All primary gains = 0; retain the episode | An abstention does not become a supported answer merely by carrying a source. |
| One arm contains q1 twice while all arms have the same set of IDs | Reject the report with the arm and duplicate ID | Set equality alone does not guarantee equal per-question weighting. |

**Compatibility and diagnostics.** An absent `citation_ranked_ids` field retains
the legacy `ranked_ids` fallback; an explicitly empty or null field does not.
That fallback preserves compatibility, not proof that an old ranking represents
final citations. When `visible_ranked_ids` is present, visible metrics are scored
from that list, including an empty list. Otherwise, historical unprefixed metrics
remain the visible diagnostics. Attempted citation IDs are retained even when the
episode receives zero primary gain.

**Failure versus abstention.** The runner and reporter apply the existing
`agent_ok` status and an exact `NOT FOUND` sentinel, ignoring case and surrounding
whitespace. A normal sentence containing those words is not an abstention.
`failure_rate` continues to measure `agent_ok` failures; a successfully executed
abstention receives zero gain without being reclassified as an execution failure.
These checks do not provide full validation of every malformed answer schema.

The self-contained regression suite uses temporary synthetic questions, pages,
skills, and a fake runner. It needs neither a model, a graph service, nor real
benchmark data. From the repository root, in an environment with pytest and
PyYAML installed:

```bash
python -m pytest -c evaluation/pyproject.toml dsh_plugin/agent_eval/test_evaluation_contracts.py
```

These checks validate reporting behavior, not label quality, answer quality, or
the impact on historical results. No benchmark rerun or result-table revision is
implied by them.

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
