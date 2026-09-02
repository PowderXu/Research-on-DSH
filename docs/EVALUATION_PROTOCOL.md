# Evaluation protocol

## Current development setting and formal system setting

The intended benchmark is corpus-aware zero-shot evaluation. A submitted system
may index the pinned documentation corpus and use its normal pretrained
components, but it receives no benchmark answers, qrels, aspects, supervised
update, or in-context benchmark examples.

The dataset retains a historical physical split of 33 construction, 73
validation, and 361 legacy-`test` questions. The 361-question partition was
inspected while the retrieval and evaluation code were developed, so results on
it are exploratory. These physical names remain unchanged so historical
retrieval and agent artifacts remain reproducible.

Judge development uses an independent, evaluator-only partition overlay across
all 467 normalized questions:

| Project | Calibration train | Calibration validation | Final test |
|---|---:|---:|---:|
| GitHub Docs | 118 | 39 | 40 |
| Prisma | 75 | 25 | 25 |
| Supabase | 31 | 10 | 11 |
| Tailwind CSS | 56 | 19 | 18 |
| **Total** | **280 (60%)** | **93 (20%)** | **94 (20%)** |

Calibration train is used to revise general judge rules. Calibration validation
selects a candidate and freezes its rubric, prompt, thresholds, and model
settings. The final-test cases are scored once after that freeze, and their
judge outputs and aggregate scores cannot be used for another edit. The overlay
never changes agent inputs, tools, trajectories, or answers; it controls only
evaluator development. It is prospectively held out from this judge-rule
optimization, not historically untouched: all 467 records already passed
dataset construction and normalization. Nevertheless, evaluator overfitting can
bias reported answer quality, so only answer-quality scores produced by a judge
that passes the frozen 94-question final test are confirmatory. A formal
zero-shot *system*
result additionally requires a newly collected sealed temporal or
source-disjoint system cohort after retrieval parameters, skills, graph schema,
and prompts are frozen.

## Benchmark weak supervision

The benchmark assumes that, for most retained questions, the
platform-selected answer together with its resolved internal documentation is
sufficiently complete and correct to resolve the question. These references are
treated as **noisy positives**, not human gold labels. This check is separate
from upstream dataset normalization, which constructs locally reproducible QA
and evidence packages but does not verify the source-answer assumption.

For every question in the evaluator overlay, the protocol pairs the
noisy-positive reference with one independently reviewed useful-but-partial
answer and one independently reviewed decisively incorrect answer. General
rules may be changed only from calibration-train evidence. Calibration
validation is used for model/rubric selection and the decision to freeze. The
frozen judge passes final confirmation only if all of these predeclared
aggregate gates pass on the 94-question final test:

- independent reviewer retention of noisy-positive references >= 0.80;
- judge recall for noisy-positive, partial, and incorrect answers >= 0.80 for
  each class;
- incorrect-to-`complete` rate <= 0.05;
- macro-F1 >= 0.80; and
- valid balanced-triple rate >= 0.98.

The completed live selection followed this contract. Iteration 1 used
`--max-variant-repairs 1`: train passed, while validation failed only balanced
coverage (0.9462 < 0.98). Iteration 2 used
`--max-variant-repairs 2`: train and validation passed with balanced rates
0.9964 and 0.9892. That exact setting, rubric SHA-256
`840d17a4552044b780ed146b5916eb6abad76e06820a07a35b9582e3edadde34`,
`gpt-5.6-luna`, and medium reasoning were frozen before final test. The fresh
94-question final run passed every gate: 1.0000 independent source-review rate,
0.9355 complete recall, 0.9247 partial recall, 0.9892 incorrect recall, 0.0108
incorrect-to-`complete`, 0.9496 macro-F1, and 0.9894 balanced-triple rate. Its
artifact is
`results/runs/dataset-analysis/aspect-calibration-v2-final-test/`.

Passing means that the judge is compatible with the benchmark assumption and
discriminates the controlled silver contrasts. It does not prove source-answer
correctness or human alignment. Independent human verification of the source
assumption is future work.

Every resumable run freezes hashes of the aspects, evaluator-split manifest,
selected question IDs, rubric, and reused cases, together with the partition,
sample controls, model, reasoning effort, and gate thresholds. The three
partitions must be disjoint and their union must exactly cover the 467
normalized IDs. A mismatch must use a new output directory. `--report-only`
validates the existing complete/partial/incorrect triples and judge rows, then
regenerates reports and the run contract without an API call. The exact command
is documented in
[`ASPECT_EVALUATION.md`](ASPECT_EVALUATION.md).

## Retrieval implementation contract

The reported benchmark runs three installed DSH-agent configurations:

1. the filesystem skill lets the agent adaptively choose bounded path matching,
   text search, and document reads;
2. the hybrid skill exposes one `docsqa_search` capability backed by BM25 and
   HNSW, fused with reciprocal-rank fusion;
3. the Neo4j skill exposes the same hybrid search and an optional
   `docsqa_expand` capability. The agent decides whether to call expansion.

The hybrid backend takes at most 50 candidates from each lexical and dense
retriever and returns ten fused documents. Dense retrieval uses
`sentence-transformers/all-MiniLM-L6-v2` on CPU. Graph expansion accepts at
most five hybrid seeds, follows no more than two authored-link hops, admits at
most ten candidates per seed across all edge families, and caps its candidate
pool at 50. These are backend bounds, not a claim that graph expansion runs on
every question; it was successfully used on 4 of 361 reported trajectories.

A direct no-model harness remains for implementation checks. Its fixed planner
and always-applied graph path do not describe the integrated agent experiment
and are not reported as paper results. For the integrated benchmark, hold the
corpus revision, question IDs, model, skills, non-retrieval plugins, hardware,
and execution order fixed. Report final-source Recall@10, Hit@10, nDCG@10,
all-qrels-found@10, end-to-end p50/p95 latency, tokens, tool calls, validity
failures, and graph-application rate.

### Candidate-window rationale

There is no single vendor-neutral production candidate count: a candidate
window is a latency/recall control and must be reported per stage. The
development contract selects its 50/5/10 windows from current first-party
operating examples and defaults:

- Elasticsearch's official RRF example retrieves 50 kNN and 50 standard-search
  results, fuses them in a rank window of 50, and returns the default final size
  of 10. Its documentation explicitly states that a larger rank window can
  improve relevance at a performance cost: [Reciprocal rank fusion](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion).
- Azure AI Search's balanced-hybrid guidance starts with vector `k` in the
  30–50 range and final `top` in the 10–20 range; its semantic ranker consumes
  at most 50 results:
  [Create a hybrid query](https://learn.microsoft.com/en-us/azure/search/hybrid-search-how-to-query).
- Neo4j GraphRAG's retriever API defaults to `top_k=5`; its
  `effective_search_ratio` defaults to 1 and multiplies the vector candidate
  pool, so this benchmark retains ratio 1 rather than silently doubling the
  pool: [Neo4j Hybrid retriever API](https://neo4j.com/docs/neo4j-graphrag-python/current/_modules/neo4j_graphrag/retrievers/hybrid.html).
- Microsoft GraphRAG currently defaults local search to 10 mapped entities and
  10 relationships. We use that documented relationship scale as the per-seed
  traversal cap: [GraphRAG defaults](https://github.com/microsoft/graphrag/blob/main/packages/graphrag/graphrag/config/defaults.py).

These sources motivate an operating range; they do not prove that 50/5/10 is
optimal for this corpus. Sensitivity studies on the reused 361-question
partition are development analyses only. Once a contract is selected, it must
be frozen before the new sealed cohort is acquired or opened.

## Integrated DSH-harness benchmark

Run all arms through the same pinned DSH headless profile. Hold question order,
requested model, actual model, answer schema, timeout, non-KB plugin inventory,
machine, and answer budget fixed. Only the retrieval tool capability and its
matching skill may vary.

The primary ranking is the agent's final ordered `sources` array. The first
documents visible in tool results are retained as diagnostics only. A failed,
malformed, or abstained episode stays in the denominator with zero gain.

Report retrieval metrics plus fresh/cached/output tokens, p50 end-to-end
latency, tool calls, model steps, failure rate, actual model ID, and graph-tool
application rate. Report p95 as an optional diagnostic when the complete run
records it. If models differ, the result is descriptive and cannot support a
causal plugin comparison.

Answer quality is a separate primary agent outcome. The current protocol uses
model-reviewed silver, question-specific aspects built from the normalized
reference and exact local evidence. Score anonymous paired arms with grounded
aspect coverage of 0, 0.5, or 1. Because the agent sees only the pinned documentation, the
primary score is Corpus-Conditioned Grounded Weighted Aspect Coverage:

```text
C-GWAC = sum(document-supported aspect weight × grounded coverage)
         / sum(document-supported aspect weight)
```

A question enters the C-GWAC denominator only when at least one critical aspect
has local-document support. Report all-aspect GWAC separately: it includes
accepted-answer- or question-only information and measures the historical
support-answer gap as well as the agent. Execution failures score zero and are
incorrect. Report both scores with critical aspect success,
complete/partial/incorrect rates, unsupported claims, citation integrity,
latency, and tokens. Do not require wording overlap with the accepted answer:
supported alternatives can be fully correct. Do not combine quality, latency,
and tokens into one opaque utility score. The aspect-construction and
weak-supervision protocol is documented in
[`ASPECT_EVALUATION.md`](ASPECT_EVALUATION.md).

## Graph construction and expansion

The graph is built before queries from corpus-only structure:

- page and chunk nodes;
- contextual Markdown links;
- route membership;
- shared reusable-content references;
- exact code-shaped identifiers found in documentation.
- image-derived local text attached to its source unit; and
- query-blind KGGen entity-predicate-entity claims supported by exact units.

At query time, the graph remains static but traversal is dynamic: expansion
starts from current hybrid seeds, follows at most two Markdown-link hops, and
selects a bounded set of neighbors whose chunks match the query. No evaluation
label selects the queries or paths.

## Leakage and reporting rules

- Call results on the legacy-`test` 361-question partition development or
  exploratory; do not describe them as held out or final.
- Do not use final-test cases, labels, errors, or aggregate scores to revise the
  judge. If the judge changes after final test is opened, create a new final
  cohort rather than reusing those 94 questions as confirmation.
- Do not tune a formal submission on the new sealed cohort's questions, qrels,
  aspects, answer references, or per-question failures.
- Freeze the submitted retrievers, skills, graph schema, prompts, and judge
  before acquiring or opening that sealed cohort.
- Use identical paired question IDs for all arms.
- Preserve zeroes for failures instead of dropping them.
- Do not pool plugin latency with end-to-end agent latency.
- Do not infer relevance from missing qrels.
- Do not attribute accepted-answer-only aspects to a retrieval failure; report
  corpus sufficiency and corpus-conditioned answer quality explicitly.
- Report small pilots as engineering checks, not system rankings.
- Treat the selected 280/93 development results and frozen 94-question final
  result as evaluator validation only. They do not convert retrieval or agent
  measurements on the reused 361-question physical `test` partition into a
  final system result. Independent human verification of the source-answer
  assumption remains future work; a cross-model audit is a sensitivity check,
  not a replacement for that verification.
