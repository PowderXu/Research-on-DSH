# Evaluation protocol

## Scope

DocsQA-Repo evaluates an agent that must answer a real technical support
question from a pinned local Markdown/MDX repository. The benchmark separates:

1. **trajectory retrieval**: which documents the agent actually found and
   read;
2. **final-answer quality**: whether the response covers required facts and
   actions using permitted local evidence; and
3. **efficiency**: model tokens, tool calls, and end-to-end latency.

The paper studies the dataset and benchmark. Filesystem, hybrid, and
Neo4j-capable DSH configurations are baseline systems used to demonstrate it.

## Data roles

The normalized package contains 467 questions. It preserves a historical
physical split used by existing experiments:

| Physical label | Questions | Current role |
|---|---:|---|
| Train | 33 | dataset-construction examples |
| Validation | 73 | historical development |
| Test | 361 | current development/evaluation pool |

The 361-question pool is not held out because it was inspected while the
systems were developed. A confirmatory system result requires a new temporal or
source-disjoint cohort collected after systems and the answer evaluator are
frozen.

Aspect-rule optimization has a separate project-stratified 60/20/20 split over
all 467 records:

| Rule-optimization partition | Questions | Use |
|---|---:|---|
| Train | 280 | produce scored trajectories and rule-edit feedback |
| Validation | 94 | select the best global rule |
| Held-out test | 93 | evaluate the frozen rule once |

This second split controls overfitting of the aspect-construction rule. It does
not train, prompt, or partition evaluated agents. The exact method is in
[Aspect-rule optimization](RULE_OPTIMIZATION.md).

## Zero-shot system protocol

For each question, a submitted system receives:

- the question text, including local image-derived text when present;
- access to the same pinned documentation corpus; and
- the same non-retrieval tools and final-answer schema.

It does not receive the accepted answer, qrels, frozen aspects, aspect-rule
optimization examples, or another benchmark answer. The system may make
multiple search and read calls. All calls, returned identifiers, and the final
response must be recorded in the trajectory.

Agent failures remain in every denominator. A trajectory fails its validity
gate if it does not establish a successful search-to-read evidence path,
returns unresolved sources, violates the final schema, or has an execution
error.

## Trajectory retrieval metrics

The evaluation unit is a canonical documentation page and the cutoff is ten.
Qrels are the distinct in-corpus pages explicitly linked by the accepted source
answer. Because those qrels are sparse, a retrieved page without a qrel has zero
measured gain but is unjudged, not proven irrelevant.

- **Recall@10**: fraction of the question's qrel pages present in the final ten
  trajectory sources.
- **Hit@10**: 1 when at least one qrel is present, otherwise 0.
- **nDCG@10**: discounted gain of qrel pages, normalized by the ideal ordering.
- **AllSupport@10**: 1 when every qrel is present, otherwise 0.

Report macro means over questions and the factual slices in
[`evaluation/kbbench/protocol.json`](../evaluation/kbbench/protocol.json):
project, intent, evidence category, evidence structure, exact qrel count, qrel
count group, and presence of question-image text.

## Final-answer metric

The primary metric is **Corpus-Conditioned Grounded Weighted Aspect Coverage
(C-GWAC)**. Each frozen question-specific aspect has importance (w_i) and an
answer-support value (c_i\in\{1,0.5,0\}) for full, partial, or missing/incorrect
coverage. Only aspects supported by permitted local documentation enter the
corpus-conditioned score:

```text
C-GWAC = sum(w_i * c_i) / sum(w_i)
```

An LLM judge assigns the aspect support labels from the candidate answer and
local evidence. Deterministic code computes the score. The judge may not add
new aspects or alter their weights. Agent failures receive zero.

Secondary answer diagnostics are critical-aspect success, unsupported-claim
rate, and citation-integrity rate. The normalization score documented in
[`NORMALIZED_DATASET.md`](NORMALIZED_DATASET.md) is only a dataset-construction
filter; it is not C-GWAC and is not used to rank agent answers.

## Aspect-rule optimization and freeze

The benchmark assumes that most platform-selected accepted answers are correct
and sufficiently complete when combined with their internally linked local
documentation. `gpt-5.6-luna` applies one shared general rule to construct
per-question aspects and rate source-answer coverage. Deterministic code
validates mappings and calculates weighted coverage. SkillOpt 0.2.0 uses
`gpt-5.6-sol` to propose bounded edits to the general rule from train
trajectories; validation selects the best rule.

A source record passes when its aspect schema is valid and weighted source
coverage is at least 0.80. The frozen rule is eligible for downstream answer
evaluation only when at least 80% of validation records pass. The held-out rule
test is run once after freeze and is not returned to the optimizer.

For final materialization, exact evidence IDs are bound deterministically from
the normalized claims selected by each aspect. This repairs only opaque-ID
transcription; it cannot change aspect semantics, claim mappings, importance,
or criticality. Binding uses no fuzzy path resolution, live URL, or outside
knowledge, and every change is logged beside the retained raw model output.

This procedure creates silver aspects. It is not independent verification that
accepted answers or aspects are correct. A project-stratified expert audit is
required before confirmatory publication claims.

## Baseline configurations

All three DSH systems use `gpt-5.6-luna`, the same corpus and question order,
the same non-retrieval configuration, and the same final-answer contract.

| Arm | Retrieval path |
|---|---|
| Filesystem | DSH skill plus bounded local filesystem search/read tools |
| Hybrid | BM25 + HNSW, fused by reciprocal-rank fusion, then evidence reads |
| Neo4j-capable | identical hybrid seeds plus conditional bounded graph expansion |

The Neo4j arm uses five hybrid seeds, at most two graph hops, entity degree at
most 20, ten graph candidates per seed, and 50 total graph candidates. Graph
use and expansion counts must be logged per question. Index and graph build
time are offline diagnostics, not query latency.

## Efficiency and fairness

Report mean model tokens per QA, mean tool calls per QA, p50 latency, p95
latency, and validity failures. Execute arms sequentially so latency does not
share CPU or service capacity. Exclude offline index construction, graph
ingestion, and answer-judge cost from agent query latency, but report them
separately when available.

A matched comparison fixes corpus hashes, question IDs and order, model and
reasoning effort, non-retrieval tools, top-k, filesystem binary, embedding
model, cache policy, answer schema, and scorer version. Change only the
retrieval plugin/skill configuration. Mixed-generation runs must be labeled as
sensitivity analyses, not causal architecture comparisons.

## Reproduction boundaries

- dataset building: [`evaluation/dataset/scripts/README.md`](../evaluation/dataset/scripts/README.md)
- normalization: [`NORMALIZED_DATASET.md`](NORMALIZED_DATASET.md)
- rule optimization: [`RULE_OPTIMIZATION.md`](RULE_OPTIMIZATION.md)
- retrieval scoring: [`evaluation/kbbench/README.md`](../evaluation/kbbench/README.md)
- DSH baseline composition: [`PLUGIN_DESIGN.md`](PLUGIN_DESIGN.md)
- commands and maintained values: [`RESULTS.md`](RESULTS.md)

Generated corpora, indexes, split manifests, trajectories, aspects, and model
outputs remain under ignored local paths. A paper release must publish
license-compatible derived artifacts and hashes needed to verify each reported
number.
