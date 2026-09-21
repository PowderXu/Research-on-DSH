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

The normalized package contains one pool of 467 questions, without train,
validation, or test partitions:

- `questions.jsonl`: question text, IDs, question provenance, and question images.
- `answers.jsonl`: reference answers, accepted-answer provenance, qrels, and grading metadata.
- `corpus.jsonl`: the pinned documentation available to the evaluated system.

Records are matched by `question_id`; every question has exactly one answer record.
Evaluators use the entire pool unless an explicit pilot selector is supplied.
Historical results keep their original evaluated cohorts and do not become
467-question results after this storage migration. Existing development exposure
also remains part of their provenance.

Answer evaluation reads frozen annotations from the pinned dataset.
Their format and source are documented in [aspect annotations](ASPECT_ANNOTATIONS.md).

## Zero-shot system protocol

For each question, a submitted system receives:

- the question text, including local image-derived text when present;
- access to the same pinned documentation corpus; and
- the same non-retrieval tools and final-answer schema.

It does not receive the accepted answer, qrels, frozen aspects, or another
benchmark answer. The system may make multiple search and read calls. All
calls, returned identifiers, and the final response must be recorded in the
trajectory.

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

The primary metric is **Weighted Aspect Coverage (WAC)**, following the
[BRIGHT-Pro paper](https://arxiv.org/abs/2605.04018) and its
[official agent-answer evaluator](https://github.com/yale-nlp/Bright-Pro/blob/main/agentic_retrieval/scripts_evaluation/judge.py).
Each frozen question-specific aspect has importance `w_i` and an answer-coverage
value `c_i` in `{1, 0.5, 0}` for full, partial, or missing/incorrect coverage.
Every frozen aspect enters the score:

```text
WAC(q) = sum_{i in A_q}(w_i * c_i) / sum_{i in A_q}(w_i)
```

An LLM judge assigns the aspect support labels from the candidate answer and
local evidence. Deterministic code computes the score. The judge may not add
new aspects or alter their weights. Agent failures receive zero.

Secondary answer diagnostics are critical-aspect success, unsupported-claim
rate, and citation-integrity rate. The normalization score documented in
[`NORMALIZED_DATASET.md`](NORMALIZED_DATASET.md) is only a dataset-construction
filter; it is not WAC and is not used to rank agent answers. Whether each aspect
has pinned local-document support is checked as a dataset-quality property; it
does not change the WAC denominator during agent evaluation.

The judge receives the complete answer and complete corpus text for each of
the first ten distinct candidate source IDs that resolve in the corpus, plus
the frozen gold evidence. The ten-source selection is unchanged; document
text and answers are no longer cut off at 6,000 and 12,000 characters.

Before any judge API calls, every selected question is checked against
`--max-prompt-bytes` (default: 500,000 UTF-8 bytes for the serialized input and
system instructions). This is a local size budget, not a model token limit;
it excludes the response schema and API framing. An oversized input stops the
run without shortening evidence or dropping the question. Supply a positive
integer to change the budget, or `--max-prompt-bytes none` (also accepted: the
flag without a value) to disable the local byte limit. Omitting the flag keeps
the 500,000-byte default; Python callers can pass `max_prompt_bytes=None` for
no local limit. Reports record an unlimited budget as JSON `null` and still
record actual input size. Model context limits and request timeouts still
apply. API truncation remains disabled, so context overflow fails instead of
producing a score from shortened input. Reports record the input policy, and
its version is included in the prompt hash so caches from the previous
truncation policy are not reused.

## Frozen aspect annotations

Answer evaluation reads `aspects.jsonl` from the pinned dataset release for
all 467 questions. Aspect descriptions, weights, and evidence mappings stay
fixed across systems. The LLM assigns coverage labels to each new answer;
deterministic code computes its WAC score. Checks validate aspect schemas,
IDs, weights, and evidence mappings before scoring.

The existing annotations are model-generated silver labels. Their historical
construction is documented in [aspect annotations](ASPECT_ANNOTATIONS.md).
These labels have not been independently verified by domain experts.

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
- fixed aspects and annotation provenance: [`ASPECT_ANNOTATIONS.md`](ASPECT_ANNOTATIONS.md)
- retrieval scoring: [`evaluation/kbbench/README.md`](../evaluation/kbbench/README.md)
- DSH baseline composition: [`PLUGIN_DESIGN.md`](PLUGIN_DESIGN.md)
- commands and maintained values: [`RESULTS.md`](RESULTS.md)

Generated corpora, indexes, trajectories, aspects, and model
outputs remain under ignored local paths. A paper release must publish
license-compatible derived artifacts and hashes needed to verify each reported
number.
