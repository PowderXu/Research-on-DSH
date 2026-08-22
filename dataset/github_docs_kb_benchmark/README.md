# GitHub Docs knowledge-base retrieval benchmark

This directory is a portable retrieval benchmark for knowledge bases built from
many small, linked Markdown files. It pairs a pinned snapshot of the
[`github/docs`](https://github.com/github/docs) documentation with 328 real
GitHub Community support questions whose accepted answers link to one or more
GitHub Docs pages.

The benchmark is intended to compare retrieval systems under the same agent or
harness, for example:

- filesystem search with a shared retrieval skill;
- BM25 plus dense/HNSW retrieval and rank fusion;
- a Neo4j-backed GraphRAG retriever using the supplied Markdown structure.

It evaluates retrieval only. Answer generation, tool orchestration, and agent
quality should be reported as separate experimental layers.

## Contents

```text
github_docs_kb_benchmark/
├── README.md
├── data/
│   ├── corpus.jsonl
│   ├── questions.jsonl
│   ├── dataset_manifest.json
│   ├── source_manifest.json
│   ├── graph_schema.json
│   └── splits/
│       ├── train.json
│       ├── validation.json
│       ├── test.json
│       └── split_manifest.json
├── benchmark/
│   ├── README.md
│   ├── protocol.json
│   ├── prediction_schema.json
│   └── evaluate_retrieval.py
└── scripts/
    └── rebuild_from_project.py
```

`corpus.jsonl` contains the normalized, rendered corpus and its extracted
Markdown-link metadata, so a retriever can run without a second copy of the
165 MB repository checkout. `source_manifest.json` pins the original repository
and commit for systems that need raw Markdown, frontmatter, includes, or code
examples.

## Dataset construction

The documentation corpus contains 3,740 canonical pages rendered from
`github/docs/content` at commit
`c34e3dccad00f61133c799d20e7d1208a0e6cc92`. Each corpus row includes:

- `doc_id`: canonical page identifier used for scoring;
- `source_path` and `route`: repository and documentation paths;
- `title`, `short_title`, `content_type`, and version metadata;
- `rendered_text`: normalized searchable page content;
- `outgoing_ids` and `link_edges`: explicit Markdown-link targets and edge
  context;
- `reusable_ids`: referenced reusable content.

The 328 questions come from public GitHub Community discussions. A qrel is a
distinct canonical `docs.github.com` page linked in the accepted answer. URLs
were removed from the visible query text to prevent direct URL leakage.

The qrels contain 421 relevant page judgments:

| Relevant pages per question | Questions |
|---:|---:|
| 1 | 263 |
| 2 | 46 |
| 3 | 14 |
| 4 | 2 |
| 5 | 2 |
| 6 | 1 |

These qrels are real and reproducible, but sparse: an unlinked retrieved page is
treated as unjudged/no gain, not proven irrelevant. Therefore, nDCG measures how
well a system ranks the accepted answer's cited pages; it does not establish
that every other page is wrong.

## Question categories

Every question has two human-readable category dimensions already grounded in
the source data:

- `intent_category`: troubleshooting, how-to/configuration,
  policy/billing/account, capability/limit, explanation/comparison, or another
  product question;
- `evidence_category`: direct single page, section-specific single page,
  variant/composed anchor, linked multi-page evidence, or dispersed multi-page
  evidence.

The package also derives two factual fields:

- `qrel_count`: number of distinct relevant canonical pages;
- `evidence_structure`: `single`, `linked`, or `dispersed` based on the qrels and
  evidence category.

The earlier heuristic `graph_opportunity` label is deliberately excluded. Graph
use should be chosen by the tested retrieval policy, not supplied as an oracle
feature by the benchmark.

## Frozen splits

The original 82-question development portion is deterministically partitioned
into 55 training and 27 validation questions. The original 246-question test
portion remains untouched. Use training for skill/prompt or retrieval-policy
optimization, validation for selection, and test exactly once for final
reporting.

Do not train or tune on `test.json`. The split IDs, construction algorithm, and
category counts are recorded in `data/splits/split_manifest.json`.

## Running the benchmark

Create one JSON Lines prediction per question:

```json
{"question_id":"43","ranked_ids":["/graphql/guides/using-the-graphql-api-for-discussions"],"latency_ms":18.4}
```

Then run:

```bash
python dataset/github_docs_kb_benchmark/benchmark/evaluate_retrieval.py \
  --questions dataset/github_docs_kb_benchmark/data/splits/test.json \
  --corpus dataset/github_docs_kb_benchmark/data/corpus.jsonl \
  --predictions path/to/predictions.jsonl \
  --output path/to/retrieval_report.json
```

The evaluator reports Hit and Recall at 1/5/10/20, binary nDCG@10,
AllSupport@10, p50/p95 latency when supplied, and token counts when supplied.
It produces macro averages overall and slices by intent, evidence category,
evidence structure, and qrel count. See [benchmark/README.md](benchmark/README.md)
for the exact contract.

## Fair comparison protocol

For an apples-to-apples KB comparison:

1. Give every arm the same corpus snapshot, questions, split, top-k, model,
   agent harness, non-KB tools, system prompt, and answer budget.
2. Change only the retrieval plugin/backend and its required retrieval-specific
   instructions.
3. Time from retrieval request to the final ranked evidence package, excluding
   one-time indexing. Report index build time and storage separately.
4. Log the ranked canonical page IDs before answer generation.
5. Report the full test set and category slices, not only a simple mean.
6. Keep failed and abstained queries in the denominator.

For agent-level experiments, additionally report end-to-end answer accuracy,
total model tokens, end-to-end latency, tool calls, and failures. Retrieval
metrics alone cannot prove that one agent orchestration design is better.

## Rebuilding

The packaged corpus is already runnable and can be verified from the repository
root with `python scripts/verify_package.py`. To regenerate it from legacy
construction artifacts, first place `questions.jsonl`, `corpus.jsonl`, and
`graph_schema_v2.json` under `data/build_input/github_docs_v2/`, and place the
frozen `train/val/test/items.json` files under
`data/build_input/github_docs_v2_split/`. Then run:

```bash
python dataset/github_docs_kb_benchmark/scripts/rebuild_from_project.py
```

This streams/copies the normalized corpus, removes the deprecated heuristic
field, derives factual labels, rebuilds split metadata, and writes SHA-256 file
hashes. The raw GitHub Docs checkout is intentionally not copied.

## Provenance and licensing

GitHub Docs content is made available under
[CC BY 4.0](https://github.com/github/docs/blob/main/LICENSE). The question and
accepted-answer metadata originate from public GitHub Community discussions and
are user-generated content; they are not automatically covered by the Docs
repository license. Review GitHub's terms before redistributing question or
answer text. For a public artifact, a conservative option is to distribute the
discussion IDs/URLs, construction code, and hashes, then rebuild locally.

This benchmark is not an official GitHub product and its qrels should not be
interpreted as an exhaustive relevance assessment.
