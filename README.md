# DocsQA-Repo

A benchmark for answering real community questions using a specified product's
GitHub documentation. The research asks how well existing retrieval methods find
and use the required evidence, and whether documentation content and structure
help explain their failures.

This README summarizes the [research analysis and plan](docs/RESEARCH_ANALYSIS.md).
Status: **2026-09-22 (UTC)**.

## Dataset

The release contains **467 questions**, separate reference answers, **4,860
pages**, 601 cited-page judgments and 1,926 frozen answer aspects across GitHub
Docs, Prisma, Supabase and Tailwind CSS. All questions form one pool without
train/validation/test partitions.

Data and source manifests live in [docsqa-data](https://github.com/PowderXu/docsqa-data).
This repository contains the benchmark code and evaluation protocol.
Cited pages provide sparse relevance labels: unjudged pages may still be useful.

## Baseline findings

The current evaluator was rerun on all **467 questions**, with five methods and
two search scopes. Product-only search restricts candidates to the question's
product and is the primary setting; full-corpus search permits all four products
as a supplementary control. Both settings cover GitHub Docs, Prisma, Supabase,
and Tailwind CSS separately. **Recall@10** averages
recovery of the question's cited pages, not answer correctness.

| Method | Product-only Recall@10 | Full-corpus Recall@10 |
|---|---:|---:|
| BM25 | 46.6% | 43.8% |
| Dense | 62.9% | 61.3% |
| Hybrid | 60.5% | 58.4% |
| Hybrid + reranker | 49.7% | 48.9% |
| Hybrid + native links | 62.2% | 60.7% |

All product-only results stayed within scope. Reports separate scope and product
slices. Both modes share global BM25 statistics, while scoped dense search is
exact and unrestricted dense search uses HNSW; their difference cannot be
attributed to filtering alone. The earlier separate-product-index experiment
and its candidate/ranking failure analysis are retained in
[full results](docs/RESULTS.md) under their original settings.

The question pool was used during development, so these results are exploratory.
They do not establish that existing public benchmarks cannot be reused or that
we need a new RAG method.

All 467 questions have now been regenerated with Luna in each of the filesystem,
hybrid and Neo4j-capable agents, restricted to the question's product. The 1,401
episodes contain no observed outside-product access. All 467 paired full-evidence
judgments are complete and independently checked.

| Agent | Answer coverage (WAC) | Tokens / QA |
|---|---:|---:|
| Filesystem | 71.49% | 33,394 |
| Hybrid | 71.07% | 13,192 |
| Neo4j-capable | 72.19% | 13,686 |

All three descriptive paired 95% intervals for WAC differences include zero.
The indexed arms use fewer generation tokens, while this run does not establish
a clear overall answer-coverage advantage. Project-level results differ and are
reported in [full results](docs/RESULTS.md). Neo4j expansion was used on only
2/467 questions, so its arm does not establish a general graph-traversal benefit.

## Research-plan status

| Step | Status |
|---|---|
| 1. Audit labels | Deferred as requested; existing labels used unchanged. |
| 2. Compare standard baselines | Completed: five methods × 467 questions × two search scopes. |
| 3. Analyze failures | Page-level analysis completed; dataset-wide passage analysis remains unfinished. |
| 4. Add a method only if justified | Existing methods retained; a new method is not required for a benchmark paper. |
| 5. Evaluate LLM / agent value | Three product-scoped agents regenerated and fully judged; matched fixed-RAG versus adaptive-agent comparison remains pending. |

The fixed-RAG versus adaptive-agent comparison should use the same questions, Luna model, scoring
rules and explicit evidence/token budgets, measuring answer quality, latency,
tokens and cost. The current three-arm run compares agent tool setups; it does
not include a fixed-evidence control. The small historical fixed-evidence pilot
used different conditions and cannot substitute for that comparison.

For details, see the [dataset design](docs/DATASET_DESIGN.md) and
[evaluation protocol](docs/EVALUATION_PROTOCOL.md). Installation and execution
instructions are in [evaluation](evaluation/README.md) and
[agent setup](dsh_plugin/README.md).
