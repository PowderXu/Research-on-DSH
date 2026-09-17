# DocsQA-Repo

A benchmark for answering real community questions using a specified product's
GitHub documentation. The research asks how well existing retrieval methods find
and use the required evidence, and whether documentation content and structure
help explain their failures.

This README summarizes the [research analysis and plan](docs/RESEARCH_ANALYSIS.md).
Status: **2026-09-16**.

## Dataset

The release contains **467 questions**, separate reference answers, **4,860
pages**, 601 cited-page judgments and 1,926 frozen answer aspects across GitHub
Docs, Prisma, Supabase and Tailwind CSS. All questions form one pool without
train/validation/test partitions.

Data and source manifests live in [docsqa-data](https://github.com/PowderXu/docsqa-data).
This repository contains the benchmark code and evaluation protocol.
Cited pages provide sparse relevance labels: unjudged pages may still be useful.

## Baseline findings

Five existing methods were evaluated on the same 467 questions, each searching
its product's documentation with the same chunking and final top 10.
**Recall@10** averages the fraction of a question's labelled pages recovered;
it measures retrieval, not answer correctness.

| Method | Recall@10 |
|---|---:|
| BM25 | 47.2% |
| Dense | 62.9% |
| Hybrid | 60.7% |
| Hybrid + reranker | 51.6% |
| Hybrid + native document links | 61.0% |

- **Performance varies across the data.** The same hybrid method reaches 39.2%
  recall on Prisma and 94.1% on Tailwind. Content, question mix and corpus size
  also differ; document structure has not been isolated as the cause.
- **Failures occur at both retrieval and ranking.** Of 601 annotated
  question–page pairs, hybrid recovers 329 in its top 10, misses 147 from its
  candidate pool and ranks another 125 below 10.
- **Added components did not reliably improve retrieval.** Native-link expansion
  gives no clear overall recall gain; the tested reranker reduces recall.
  These findings apply to the evaluated configurations.

The question pool was used during development, so these results are exploratory.
They do not establish that existing public benchmarks cannot be reused or that
we need a new RAG method. See [full results and failure analysis](docs/RESULTS.md).

## Research-plan status

| Step | Status |
|---|---|
| 1. Audit labels | Deferred as requested; existing labels used unchanged. |
| 2. Compare standard baselines | Completed: five methods on all 467 questions. |
| 3. Analyze failures | Page-level analysis completed; dataset-wide passage analysis remains unfinished. |
| 4. Add a method only if justified | Existing methods retained; a new method is not required for a benchmark paper. |
| 5. Evaluate LLM / agent value | Matched fixed-RAG versus adaptive-agent comparison remains pending. |

The next agent comparison should use the same questions, Luna model, scoring
rules and explicit evidence/token budgets, measuring answer quality, latency,
tokens and cost. Earlier agent runs and the small fixed-evidence pilot used
different conditions and cannot substitute for that comparison.

For details, see the [dataset design](docs/DATASET_DESIGN.md) and
[evaluation protocol](docs/EVALUATION_PROTOCOL.md). Installation and execution
instructions are in [evaluation](evaluation/README.md) and
[agent setup](dsh_plugin/README.md).
