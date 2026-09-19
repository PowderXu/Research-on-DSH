# Research analysis and plan

Status: 2026-09-16. **The plan is partly complete. Step 1 is deferred at the
project owner's request.** Implementation and software checks are not substitutes
for completed research experiments.

## Current task and data

Given a real community question, retrieve evidence from the specified product's
documentation and use it to answer the user. The pinned release has 467 questions,
467 separate answer records, 4,860 documentation pages, 601 cited-page judgments,
and 1,926 frozen answer aspects across GitHub Docs, Prisma, Supabase and Tailwind
CSS. There are no dataset partitions. See [dataset design](DATASET_DESIGN.md).

The page judgments record citations from accepted community answers. Unjudged
pages may still be useful. Page-retrieval scores measure recovery of those
citations; answer-aspect coverage measures a different outcome.

## Recommended research plan

| Step | Current status | Action |
|---|---|---|
| 1. Audit gold labels | **Deferred by project owner** | Use the existing labels unchanged for current experiments. |
| 2. Compare standard retrieval baselines | **Completed: 467 questions, five methods** | BM25, dense, hybrid, hybrid + reranker, and existing native-link expansion; same questions and product corpus. |
| 3. Explain failures | **Page-level analysis completed; passage analysis partial** | Candidate/rank failures and product, intent and citation-count slices are saved. Passage-failure frequency remains unmeasured. |
| 4. Add a method only when justified | **Following this rule** | Keep the existing retrieval methods. A benchmark paper does not require a new RAG algorithm. |
| 5. Evaluate LLM / agent value | Partial; matched comparison pending | Compare fixed evidence and adaptive search on the same cases with Luna, recording answer quality, latency, tokens and cost. |

### Baseline comparison

Freeze the corpus revision, questions, chunker, candidate depths and final top 10
before running. Search only the question's product documentation. Record complete
candidate traces so a missing candidate can be distinguished from a ranking
failure. Reuse existing algorithms without tuning against this question pool.

BM25 versus dense tests lexical versus semantic matching. Hybrid tests their
combination. The reranker tests ordering of retrieved candidates. Native-link
expansion tests whether existing documentation links improve page recovery.
Results and experiment conditions are in [the result index](RESULTS.md).

On the current pool, dense Recall@10 is 0.629, hybrid 0.607, reranked hybrid
0.516 and native-link hybrid 0.610. Link expansion recovered additional cited
pages into its candidate pool but did not produce a clear overall top-10 gain.
The current reranker hurt recall; this is evidence about that model and chunk
policy, not proof that reranking in general is ineffective.

### Failure analysis and interpretation

For each cited page, distinguish: absent from the candidate pool; present but
below rank 10; or recovered. A recovered page can still expose the wrong passage
or produce an incomplete answer. The earlier seven-question fixed-RAG pilot
contains such passage failures, but does not estimate their dataset-wide rate.

Compare single- and multiple-citation questions without assuming that multiple
citations prove necessary multi-file reasoning. Differences between products
also reflect content, corpus size and question mix; they do not isolate a causal
effect of directory depth or graph connectivity.

### Agent comparison still needed

The historical 361-question FS/hybrid/Neo4j run compares three agent tool setups.
The seven-question fixed-evidence pilot compares retrieved snippets with selected
source passages. Neither is a matched fixed-RAG versus adaptive-agent experiment.
Use identical cases, model, answer rubric and explicit evidence/token budgets for
that comparison. Do not combine scores from these different experiments.

## Paper claim supported so far

The project studies reliable answers from product documentation using real
community questions, page retrieval and fine-grained answer requirements.
Whether its data exposes gaps in existing benchmarks is an empirical question.
Current evidence does not establish that public benchmarks cannot be reused,
that topology alone causes failures, or that a new RAG method is necessary.

Detailed exploratory artifacts remain in the ignored local research directory.
The current pool has been used during development, so its results are exploratory.
