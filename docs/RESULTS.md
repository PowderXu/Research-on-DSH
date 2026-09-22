# Current results

> The current dataset is one 467-question pool with no dataset partitions.
> Current retrieval and agent comparisons use that full pool. Historical
> tables retain their original sample sizes and settings.

Status date: **2026-09-22 (UTC)**.

This file is the maintained result index. Generated datasets, model outputs,
trajectories, indexes and detailed reports remain in ignored local directories.
Current results are exploratory because the question pool was used in development.
See [the research plan](RESEARCH_ANALYSIS.md) for completed and pending steps.

## Fresh project-scoped agents and full-evidence judging (2026-09-22)

All **467 questions × three Luna agents = 1,401 fresh episodes** were completed. Each
question searches only its own product. The four project groups are GitHub Docs (197),
Prisma (125), Supabase (52), and Tailwind CSS (93). All episodes remain in the
denominator, including 6 filesystem, 5 hybrid and 5 Neo4j validity failures. No
outside-product document access was observed in the recorded traces.

**Answer scores are new:** the judge evaluated all three anonymous answers together for
every question, using full answers, full selected document text and the unchanged 1,926
frozen aspects. WAC is weighted answer coverage; invalid trajectories receive zero WAC.
Recall/nDCG below score the final ordered citations, at most ten distinct pages. They do
not score every page returned by the agent tools and are not directly interchangeable
with the retrieval-only top-ten metrics.

### Overall

| Arm | Recall@10 | nDCG@10 | WAC | Critical-aspect success | Material claim issues | Citation integrity | Invalid trajectories |
|---|---:|---:|---:|---:|---:|---:|---:|
| Filesystem | 0.5458 | 0.4944 | 0.7149 | 0.5883 | 0.2377 | 1.0000 | 6/467 |
| Hybrid | 0.4725 | 0.4277 | 0.7107 | 0.5792 | 0.2505 | 0.9979 | 5/467 |
| Neo4j-capable | 0.5086 | 0.4411 | 0.7219 | 0.5930 | 0.2463 | 1.0000 | 5/467 |

### Answer quality by project

| Project | Arm | Questions | WAC | Critical-aspect success | Material claim issues | Citation integrity |
|---|---|---:|---:|---:|---:|---:|
| GitHub Docs | Filesystem | 197 | 0.6779 | 0.5069 | 0.2893 | 1.0000 |
| GitHub Docs | Hybrid | 197 | 0.6733 | 0.5094 | 0.2741 | 1.0000 |
| GitHub Docs | Neo4j-capable | 197 | 0.7046 | 0.5391 | 0.2792 | 1.0000 |
| Prisma | Filesystem | 125 | 0.7566 | 0.6575 | 0.1840 | 1.0000 |
| Prisma | Hybrid | 125 | 0.7870 | 0.6805 | 0.1920 | 0.9920 |
| Prisma | Neo4j-capable | 125 | 0.7580 | 0.6915 | 0.1920 | 1.0000 |
| Supabase | Filesystem | 52 | 0.6785 | 0.5785 | 0.2692 | 1.0000 |
| Supabase | Hybrid | 52 | 0.6501 | 0.5337 | 0.3269 | 1.0000 |
| Supabase | Neo4j-capable | 52 | 0.6841 | 0.5673 | 0.2692 | 1.0000 |
| Tailwind CSS | Filesystem | 93 | 0.7575 | 0.6733 | 0.1828 | 1.0000 |
| Tailwind CSS | Hybrid | 93 | 0.7214 | 0.6163 | 0.2366 | 1.0000 |
| Tailwind CSS | Neo4j-capable | 93 | 0.7311 | 0.5891 | 0.2366 | 1.0000 |

### Final-source recovery and efficiency by project

| Project | Arm | Recall@10 | Hit@10 | nDCG@10 | AllSupport@10 | p50 / p95 latency | Tokens / QA | Tool calls / QA | Invalid trajectories | Graph used |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GitHub Docs | Filesystem | 0.5256 | 0.5838 | 0.4760 | 0.4721 | 9.03 / 14.81 s | 32,814 | 7.52 | 1/197 | 0/197 |
| GitHub Docs | Hybrid | 0.5060 | 0.5635 | 0.4593 | 0.4619 | 9.36 / 14.20 s | 12,352 | 5.63 | 3/197 | 0/197 |
| GitHub Docs | Neo4j-capable | 0.5729 | 0.6193 | 0.4954 | 0.5330 | 8.38 / 12.58 s | 11,766 | 4.64 | 0/197 | 0/197 |
| Prisma | Filesystem | 0.2467 | 0.3280 | 0.2327 | 0.1760 | 9.60 / 17.23 s | 39,901 | 8.35 | 4/125 | 0/125 |
| Prisma | Hybrid | 0.1440 | 0.1920 | 0.1388 | 0.1040 | 10.71 / 15.72 s | 14,303 | 5.67 | 1/125 | 0/125 |
| Prisma | Neo4j-capable | 0.1533 | 0.2080 | 0.1418 | 0.1040 | 10.01 / 14.54 s | 15,109 | 5.18 | 4/125 | 2/125 |
| Supabase | Filesystem | 0.7596 | 0.8077 | 0.6751 | 0.7115 | 9.44 / 19.44 s | 31,385 | 9.06 | 1/52 | 0/52 |
| Supabase | Hybrid | 0.5865 | 0.6346 | 0.5454 | 0.5385 | 9.13 / 11.87 s | 11,419 | 5.42 | 1/52 | 0/52 |
| Supabase | Neo4j-capable | 0.6250 | 0.6731 | 0.5286 | 0.5769 | 10.03 / 15.40 s | 12,105 | 4.79 | 1/52 | 0/52 |
| Tailwind CSS | Filesystem | 0.8710 | 0.8925 | 0.7842 | 0.8495 | 10.34 / 14.96 s | 26,997 | 9.01 | 0/93 | 0/93 |
| Tailwind CSS | Hybrid | 0.7796 | 0.7849 | 0.6833 | 0.7742 | 11.74 / 15.58 s | 14,470 | 5.88 | 0/93 | 0/93 |
| Tailwind CSS | Neo4j-capable | 0.7849 | 0.7957 | 0.6794 | 0.7742 | 9.59 / 24.21 s | 16,724 | 5.84 | 0/93 | 0/93 |

Material claim issues count answers with material unsupported claims or contradictions.
Critical-aspect and citation-integrity diagnostics retain their judged values even for
invalid trajectories; interpret them alongside coverage and validity.

### Paired answer-quality comparison

| WAC contrast | Mean delta | Descriptive 95% paired interval |
|---|---:|---:|
| Hybrid − Filesystem | -0.0042 | [-0.0244, +0.0162] |
| Neo4j-capable − Hybrid | +0.0112 | [-0.0054, +0.0277] |
| Neo4j-capable − Filesystem | +0.0070 | [-0.0135, +0.0280] |

Intervals use 10,000 paired resamples of the same 467 questions (seed 20260831). These
are descriptive intervals on a development-exposed pool, not evidence of generalization
to new products.

All three intervals include zero. This run does not establish a clear overall
WAC advantage for one arm, although the indexed arms use substantially fewer
generation tokens. Final-source recovery and answer coverage also differ:
Prisma hybrid Recall@10 is 0.1440 while WAC is 0.7870. Sparse cited-page labels
and answer aspects measure different targets; neither should replace the other
or the material-claim-issue diagnostics.

### Usage, execution and provenance

| Arm | Generation tokens | Tokens / QA | p50 / p95 agent latency | Tool calls / QA |
|---|---:|---:|---:|---:|
| Filesystem | 15,594,809 | 33,394 | 9.53 / 16.03 s | 8.21 |
| Hybrid | 6,160,798 | 13,192 | 10.22 / 15.21 s | 5.67 |
| Neo4j-capable | 6,391,246 | 13,686 | 9.41 / 14.37 s | 5.04 |

The 467 final paired judgments report 21,503,505 tokens; median judge latency is 16.25
s. Generation tokens are runner-reported totals, including cached input; they are not
dollar cost. The judge's largest complete input is 875,394 UTF-8 bytes, and 24 questions
exceed the default 500,000-byte local ceiling. This run explicitly used
`--max-prompt-bytes none`; API input truncation remained disabled.

All final judgments use Luna with medium reasoning, streaming, exactly three candidates
and the frozen aspect IDs/counts, and a uniform 16,384-token output budget. Incomplete
or malformed responses receive no score. Earlier attempts yielded 49 valid diagnostic
judgments (46 unbounded, two bounded-v1, one enum-format probe); none enters these
tables. The final schema enumerates the five overall-quality values explicitly,
resolving a repeated whitespace-generation failure at that field without changing its
meaning. Failed or interrupted requests are retained in local attempt logs and may lack
returned token usage, so reported successful-response usage is not a complete billing
total. Technical pilot judgments were reused only under the unchanged final
configuration.

Answer generation ran from September 21, 22:32 to September 22, 02:38 UTC. Final judging
completed September 22 at 04:19 UTC. Agent arms ran sequentially with isolated DSH homes
and four Torch threads; the judge used four workers. Every answer is the original
first-pass output. The eight technical-pilot judgments were reused within the unchanged
final configuration; earlier diagnostic judgments were excluded.

The dataset commit and model/index revisions match the September 21 retrieval rerun
below. Source provenance is `5cebdb9` plus this PR’s scope-enforcement changes for
generation, followed by its bounded structured-output judge change. Phase-specific
code/data hashes and all rollout hashes were verified. Independent checks recomputed all
1,401 retrieval rows, all 1,401 WAC rows, and every project aggregate. Detailed outputs
and verification records stay in the ignored local `phase13_scoped_agents/` directory.

The Neo4j snapshot includes the existing KGGen artifact. Its identifiers are:

```text
snapshot:   037b7865cd56839115012d3d485c5f9dce9cf6ed896812378e3056c84b711ca8
corpus:     4e7504a8e497e805f3b81905bbd380cfd34acf3cd7ef38d61c9519e491d1e519
chunks:     79f6bc546ce4fffa7630c0980f579e041d2626db731a8a6066886d892e8b6566
KGGen:      66f2f52ff675f561d07d41b3699e22c82f466add08f091b7adc4bbe39abcc183
embeddings: c07139d92f735739fc6e3a41c960a25e8ed876aa65e9934a97ab721a39eca21c
```

Graph expansion succeeded on only **2/467 Neo4j episodes, both Prisma**. The arm-level
comparison therefore cannot establish a general graph-traversal benefit. Product
differences also mix content, corpus size and question types; they do not isolate
topology. This run has no full-corpus agent control and no matched fixed-RAG answer
control. Historical 361-question scores used a different cohort, scope and truncated
judge inputs, so before/after differences cannot be attributed to removing truncation
alone.

## Current-entry-point scope rerun (2026-09-21)

The current evaluator was rerun on all 467 questions with five methods and
both `--search-scope project` and `--search-scope corpus`: **4,670 evaluations**.
Models were loaded offline; no LLM API calls were made. Each run uses the same
pinned 4,860 pages, cached embeddings, chunking, candidate/rerank/graph budgets,
and final top 10 as the earlier five-method comparison.

| Method | Product-only Recall@10 | Full-corpus Recall@10 | Product-only nDCG@10 | Full-corpus nDCG@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.4660 | 0.4379 | 0.2913 | 0.2693 |
| Dense | 0.6294 | 0.6133 | 0.4044 | 0.3957 |
| Hybrid | 0.6045 | 0.5842 | 0.3822 | 0.3719 |
| Hybrid + reranker | 0.4971 | 0.4892 | 0.2998 | 0.2947 |
| Hybrid + native links | 0.6218 | 0.6068 | 0.3918 | 0.3806 |

### Four projects, two search settings

The four projects are dataset groups. The two scopes are search settings,
and each setting is evaluated separately on all four groups:

- **Primary setting (`project`):** each question searches only its own product’s documentation.
- **Supplementary control (`corpus`):** each question can search documentation from all four products.

| Project | Questions | Pages available in primary setting |
|---|---:|---:|
| GitHub Docs | 197 | 3,208 |
| Prisma | 125 | 685 |
| Supabase | 52 | 770 |
| Tailwind CSS | 93 | 197 |
| **Total** | **467** | **4,860** |

Recall is the fraction of cited pages recovered; Hit is the fraction of
questions with at least one cited page recovered; nDCG also rewards earlier
positions; AllSupport is the fraction with every cited page recovered.
All metrics use the top 10 and are averaged over questions, not projects.

### Primary results: search the question’s product

| Project | Method | Recall@10 | Hit@10 | nDCG@10 | AllSupport@10 |
|---|---|---:|---:|---:|---:|
| GitHub Docs | BM25 | 0.4432 | 0.4822 | 0.2794 | 0.4061 |
| GitHub Docs | Dense | 0.6088 | 0.6650 | 0.4107 | 0.5482 |
| GitHub Docs | Hybrid | 0.5853 | 0.6396 | 0.3984 | 0.5381 |
| GitHub Docs | Hybrid + reranker | 0.4863 | 0.5330 | 0.3124 | 0.4467 |
| GitHub Docs | Hybrid + native links | 0.6213 | 0.6650 | 0.4233 | 0.5838 |
| Prisma | BM25 | 0.2587 | 0.3120 | 0.1540 | 0.2080 |
| Prisma | Dense | 0.4640 | 0.5360 | 0.2719 | 0.4000 |
| Prisma | Hybrid | 0.4080 | 0.4560 | 0.2255 | 0.3600 |
| Prisma | Hybrid + reranker | 0.2747 | 0.3360 | 0.1449 | 0.2240 |
| Prisma | Hybrid + native links | 0.3960 | 0.4480 | 0.2255 | 0.3440 |
| Supabase | BM25 | 0.4135 | 0.4615 | 0.2606 | 0.3654 |
| Supabase | Dense | 0.5673 | 0.6154 | 0.3699 | 0.5192 |
| Supabase | Hybrid | 0.5673 | 0.6154 | 0.3455 | 0.5192 |
| Supabase | Hybrid + reranker | 0.5288 | 0.5769 | 0.2890 | 0.4808 |
| Supabase | Hybrid + native links | 0.5962 | 0.6346 | 0.3428 | 0.5577 |
| Tailwind CSS | BM25 | 0.8226 | 0.8387 | 0.5183 | 0.8065 |
| Tailwind CSS | Dense | 0.9301 | 0.9355 | 0.5887 | 0.9247 |
| Tailwind CSS | Hybrid | 0.9301 | 0.9462 | 0.5788 | 0.9140 |
| Tailwind CSS | Hybrid + reranker | 0.8011 | 0.8172 | 0.4873 | 0.7849 |
| Tailwind CSS | Hybrid + native links | 0.9409 | 0.9462 | 0.5761 | 0.9355 |

### Supplementary results: search all four products

| Project | Method | Recall@10 | Hit@10 | nDCG@10 | AllSupport@10 |
|---|---|---:|---:|---:|---:|
| GitHub Docs | BM25 | 0.4347 | 0.4721 | 0.2724 | 0.4010 |
| GitHub Docs | Dense | 0.5986 | 0.6548 | 0.4068 | 0.5381 |
| GitHub Docs | Hybrid | 0.5752 | 0.6294 | 0.3942 | 0.5279 |
| GitHub Docs | Hybrid + reranker | 0.4931 | 0.5431 | 0.3152 | 0.4518 |
| GitHub Docs | Hybrid + native links | 0.6213 | 0.6650 | 0.4187 | 0.5838 |
| Prisma | BM25 | 0.2467 | 0.2880 | 0.1465 | 0.2080 |
| Prisma | Dense | 0.4640 | 0.5360 | 0.2694 | 0.4000 |
| Prisma | Hybrid | 0.3960 | 0.4400 | 0.2195 | 0.3520 |
| Prisma | Hybrid + reranker | 0.2747 | 0.3360 | 0.1434 | 0.2240 |
| Prisma | Hybrid + native links | 0.4000 | 0.4560 | 0.2215 | 0.3440 |
| Supabase | BM25 | 0.3846 | 0.4231 | 0.2346 | 0.3462 |
| Supabase | Dense | 0.5481 | 0.5962 | 0.3484 | 0.5000 |
| Supabase | Hybrid | 0.5481 | 0.5962 | 0.3356 | 0.5000 |
| Supabase | Hybrid + reranker | 0.5000 | 0.5577 | 0.2716 | 0.4423 |
| Supabase | Hybrid + native links | 0.5481 | 0.5962 | 0.3152 | 0.5000 |
| Tailwind CSS | BM25 | 0.7312 | 0.7419 | 0.4470 | 0.7204 |
| Tailwind CSS | Dense | 0.8817 | 0.8925 | 0.5683 | 0.8710 |
| Tailwind CSS | Hybrid | 0.8763 | 0.8925 | 0.5498 | 0.8602 |
| Tailwind CSS | Hybrid + reranker | 0.7634 | 0.7634 | 0.4677 | 0.7634 |
| Tailwind CSS | Hybrid + native links | 0.8871 | 0.8925 | 0.5505 | 0.8817 |

### Scope checks and interpretation

| Method | Full-corpus questions returning any outside-product page | Project minus corpus Recall@10 | Descriptive 95% paired interval |
|---|---:|---:|---:|
| BM25 | 230/467 | +0.0282 | [+0.0150, +0.0428] |
| Dense | 84/467 | +0.0161 | [+0.0064, +0.0278] |
| Hybrid | 147/467 | +0.0203 | [+0.0054, +0.0364] |
| Hybrid + reranker | 127/467 | +0.0079 | [-0.0075, +0.0239] |
| Hybrid + native links | 69/467 | +0.0150 | [-0.0021, +0.0332] |

Intervals use 10,000 paired resamples of the same 467 questions (seed
20260921). They describe this reused pool and do not establish a causal effect
of scope. Outside-product pages are permitted in the supplementary setting;
their presence alone does not establish answer irrelevance.

Every project-scoped result contains zero outside-product pages. All 4,670
per-query Recall, Hit and nDCG values were independently recomputed; input and
source hashes stayed unchanged during execution. Reports now include
`by_project` and `by_search_scope` tables for every method. Full scope/product
metrics, AllSupport@10, per-query rankings, paired descriptive intervals and
provenance are retained locally in `phase12_scope_rerun/`.

**Interpretation:** this entry point shares full-corpus BM25 statistics. Its
project scope uses exact cosine over allowed chunks; corpus scope uses HNSW.
Dense/hybrid contrasts therefore include an implementation difference, not
just filtering. The earlier product-index experiment below built separate
indexes and must remain a separate result. Neither experiment establishes that
document structure alone causes the differences.

The historical 361-question agent answers were not regenerated with enforced
project scope, and their truncated-input judge scores were not rerun in this
retrieval experiment. They are not updated full-evidence answer results.

### Reproduction and provenance

The run uses data commit `19af578bead6c8317d29598c409e982886951cbe` from
`PowderXu/docsqa-data` (manifest SHA-256
`c6193cc88cdf88adc2c8561441b03280415bf871e0b22f7d006a5968c714a361`).
The 4,860 pages produce 31,608 chunks. Models are
`sentence-transformers/all-MiniLM-L6-v2` (revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`) and
`cross-encoder/ms-marco-MiniLM-L-6-v2` (revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`).
The run records source base `ad9ba34` plus this PR’s aggregate-report change;
full file hashes are retained in the local provenance record.

Reproduce either scope using the downloaded dataset and the same cached models:

```bash
PYTHONPATH=evaluation:. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  evaluation/.venv/bin/python -m kbbench.retrieval \
  --dataset-dir evaluation/dataset/evaluation_data/normalized \
  --cache-dir dsh_plugin/plugin/data/hybrid/indexes \
  --output-dir results/runs/retrieval/scope-project \
  --search-scope project --local-files-only --device cpu --top-k 10 \
  --retrieval-depth 50 --rerank-depth 45 \
  --graph-seed-depth 5 --graph-max-neighbors 10 --graph-max-hops 2 \
  --graph-max-candidates 50 --graph-hop-decay 0.5 --graph-weight 0.25 \
  --methods bm25 hnsw bm25_hnsw_rrf bm25_hnsw_rerank hybrid_multihop_link_expansion
```

Use `--search-scope corpus` and a different output directory for the full-corpus
run. Hardware latency is descriptive; the local runner fixed Torch to four
threads and ran scopes sequentially.

## Earlier separate-product-index comparison (2026-09-16)

Five existing methods were run on all 467 questions (2,335 evaluations), each
restricted to its product's documentation. All use the same corpus and chunker,
50 chunk candidates per base retriever and a final top 10. Hybrid uses RRF;
the reranker scores up to 45 hybrid pages with `ms-marco-MiniLM-L-6-v2`.
Dense retrieval uses `all-MiniLM-L6-v2`. Native-link expansion uses the existing
offline graph baseline: five seeds, two hops, at most 50 discovered non-seed
pages, weight 0.25 and hop decay 0.5. This is separate from the Neo4j agent arm.
No new retrieval method, API call or parameter tuning was introduced.

| Method | Recall@10 | Hit@10 | nDCG@10 | AllSupport@10 |
|---|---:|---:|---:|---:|
| BM25 | 0.4718 | 0.5096 | 0.2895 | 0.4347 |
| Dense | 0.6294 | 0.6788 | 0.4044 | 0.5803 |
| Hybrid | 0.6070 | 0.6510 | 0.3911 | 0.5653 |
| Hybrid + reranker | 0.5164 | 0.5589 | 0.3135 | 0.4775 |
| Hybrid + native links | 0.6104 | 0.6488 | 0.3799 | 0.5739 |

Hybrid recovers 329 of the 601 annotated question–page pairs in its top 10;
147 are absent from its candidate pool and 125 rank below 10. Native links
reduce candidate misses to 108, but increase ranking misses to 164; the total
recovered pairs stays at 329. Its per-question Recall@10 change is only
+0.34 percentage points (descriptive paired 95% bootstrap interval
−1.66 to +2.39). The reranker reduces mean recall by 9.06 points
(interval −13.13 to −5.07); these outcomes were not tuned away.

There are 366 single-citation and 101 multiple-citation questions. Hybrid
Recall@10 is 0.669 versus 0.381; native-link hybrid is 0.680 versus 0.357.
Multiple citations do not prove necessary multi-file reasoning. Product and
intent slices are retained, but do not isolate the effect of document topology.

Input/code fingerprints stayed unchanged throughout the run. All 2,335 scores
passed independent Recall, Hit and nDCG recomputation. Detailed rankings,
candidate traces, failure slices and frozen settings are retained in the ignored
local `phase11_research_plan` research directory. The evaluated release files
have the same manifest hash as the newly pinned data commit `19af578`.
These metrics measure cited-page recovery, not full answer correctness; unjudged
pages may be useful. Step 1 is deferred as requested. The matched fixed-RAG
versus adaptive-agent answer comparison remains pending.

## Dataset construction

The fixed manifest contains 798 answered support discussions. Deterministic
source validation retains 556 packages whose required links and linked answers
can be resolved locally. The v14 normalization run retains 467 records above
its dataset-construction gate.

| Project | Structurally eligible | Normalized | Rejected by normalization |
|---|---:|---:|---:|
| GitHub Docs | 232 | 197 | 35 |
| Prisma | 148 | 125 | 23 |
| Supabase | 67 | 52 | 15 |
| Tailwind CSS | 109 | 93 | 16 |
| **Total** | **556** | **467** | **89** |

The normalized package has 4,860 searchable documentation pages and 601 sparse
page qrels. Sixty-nine normalized questions contain locally reproduced
image-derived question text; 53 occur in the 361-question development pool.

Reproduce the normalized package from retained v14 work state without a model
call:

```bash
PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m dataset_analysis.normalize_dataset \
  --dataset evaluation/dataset/evaluation_data/combined \
  --work-dir results/runs/dataset-analysis/normalization-v14 \
  --output-dir evaluation/dataset/evaluation_data/normalized \
  --model gpt-5.6-luna \
  --reasoning-effort medium \
  --threshold 0.90 \
  --max-repairs 2 \
  --materialize-only
```

## Fixed answer aspects

The pinned dataset provides 1,926 frozen aspects for all 467 questions
(mean 4.12 per question). These model-generated annotations specify required
facts, actions, and conditions, together with weights and evidence mappings.
Each evaluated system uses the same annotations; the LLM judge assigns coverage
labels to its answers and deterministic code computes WAC.

The annotations were prepared before agent evaluation and have not been
independently verified by domain experts. Their schema, scoring boundary, and
historical source are documented in [aspect annotations](ASPECT_ANNOTATIONS.md).

## Historical trajectory retrieval evaluation (361 questions)

The historical 361-question comparison used contemporaneous source versions.
The original answers and scores are preserved below. The fresh whole-pool run
uses enforced project scope and a full-evidence judge. Historical arms have matching question IDs, corpus, model, non-retrieval tools,
answer contract, scoring implementation, skill/runtime generation, and locked
dependencies. The three runners used isolated DSH homes and separate indexed
services.

| Agent arm | Recall@10 | Hit@10 | nDCG@10 | AllSupport@10 | p50 / p95 latency | Tokens / QA | Tool calls / QA | Invalid trajectories | Graph used |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Filesystem | 0.4449 | 0.4792 | 0.3944 | 0.4127 | 9.65 / 15.12 s | 38,034 | 7.91 | 43/361 | 0/361 |
| Hybrid | 0.4829 | 0.5208 | 0.4324 | 0.4488 | 9.64 / 14.93 s | **11,786** | 4.89 | 3/361 | 0/361 |
| Neo4j-capable | **0.5000** | **0.5346** | **0.4491** | **0.4681** | **9.08 / 13.81 s** | 12,680 | **4.36** | **2/361** | **9/361** |

The filesystem system uses about 3.23 times the hybrid tokens and has 40 more
invalid trajectories. Neo4j graph expansion occurs on only nine questions, so
the aggregate Neo4j-capable row does not isolate a graph-expansion effect.

## Reproduce the current whole-pool agent comparison

After downloading the pinned dataset and installing the DSH profile dependencies,
prepare each arm and start Neo4j for the graph arm:

```bash
for arm in fs hybrid neo4j; do
  PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
    -m dsh_plugin.backend.prepare_plugin_data \
    --arm "$arm" \
    --source-dataset evaluation/dataset/evaluation_data/normalized
done

docker compose -f dsh_plugin/backend/compose.neo4j.yml up -d
PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m dsh_plugin.backend.ingest_neo4j
```

Run each arm against its corresponding prepared store. The commands below use
sequential execution and isolated DSH homes, as in the current run. Start with
a new output directory. The recorded comparison preserves first-pass agent
failures; it does not retry them with `--resume`.

The recorded Neo4j snapshot includes an existing KGGen artifact, not just native
Markdown links. Reproducing its graph requires the same artifact through
`ingest_neo4j --kggen-artifact` (the default location is the prepared Neo4j
store's `artifacts/kggen_graph.json`). Its hashes are recorded with the current
results. Ingesting only native links produces a different graph configuration.

```bash
RUN_ROOT="$PWD/results/runs/agents/whole-pool"
PROFILE_TEMPLATE="$PWD/dsh_plugin/dsh_home/profiles/headless"
for arm in fs hybrid neo4j; do
  ARM_HOME="$RUN_ROOT/homes/$arm"
  mkdir -p "$ARM_HOME/profiles/headless"
  cp "$PROFILE_TEMPLATE/"*.yml "$PROFILE_TEMPLATE/"*.json \
    "$ARM_HOME/profiles/headless/"
  ln -s "$PROFILE_TEMPLATE/node_modules" "$ARM_HOME/profiles/headless/node_modules"
  KBBENCH_OPENAI_ENV_FILE=/absolute/path/to/private.env \
  PYTHONPATH=evaluation:. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  evaluation/.venv/bin/python \
    -m dsh_plugin.agent_eval.runner \
    --arm "$arm" \
    --search-scope project \
    --dataset-dir evaluation/dataset/evaluation_data/normalized \
    --workspace "dsh_plugin/plugin/data/$arm/documents" \
    --cache-dir dsh_plugin/plugin/data/hybrid/indexes \
    --dsh-home "$ARM_HOME" \
    --output-dir "$RUN_ROOT/$arm"
done

PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m dsh_plugin.agent_eval.report \
  --fs "$RUN_ROOT/fs" \
  --hybrid "$RUN_ROOT/hybrid" \
  --neo4j "$RUN_ROOT/neo4j" \
  --expected-ids evaluation/dataset/evaluation_data/normalized/questions.jsonl \
  --out-dir "$RUN_ROOT/trajectory-report"
```

## Historical WAC final-answer evaluation (361 questions)

**Historical judge-input limitation:** the 361-question answer scores below
used at most the first 6,000 characters of each selected candidate document
and 12,000 characters of each answer. Frozen gold evidence was supplied
separately without these cutoffs. The current evaluator preserves full text
and applies an optional input-size limit (500,000 bytes by default) (see [the protocol](EVALUATION_PROTOCOL.md)),
and a separate output-generation budget. The fresh whole-pool run uses that
policy. Comparing it with this historical table does not isolate the effect
of removing truncation: question pool, scope and generated answers also changed.

The matched answers were judged against the fixed aspect annotations with a
paired `gpt-5.6-luna` call per question. Weighted Aspect Coverage (WAC) follows
the formula and `{0, 0.5, 1}` aspect scale used by the
[BRIGHT-Pro paper](https://arxiv.org/abs/2605.04018) and its
[official evaluator](https://github.com/yale-nlp/Bright-Pro/blob/main/agentic_retrieval/scripts_evaluation/judge.py).
WAC and all other rates use all 361 questions, including invalid agent
trajectories, which receive zero WAC.

| Agent arm | WAC (N=361) | Critical-aspect success | Complete | Partial | Incorrect | Unsupported claims | Citation integrity |
|---|---:|---:|---:|---:|---:|---:|---:|
| Filesystem | 0.6238 | 0.5130 | 0.3241 | 0.3075 | 0.3684 | **0.2382** | 0.9834 |
| Hybrid | 0.6996 | 0.5565 | 0.3518 | 0.3573 | **0.2909** | 0.2687 | 0.9834 |
| Neo4j-capable | **0.7273** | **0.5884** | **0.3684** | 0.3407 | **0.2909** | 0.2742 | 0.9834 |

| Paired WAC contrast | Mean delta | 95% paired bootstrap interval |
|---|---:|---:|
| Hybrid - filesystem | +0.0758 | [0.0409, 0.1121] |
| Neo4j-capable - hybrid | +0.0277 | [0.0057, 0.0494] |
| Neo4j-capable - filesystem | +0.1036 | [0.0698, 0.1385] |

The intervals use 10,000 paired bootstrap samples over all 361 matched
questions. Neo4j-capable retrieval has the highest coverage and critical-aspect
success, but also the highest unsupported-claim rate. Coverage and
hallucination diagnostics must therefore be interpreted together. The 361
persisted judge calls use 5,511,339 total tokens; median judge-call latency is
23.05 seconds.

## Reproduce the current full-evidence answer evaluation

After running all three agents on the full question pool, judge their matched
answers as follows. `none` disables the local input-byte ceiling; all selected
document text and answers are preserved, and API input truncation is disabled.
The output budget limits the judge's own generation and fails incomplete
judgments without recording a score.

```bash
RUN_ROOT=results/runs/agents/whole-pool
ASPECTS=evaluation/dataset/evaluation_data/normalized/aspects.jsonl

KBBENCH_OPENAI_ENV_FILE=/absolute/path/to/private.env \
PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m dsh_plugin.agent_eval.aspect_judge \
  --aspects "$ASPECTS" \
  --corpus evaluation/dataset/evaluation_data/normalized/corpus.jsonl \
  --arm fs="$RUN_ROOT/fs/rollouts.json" \
  --arm hybrid="$RUN_ROOT/hybrid/rollouts.json" \
  --arm neo4j="$RUN_ROOT/neo4j/rollouts.json" \
  --output-dir "$RUN_ROOT/aspect-judge" \
  --model gpt-5.6-luna \
  --reasoning-effort medium \
  --max-prompt-bytes none \
  --max-output-tokens 16384 \
  --request-timeout 600 \
  --workers 4 \
  --resume
```

The downloaded annotation file covers all 467 questions. `--resume` reuses only
validated judge caches with matching inputs, model, rubric and output policy.
The historical 361-question table is not a result of these whole-pool commands.
New reports contain WAC, critical-aspect success, unsupported claims,
citation integrity, and paired bootstrap intervals.

## Publication boundary

Historical results show that retrieval and answer quality can improve while
the material-claim-issue rate also increases, and that similar latency can hide
large token and validity differences. Graph-capable configuration
does not imply frequent graph use. The fixed aspect annotations and WAC judge
have not been validated against domain experts. A publication release still
needs an independent project-stratified human audit and a newly sealed system
cohort.
