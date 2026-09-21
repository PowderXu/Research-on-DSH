# Current results

> The current dataset is one 467-question pool with no dataset partitions.
> The retrieval-only comparison below uses that full pool. Later agent and
> answer tables retain their original historical sample sizes and settings.

Status date: **2026-09-21**.

This file is the maintained result index. Generated datasets, model outputs,
trajectories, indexes and detailed reports remain in ignored local directories.
Current results are exploratory because the question pool was used in development.
See [the research plan](RESEARCH_ANALYSIS.md) for completed and pending steps.

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

## Trajectory retrieval evaluation

The historical 361-question comparison used contemporaneous source versions.
It was not rerun with the project-scope restriction or the full-evidence judge. All arms have matching question IDs, corpus, model, non-retrieval tools,
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

To produce a fully matched current-source run, first prepare each arm and start
Neo4j for the graph arm:

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
sequential execution for simplicity; the reported run used isolated DSH homes
and separate indexed services so the three arms could run contemporaneously:

```bash
RUN_ROOT=results/runs/agents/whole-pool
for arm in fs hybrid neo4j; do
  KBBENCH_OPENAI_ENV_FILE=/absolute/path/to/private.env \
  PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
    -m dsh_plugin.agent_eval.runner \
    --arm "$arm" \
    --output-dir "$RUN_ROOT/$arm" \
    --resume
done

PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m dsh_plugin.agent_eval.report \
  --fs "$RUN_ROOT/fs" \
  --hybrid "$RUN_ROOT/hybrid" \
  --neo4j "$RUN_ROOT/neo4j" \
  --expected-ids evaluation/dataset/evaluation_data/normalized/questions.jsonl \
  --out-dir "$RUN_ROOT/trajectory-report"
```

## WAC final-answer evaluation

**Historical judge-input limitation:** the 361-question answer scores below
used at most the first 6,000 characters of each selected candidate document
and 12,000 characters of each answer. Frozen gold evidence was supplied
separately without these cutoffs. The current evaluator preserves full text
and applies an optional input-size limit (500,000 bytes by default) (see [the protocol](EVALUATION_PROTOCOL.md)),
but these answer scores have not been rerun under that policy. The effect on
scores and rankings is unmeasured; retrieval Recall and nDCG are unaffected.

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

After downloading the pinned dataset and running all three agents on the full
question pool, judge their matched answers as follows:

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
  --request-timeout 600 \
  --workers 4 \
  --resume
```

The downloaded annotation file covers all 467 questions. The historical table
above used 361 questions and is not a result of these whole-pool commands.
New reports contain WAC, critical-aspect success, unsupported claims,
citation integrity, and paired bootstrap intervals.

## Publication boundary

Current results show that retrieval and answer quality can improve while
unsupported-claim rate also increases, and that similar latency can hide large
token and validity differences. They also show that graph-capable configuration
does not imply frequent graph use. The fixed aspect annotations and WAC judge
have not been validated against domain experts. A publication release still
needs an independent project-stratified human audit and a newly sealed system
cohort.
