# Current results

> The current dataset is one 467-question pool with no dataset partitions.
> The retrieval-only comparison below uses that full pool. Later agent and
> answer tables retain their original historical sample sizes and settings.

Status date: **2026-09-16**.

This file is the maintained result index. Generated datasets, model outputs,
trajectories, indexes and detailed reports remain in ignored local directories.
Current results are exploratory because the question pool was used in development.
See [the research plan](RESEARCH_ANALYSIS.md) for completed and pending steps.

## Current-release retrieval comparison

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

The completed 361-question comparison uses contemporaneous current-source
runs. All arms have matching question IDs, corpus, model, non-retrieval tools,
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
and rejects oversized inputs (see [the protocol](EVALUATION_PROTOCOL.md)),
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
