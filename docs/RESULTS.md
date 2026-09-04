# Current results

Status date: **2026-09-04**.

This file is the only maintained result index. Generated datasets, model
outputs, trajectories, indexes, and per-run reports remain under ignored local
paths. The 361-question system results are exploratory because that physical
split was inspected during development.

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

## Aspect-rule optimization

The current implementation creates a deterministic project-stratified split
from all 467 normalized records:

| Partition | GitHub Docs | Prisma | Supabase | Tailwind CSS | Total |
|---|---:|---:|---:|---:|---:|
| Train | 118 | 75 | 31 | 56 | **280** |
| Validation | 40 | 25 | 10 | 19 | **94** |
| Held-out test | 39 | 25 | 11 | 18 | **93** |

The complete authorized Sol/Luna run finished on 2026-09-03:

| Stage | Questions | Pass rate | Mean source coverage |
|---|---:|---:|---:|
| Initial-rule validation | 94 | 0.9574 | 0.9657 |
| Best-rule validation | 94 | **0.9681** | **0.9681** |
| Frozen held-out test | 93 | 0.9462 | 0.9462 |
| Raw all-record generation | 467 | 0.9507 | — |

SkillOpt accepted one of five edits and selected step 3. The final freeze
contains all 467 questions and 1,926 aspects (mean 4.12). Raw generation
produced 445 structurally materializable records; deterministic binding of
exact evidence IDs from mapped normalized claims repaired 44 aspect fields in
the 22 raw structural failures. No API or JSON-parsing failures occurred.
`rule_eligibility.json` therefore marks the artifact
eligible for agent judging.

| Phase | Model calls | Prompt tokens | Completion tokens | Total tokens |
|---|---:|---:|---:|---:|
| SkillOpt train/selection | 911 | 6,869,089 | 1,175,093 | 8,044,182 |
| Held-out test | 93 | 437,149 | 116,210 | 553,359 |
| All-record aspect freeze | 467 | 2,240,376 | 592,087 | 2,832,463 |
| **Total** | **1,471** | **9,546,614** | **1,883,390** | **11,430,004** |

This is weak-supervision compatibility, not human-agreement evidence. The
completed final-answer results below use only this optimized rule and its
frozen aspects; superseded judge artifacts remain excluded.

Run the real optimization and aspect freeze with:

```bash
export KBBENCH_OPENAI_ENV_FILE=/absolute/path/to/private.env
PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m rule_optimization.run \
  --config evaluation/rule_optimization/config.yaml
```

The run is accepted only if `rule_eligibility.json` reports validation pass
rate at least 0.80, a completed held-out test, and all 467 frozen aspect
records. The completed local run satisfies these conditions. Outputs are written to
`results/runs/rule-optimization/skillopt-sol-luna/`.

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
RUN_ROOT=results/runs/agents/current-matched
for arm in fs hybrid neo4j; do
  KBBENCH_OPENAI_ENV_FILE=/absolute/path/to/private.env \
  PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
    -m dsh_plugin.agent_eval.runner \
    --arm "$arm" \
    --split test \
    --output-dir "$RUN_ROOT/$arm" \
    --resume
done

PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m dsh_plugin.agent_eval.report \
  --fs "$RUN_ROOT/fs" \
  --hybrid "$RUN_ROOT/hybrid" \
  --neo4j "$RUN_ROOT/neo4j" \
  --expected-ids evaluation/dataset/evaluation_data/normalized/splits/test.json \
  --out-dir "$RUN_ROOT/trajectory-report"
```

## WAC final-answer evaluation

The matched answers were judged with the optimized rule's frozen aspects and a
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

After a successful rule-optimization run has written all 467 frozen aspects,
judge the matched agent answers as follows:

```bash
RUN_ROOT=results/runs/agents/current-matched
ASPECTS=$RUN_ROOT/test_frozen_aspects.jsonl

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

`test_frozen_aspects.jsonl` is the exact question-ID intersection of the 467
all-record frozen aspects and the 361-question physical test split. The report
contains WAC, critical-aspect success, unsupported
claims, citation integrity, and paired bootstrap intervals.

## Publication boundary

Current results show that retrieval and answer quality can improve while
unsupported-claim rate also increases, and that similar latency can hide large
token and validity differences. They also show that graph-capable configuration
does not imply frequent graph use. The optimized aspect rule and WAC judge
have not been validated against domain experts. A publication release still
needs an independent project-stratified human audit and a newly sealed system
cohort.
