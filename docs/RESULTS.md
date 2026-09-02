# Current results

Status date: **2026-09-01**.

This is the single maintained result index. Generated JSONL, traces, indexes,
and run-specific reports stay under ignored `results/`. A result is removed
from this page when its dataset, representation, metric, or judge is superseded.

The 361 normalized questions stored under the physical split name `test` have
been inspected while the retrievers and evaluator were developed. This page
therefore calls them the **development partition**, and all system measurements
on them exploratory. A formal zero-shot result requires a new sealed cohort.

The current judge protocol overlays all 467 normalized questions with 280
calibration-train, 93 calibration-validation, and 94 final-test records. This
overlay is evaluator-only and does not alter agent runs. The selected evaluator
passed calibration train and validation with `--max-variant-repairs 2`, then
passed the one-time frozen 94-question final test. The system measurements later
on this page remain exploratory because they use the inspected 361-question
development partition.

## Safe replay and new runs

Frozen directories under `results/runs/` are paper evidence. Do not use them as
the output of a model-writing command. Commands in this section either replay
frozen artifacts without an API call or write to a separate current-source run
root.

### Replay frozen dataset and evaluator artifacts

In a workspace containing the frozen v14 work state, rebuild the normalized
package without a model call:

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

PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m dataset_analysis.build_weak_supervision_splits \
  --questions evaluation/dataset/evaluation_data/normalized/questions.jsonl \
  --output evaluation/dataset/evaluation_data/normalized/weak_supervision_split.json \
  --seed 20260901
```

Recompute the frozen final-test judge report from retained silver cases and
judgments:

```bash
PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m dataset_analysis.calibrate_aspect_judge \
  --aspects results/runs/dataset-analysis/aspects-v2-all/aspects.jsonl \
  --output-dir results/runs/dataset-analysis/aspect-calibration-v2-final-test \
  --model gpt-5.6-luna \
  --reasoning-effort medium \
  --question-manifest evaluation/dataset/evaluation_data/normalized/weak_supervision_split.json \
  --manifest-partition final_test \
  --rubric-role frozen \
  --max-variant-repairs 2 \
  --min-source-answer-complete-recall 0.80 \
  --min-partial-recall 0.80 \
  --min-incorrect-recall 0.80 \
  --max-incorrect-false-complete-rate 0.05 \
  --min-macro-f1 0.80 \
  --min-balanced-case-rate 0.98 \
  --report-only
```

Recompute the historical trajectory aggregates into a new directory:

```bash
PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m dsh_plugin.agent_eval.report \
  --fs results/runs/agents/development/fs \
  --hybrid results/runs/agents/development/hybrid \
  --neo4j results/runs/agents/development/neo4j \
  --expected-ids evaluation/dataset/evaluation_data/normalized/splits/test.json \
  --artifact-replay \
  --out-dir results/runs/agents/replayed-trajectory-report
```

Artifact replay validates the recorded contracts and identities without
claiming that the historical runtime equals the current checkout. Recompute
the C-GWAC aggregates and paired bootstrap intervals from retained per-question
judgments, again without a model call:

```bash
PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m dsh_plugin.agent_eval.aspect_judge \
  --artifact-replay-dir results/runs/agents/development/aspect-judge \
  --output-dir results/runs/agents/replayed-aspect-report
```

This replay verifies that the regenerated report exactly equals the frozen
report and refuses to overwrite its source directory. Re-judging answers is a
new experiment and must use a new output directory.

The complete aspect-construction and two-iteration weak-supervision procedure is
documented in [Aspect evaluation](ASPECT_EVALUATION.md). It must also write to a
new run root when repeated because model outputs are stochastic.

### Run a current-source DSH experiment

Install the environment, build the DSH profile, and prepare the same normalized
corpus for all three arms:

```bash
python3 -m venv evaluation/.venv
evaluation/.venv/bin/pip install -e "./evaluation[retrieval,graph,judge,test]"
npm ci --prefix dsh_plugin
npm run --prefix dsh_plugin setup:dsh-profile

for arm in fs hybrid neo4j; do
  PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
    -m dsh_plugin.backend.prepare_plugin_data \
    --arm "$arm" \
    --source-dataset evaluation/dataset/evaluation_data/normalized \
    --source-documents evaluation/dataset/docs \
    --copy-documents
done
```

The Neo4j arm additionally requires the local database and graph ingestion:

```bash
docker compose -f dsh_plugin/backend/compose.neo4j.yml up -d
PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m dsh_plugin.backend.ingest_neo4j
```

Run arms sequentially under a new root. `--allow-model-download` is needed only
until MiniLM is present in the local model cache:

```bash
RUN_ROOT=results/runs/agents/current-source

for arm in fs hybrid neo4j; do
  KBBENCH_OPENAI_ENV_FILE=/absolute/path/to/private.env \
  PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
    -m dsh_plugin.agent_eval.runner \
    --arm "$arm" \
    --split test \
    --allow-model-download \
    --output-dir "$RUN_ROOT/$arm" \
    --resume
done
```

Aggregate the new trajectories with strict current-source fingerprint checks,
then judge the three anonymous answers together:

```bash
PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m dsh_plugin.agent_eval.report \
  --fs "$RUN_ROOT/fs" \
  --hybrid "$RUN_ROOT/hybrid" \
  --neo4j "$RUN_ROOT/neo4j" \
  --expected-ids evaluation/dataset/evaluation_data/normalized/splits/test.json \
  --out-dir "$RUN_ROOT/trajectory-report"

KBBENCH_OPENAI_ENV_FILE=/absolute/path/to/private.env \
PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m dsh_plugin.agent_eval.aspect_judge \
  --aspects results/runs/dataset-analysis/aspects-v2-test/aspects.jsonl \
  --corpus evaluation/dataset/evaluation_data/normalized/corpus.jsonl \
  --arm fs="$RUN_ROOT/fs/rollouts.json" \
  --arm hybrid="$RUN_ROOT/hybrid/rollouts.json" \
  --arm neo4j="$RUN_ROOT/neo4j/rollouts.json" \
  --output-dir "$RUN_ROOT/aspect-judge" \
  --model gpt-5.6-luna \
  --reasoning-effort medium \
  --workers 6 \
  --resume
```

Run the arms sequentially rather than concurrently; otherwise their latency
measurements share compute. A current-source report intentionally rejects any
stored/current skill, runtime, corpus, model, or dependency mismatch.

## Dataset normalization

The v14 `gpt-5.6-luna` run evaluated all 556 structurally eligible source
questions using only accepted-answer text, pinned internal documentation, and
local image-derived text. It retained 467 questions above the strict dual-grade
`> 0.90` gate.

| Project | Source | Accepted | Rejected |
|---|---:|---:|---:|
| GitHub Docs | 232 | 197 | 35 |
| Prisma | 148 | 125 | 23 |
| Supabase | 67 | 52 | 15 |
| Tailwind CSS | 109 | 93 | 16 |
| **Total** | **556** | **467 (84.0%)** | **89 (16.0%)** |

The accepted physical split is 33 train, 73 validation, and 361 legacy-`test`
questions. Because the last group has been reused during development, it is the
development/evaluation partition rather than held-out evidence.
The independent evaluator overlay is:

| Project | Calibration train | Calibration validation | Final test |
|---|---:|---:|---:|
| GitHub Docs | 118 | 39 | 40 |
| Prisma | 75 | 25 | 25 |
| Supabase | 31 | 10 | 11 |
| Tailwind CSS | 56 | 19 | 18 |
| **Total** | **280** | **93** | **94** |

Calibration train permits general judge-rule revision. Calibration validation
selects and freezes one exact evaluator. Final test is evaluated once after
freeze, and no rubric, rule, aspect, threshold, or model-setting edit may follow
inspection of its cases or scores. This split controls evaluator overfitting; it
does not change agent behavior or the historical system runs below.
It is prospectively held out from this judge-rule optimization only; all 467
records were already processed during dataset construction and normalization.
The generated package contains 4,860 searchable pages after dropping empty
navigation/source rows. The canonical
local run is `results/runs/dataset-analysis/normalization-v14/`.
Across all 467 normalized questions, 69 contain question-image-derived text; 53
of those image-bearing questions are in the 361-question development partition.

## Evaluator weak-supervision status

The merged aspect file contains all 467 selected questions and has SHA-256
`a15bfdb34667721f23adb1f6f9864867a586bfefff861d4a99eac60622341b36`.
The two development iterations and the post-freeze test used
`gpt-5.6-luna`, medium reasoning, the same v2 rubric, and the same predeclared
gates. A candidate count includes only accepted, independently reviewed
complete/partial/incorrect triples.

| Run | Repairs | Questions | Valid triples | Candidates | Accuracy | Macro-F1 | Complete recall | Partial recall | Incorrect recall | Incorrect→complete | Balanced rate | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| Iteration 1 train | 1 | 280 | 277 | 831 | 0.9543 | 0.9542 | 0.9603 | 0.9278 | 0.9747 | 0.0072 | 0.9893 | Pass |
| Iteration 1 validation | 1 | 93 | 88 | 264 | 0.9583 | 0.9582 | 0.9545 | 0.9318 | 0.9886 | 0.0114 | 0.9462 | **Fail** |
| Iteration 2 train | **2** | 280 | 279 | 837 | 0.9546 | 0.9546 | 0.9606 | 0.9319 | 0.9713 | 0.0108 | 0.9964 | Pass |
| Iteration 2 validation | **2** | 93 | 92 | 276 | 0.9601 | 0.9601 | 0.9783 | 0.9239 | 0.9783 | 0.0217 | 0.9892 | Pass |
| **Frozen final test** | **2** | **94** | **93** | **279** | **0.9498** | **0.9496** | **0.9355** | **0.9247** | **0.9892** | **0.0108** | **0.9894** | **Pass** |

Iteration 1 passed on calibration train. Validation failed only the predeclared
valid-balanced-triple gate: 0.9462 was below 0.98; every answer-label and
independent-review gate passed. Iteration 2 preserved accepted triples and
increased the domain-neutral `--max-variant-repairs` setting from 1 to 2 for
rejected triple construction. Both development partitions then passed, so that
exact setting was selected and frozen. The final-test cases were generated
fresh with `--max-variant-repairs 2`; all seven gates passed on the first and
only post-freeze run.

The final-test gate details are:

| Gate criterion | Target | Observed | Pass |
|---|---:|---:|:---:|
| Independent noisy-positive review | >= 0.80 | 1.0000 | Yes |
| Noisy-positive judge recall | >= 0.80 | 0.9355 | Yes |
| Partial-control recall | >= 0.80 | 0.9247 | Yes |
| Incorrect-control recall | >= 0.80 | 0.9892 | Yes |
| Incorrect-to-`complete` | <= 0.05 | 0.0108 | Yes |
| Macro-F1 | >= 0.80 | 0.9496 | Yes |
| Valid balanced-case rate | >= 0.98 | 0.9894 | Yes |

The canonical selected artifacts are:

- `results/runs/dataset-analysis/aspects-v2-all/`
- `results/runs/dataset-analysis/aspect-calibration-v2-train-r2/`
- `results/runs/dataset-analysis/aspect-calibration-v2-validation-r2/`
- `results/runs/dataset-analysis/aspect-calibration-v2-final-test/`

This confirms compatibility with the noisy-positive benchmark assumption and
discrimination among model-reviewed silver controls. It does not independently
prove that the source answers are correct and complete or that the judge agrees
with human experts; independent human verification remains future work. The
older 73-question check is superseded as the maintained evaluator result.

## Integrated DSH-agent evaluation

The trajectories reported below were generated before the skill-family rename
from `github-docs-{fs,hybrid,neo4j}` to
`docsqa-{fs,hybrid,neo4j}`. The rename corrects misleading four-corpus
nomenclature but changes the skill and runtime fingerprints. These recorded
numbers therefore remain historical measurements of the exact persisted
trajectories; regenerate all three arms before using the current source tree
for a new matched comparison.

The completed experiment contains three matched DSH-agent arms on the same 361
IDs. The documented procedure is sequential, although the retained artifacts
do not contain a separate launch-order manifest. Every trajectory used
`gpt-5.6-luna`, the expected arm skill, the same corpus/split, and identical
non-KB capabilities. Skill optimization was disabled. Retrieval metrics below
rank the agent's final ordered local sources, not the backend's first search
result set.

| Agent arm | Recall@10 | Hit@10 | nDCG@10 | All qrels found@10 | p50 / p95 latency | Tokens / QA | Tool calls / QA | Failed validity gate | Graph used |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Filesystem | 0.1932 | 0.2105 | 0.1771 | 0.1773 | 11.51 / 21.67 s | 32,669 | 11.7 | 171/361 (47.37%) | 0/361 |
| Hybrid | 0.4742 | 0.5152 | 0.4259 | 0.4377 | 9.49 / 14.50 s | **12,153** | 5.1 | 3/361 (0.83%) | 0/361 |
| Neo4j-capable | **0.5044** | **0.5402** | **0.4453** | **0.4709** | **8.57 / 13.42 s** | 12,500 | **4.3** | **2/361 (0.55%)** | **4/361 (1.11%)** |

Tokens include fresh input, cached input, and output. Filesystem failures were
165 cases without a valid search-discovered-and-read evidence path and 6 with
unresolved sources. Hybrid had 2 unresolved-source failures and 1 invalid JSON
answer; Neo4j had 1 unresolved-source failure and 1 invalid JSON answer. These
cases remain in the denominators and receive empty source rankings.

The Neo4j-capable skill called `docsqa_expand` on only four questions, and all
four expansions succeeded. It used ordinary hybrid search/fetch on the other
357. Therefore the arm's overall retrieval and answer differences cannot be
attributed to graph expansion alone; they also include stochastic agent
trajectory differences. The 1.11% application rate is the measured policy
behavior, not a hidden always-on graph treatment.

### Paired answer quality

The historical v2 development judge scored the three anonymous answers
together for every question. The primary metric, corpus-conditioned grounded weighted aspect
coverage (C-GWAC), averages the 0/0.5/1 satisfaction of weighted aspects that
have pinned local-document support. A question is scorable only when at least
one critical aspect has such support; 287/361 questions meet that condition.
All-aspect GWAC is diagnostic because it also includes accepted-answer-only
content that the pinned corpus may not support.

| Agent arm | C-GWAC (N=287) | All-aspect GWAC | Critical-aspect success | Complete / partial / incorrect | Unsupported-claim rate | Citation integrity |
|---|---:|---:|---:|---:|---:|---:|
| Filesystem | 0.3262 | 0.3162 | 0.2195 | 0.1025 / 0.2604 / 0.6371 | **0.1274** | **0.9972** |
| Hybrid | 0.6893 | 0.6816 | 0.4853 | 0.2881 / 0.4626 / 0.2493 | 0.2271 | 0.9889 |
| Neo4j-capable | **0.7040** | **0.6970** | **0.5018** | 0.2798 / 0.4543 / 0.2659 | 0.2521 | 0.9945 |

Complete/partial/incorrect and claim diagnostics use all 361 answers. The lower
filesystem unsupported-claim rate must be read with its 63.7% incorrect-answer
rate and 47.4% invalid-trajectory rate; it is not evidence of better answers.

| Paired C-GWAC contrast | Questions | Mean delta | 95% paired-bootstrap CI |
|---|---:|---:|---:|
| Hybrid − FS | 287 | +0.3631 | [0.3142, 0.4125] |
| Neo4j − hybrid | 287 | +0.0147 | [−0.0122, 0.0411] |
| Neo4j − FS | 287 | +0.3778 | [0.3303, 0.4270] |

The Neo4j−hybrid confidence interval includes zero. The experiment supports a
large answer-quality gain for indexed composite retrieval over filesystem
search under this harness, but it does not support a reliable graph-capability
gain over hybrid.

| Slice | N / C-GWAC-scorable | Filesystem | Hybrid | Neo4j-capable |
|---|---:|---:|---:|---:|
| GitHub Docs | 148 / 110 | 0.4332 | 0.6667 | 0.6674 |
| Prisma | 96 / 75 | 0.2622 | 0.7189 | **0.7494** |
| Supabase | 44 / 38 | 0.2416 | 0.6381 | **0.6587** |
| Tailwind CSS | 73 / 64 | 0.2675 | 0.7240 | **0.7408** |
| Dispersed evidence | 24 / 18 | 0.1969 | 0.6752 | **0.7090** |
| Explicitly linked evidence | 16 / 14 | 0.2854 | 0.6357 | **0.6380** |
| Original structure annotated single | 321 / 255 | 0.3375 | 0.6933 | **0.7073** |
| Question has local image-derived text | 53 / 40 | 0.3741 | 0.6487 | **0.6572** |
| Text-only question | 308 / 247 | 0.3184 | 0.6959 | **0.7116** |

The answer judge made 361 paired calls with six workers, consuming 4,790,721
tokens; median judge-call latency was 19.63 seconds. Judge usage is excluded
from agent tokens and latency. The canonical local artifacts are
`results/runs/agents/development/trajectory-report/` and
`results/runs/agents/development/aspect-judge/`.

## Interpretation and publication boundary

These exploratory results establish three facts for the current implementation:

1. the matched hybrid agent has substantially higher trajectory retrieval and
   paired C-GWAC than the filesystem agent;
2. the graph-capable arm invokes graph expansion on only four questions, and
   its C-GWAC interval against hybrid includes zero, so this run does not
   establish a graph advantage; and
3. one repair was insufficient to construct valid balanced controls on
   calibration validation, while the selected two-repair setting passed train,
   validation, and the frozen final test; this supports the silver-control
   protocol but is not evidence of human judge alignment.

They do **not** establish a publication-grade zero-shot leaderboard. The 361
questions were reused during development; accepted-answer citation qrels are
incomplete and can drift relative to the pinned corpus; aspect and answer labels
are same-model-family silver; agent calls are stochastic; and latency is a
single-machine, sequential observation. A formal result requires a new sealed
cohort plus an independent expert-human or cross-model judge audit. The ignored
v14 normalization work state, model-generated aspects, graph artifact, and run
traces must also be released as frozen publication artifacts for an independent
clean-clone replay. Derived BM25/HNSW indexes and Neo4j database files can be
rebuilt from those inputs and need not be distributed.

Superseded retrieval and agent reports use different datasets,
representations, or judges and must not be compared with these maintained
numbers.
