# Question-specific aspect evaluation and benchmark weak supervision

This evaluation adapts the fine-grained idea behind BRIGHT-Pro to this local
technical-documentation benchmark. It does not grade an answer with one static
`complete/partial/incorrect` prompt. Each question first receives a compact,
weighted set of answer aspects derived from its normalized question, reference
answer, claims, and exact local evidence.

Dataset normalization is an upstream, separate construction step. It creates a
standalone local question, answer, and evidence package and applies its own
quality gate. Its score is neither a weak-supervision result nor evidence that
the source answer is independently human-verified.

The implementation has four separate stages:

1. **Aspect construction.** A model proposes independently scorable answer
   requirements. Every aspect must map to an existing user requirement,
   normalized claim, and exact evidence identifier.
2. **Independent model aspect review.** A second model call may split, merge,
   remove, or add aspects. Deterministic validation rejects unknown mappings,
   duplicate descriptions, and schemas that fail to cover critical user
   requirements.
3. **Weak-supervision case construction.** The normalized platform-selected
   answer and resolved local documentation form a noisy positive reference.
   The pipeline also creates one useful-but-partial answer and one answer with
   a decisive error. A separate model reviewer assigns labels and rejects
   ambiguous or unbalanced triples.
4. **Partitioned judge development.** General judge rules are revised on
   calibration train, selected and frozen on calibration validation, and
   checked once on final test. One judge scores every candidate against the
   same aspects and evidence. The deterministic Python scorer computes the
   aggregate gates; the model does not provide the aggregate result.

The shared rubric is
[`../evaluation/dataset_analysis/rubrics/aspect_evaluation_v2.json`](../evaluation/dataset_analysis/rubrics/aspect_evaluation_v2.json).
It contains only domain-neutral semantic rules. The loaders reject case,
question, project, and dataset override fields. Each output records the rubric
version and SHA-256, and a changed rubric must use a new output directory.

## Metrics

The primary docs-only answer-quality score is **Corpus-Conditioned Grounded
Weighted Aspect Coverage (C-GWAC)**:

```text
C-GWAC = sum(document-supported aspect weight × grounded coverage)
         / sum(document-supported aspect weight)
coverage ∈ {0, 0.5, 1}
```

A question is C-GWAC-scorable only when at least one critical aspect has pinned
local-document support. Report all-aspect GWAC alongside it as a diagnostic: it
also includes accepted-answer- and question-only aspects, so its gap from
C-GWAC is not purely a retrieval or agent error. Report both with
critical-aspect success, complete/partial/incorrect rates, material
unsupported-claim rate, citation-integrity rate, tokens per QA, and end-to-end
p50 latency. Agent failures remain in the applicable denominator with zero
coverage and an `incorrect` outcome.

For retrieval, **Weighted Aspect Recall@10** measures the fraction of
retrieval-eligible answer weight covered by the top ten documents.
**alpha-nDCG@10** additionally rewards complementary evidence before redundant
documents for an already covered aspect. Aspects supported only by answer or
question text remain part of answer grading but are excluded from retrieval
denominators.

Legacy qrel Recall, Hit, and nDCG remain useful for comparison and are still
reported. They answer a different question: whether a system found the pages
linked by the historical accepted answer.

## Build model-reviewed silver aspects

Judge development uses a deterministic project-stratified overlay that is
independent of the dataset's historical physical split:

| Project | Calibration train | Calibration validation | Final test |
|---|---:|---:|---:|
| GitHub Docs | 118 | 39 | 40 |
| Prisma | 75 | 25 | 25 |
| Supabase | 31 | 10 | 11 |
| Tailwind CSS | 56 | 19 | 18 |
| **Total** | **280** | **93** | **94** |

The overlay partitions only evaluator development. It does not partition or
otherwise alter agent execution. It is held out prospectively from the current
judge-rule optimization, not from earlier dataset construction or normalization.
Generate the deterministic manifest from the normalized questions:

```bash
PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m dataset_analysis.build_weak_supervision_splits \
  --questions evaluation/dataset/evaluation_data/normalized/questions.jsonl \
  --output evaluation/dataset/evaluation_data/normalized/weak_supervision_split.json \
  --seed 20260901
```

The current all-467 artifact reuses the already completed 73 validation and 361
development records, constructs the missing 33 physical-train records, and
merges the three files under one exact manifest-coverage check:

Frozen paper directories are evidence, not resume targets. The commands below
write all new model output under a distinct `current-source` root.

From the repository root:

```bash
ANALYSIS_ROOT=results/runs/dataset-analysis/current-source

KBBENCH_OPENAI_ENV_FILE=/absolute/path/to/private.env \
PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m dataset_analysis.build_aspects \
  --normalization-work results/runs/dataset-analysis/normalization-v14/per_question.jsonl \
  --output-dir "$ANALYSIS_ROOT/aspects-train" \
  --split train \
  --model gpt-5.6-luna \
  --reasoning-effort medium \
  --workers 8 \
  --resume

PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m dataset_analysis.merge_aspect_runs \
  --input "$ANALYSIS_ROOT/aspects-train/aspects.jsonl" \
  --input results/runs/dataset-analysis/aspects-v2-validation/aspects.jsonl \
  --input results/runs/dataset-analysis/aspects-v2-test/aspects.jsonl \
  --output-dir "$ANALYSIS_ROOT/aspects-all" \
  --question-manifest evaluation/dataset/evaluation_data/normalized/weak_supervision_split.json
```

The historical physical 33/73/361 split is retained for provenance and earlier
retrieval/agent artifacts. It is not the judge-development split. The merged
artifact contains exactly 467 accepted records, has SHA-256
`a15bfdb34667721f23adb1f6f9864867a586bfefff861d4a99eac60622341b36`,
and uses rubric SHA-256
`840d17a4552044b780ed146b5916eb6abad76e06820a07a35b9582e3edadde34`.

## Run the benchmark weak-supervision check

The benchmark assumption is:

> For most retained questions, the platform-selected answer combined with its
> resolved internal documentation is sufficiently complete and correct to
> resolve the question.

This is a source-selection assumption, not a human gold label. The source
reference is therefore called a **noisy positive**. Partial and decisively
incorrect controlled answers are required so that an always-`complete` judge
cannot pass merely by agreeing with the assumption.

Use the same declared gates in every run:

```bash
ANALYSIS_ROOT=results/runs/dataset-analysis/current-source

CALIBRATE=(
  evaluation/.venv/bin/python -m dataset_analysis.calibrate_aspect_judge
  --aspects "$ANALYSIS_ROOT/aspects-all/aspects.jsonl"
  --model gpt-5.6-luna \
  --reasoning-effort medium \
  --workers 12 \
  --question-manifest evaluation/dataset/evaluation_data/normalized/weak_supervision_split.json \
  --min-source-answer-complete-recall 0.80 \
  --min-partial-recall 0.80 \
  --min-incorrect-recall 0.80 \
  --max-incorrect-false-complete-rate 0.05 \
  --min-macro-f1 0.80 \
  --min-balanced-case-rate 0.98
)

# Iteration 1: the default one repair.
KBBENCH_OPENAI_ENV_FILE=/absolute/path/to/private.env \
PYTHONPATH=evaluation:. "${CALIBRATE[@]}" \
  --manifest-partition calibration_train \
  --rubric-role candidate \
  --max-variant-repairs 1 \
  --output-dir "$ANALYSIS_ROOT/calibration-train-r1" \
  --resume

KBBENCH_OPENAI_ENV_FILE=/absolute/path/to/private.env \
PYTHONPATH=evaluation:. "${CALIBRATE[@]}" \
  --manifest-partition calibration_validation \
  --rubric-role candidate \
  --max-variant-repairs 1 \
  --output-dir "$ANALYSIS_ROOT/calibration-validation-r1" \
  --allow-gate-failure \
  --resume

# Iteration 2: keep accepted triples and retry rejected triples twice.
KBBENCH_OPENAI_ENV_FILE=/absolute/path/to/private.env \
PYTHONPATH=evaluation:. "${CALIBRATE[@]}" \
  --manifest-partition calibration_train \
  --rubric-role candidate \
  --max-variant-repairs 2 \
  --seed-cases-from "$ANALYSIS_ROOT/calibration-train-r1" \
  --output-dir "$ANALYSIS_ROOT/calibration-train-r2" \
  --resume

KBBENCH_OPENAI_ENV_FILE=/absolute/path/to/private.env \
PYTHONPATH=evaluation:. "${CALIBRATE[@]}" \
  --manifest-partition calibration_validation \
  --rubric-role candidate \
  --max-variant-repairs 2 \
  --seed-cases-from "$ANALYSIS_ROOT/calibration-validation-r1" \
  --output-dir "$ANALYSIS_ROOT/calibration-validation-r2" \
  --resume

# Freeze the selected setting and open final test once.
KBBENCH_OPENAI_ENV_FILE=/absolute/path/to/private.env \
PYTHONPATH=evaluation:. "${CALIBRATE[@]}" \
  --manifest-partition final_test \
  --rubric-role frozen \
  --max-variant-repairs 2 \
  --output-dir "$ANALYSIS_ROOT/calibration-final-test" \
  --resume
```

Iteration 1 passed calibration train but failed calibration validation only on
balanced-triple coverage: 88/93 = 0.9462, below the declared 0.98 gate. All six
other gates passed. Iteration 2 changed only the general
`--max-variant-repairs` allowance from 1 to 2; the rubric and its hash remained
unchanged. It passed train (279/280 = 0.9964 valid triples) and validation
(92/93 = 0.9892), so this exact setting was selected. The frozen final run used
fresh cases, passed with 93/94 = 0.9894 valid triples and macro-F1 0.9496, and
was not followed by another evaluator edit. If the judge changes after this
run, a new final cohort is required.

The default gate requires:

- at least 80% of noisy-positive references to remain `complete` under the
  independent case reviewer;
- at least 80% recall when the frozen judge scores the noisy-positive answers;
- at least 80% recall on both partial and incorrect controls;
- at least 0.80 macro F1;
- at most 5% of incorrect answers mislabeled `complete`; and
- valid complete/partial/incorrect triples for at least 98% of source records.

These thresholds are CLI options, not case-specific rules. Passing establishes
compatibility with the noisy-positive assumption and discrimination on the
controlled silver contrasts. It does not establish that the source answers are
human-verified truths or that the judge is human-aligned.

If a training gate fails, inspect aggregate confusion and reviewer explanations
on calibration train. Change only a general semantic rule that applies across
projects, increment the rubric version, create a new output directory, and
rerun. Use calibration validation for selection and stopping, not further
case-by-case editing. Freeze once, then run final test.
Never add a question ID, product name, expected label, local path, or case
override to the rubric.

The generated labels are model-reviewed silver labels. They do not become human
expert labels merely because the pipeline iterates.

### Rebuild a report without model calls

Once `calibration_cases.jsonl`, `per_candidate.jsonl`, and `judge_calls.jsonl`
exist for a partition, validate their exact question/candidate coverage and
rebuild `report.json`, `REPORT.md`, and `run_contract.json` locally:

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

`--report-only` makes no API call. A normal `--resume` run enforces the frozen
run contract: aspect, manifest, and selected-question hashes; partition; sample
controls; seed; rubric version and hash; model; reasoning effort; case
provenance; and all thresholds must match. A mismatch requires a new output
directory. For historical artifacts without `run_contract.json`, use their
legacy command and label them historical; they cannot be adopted as the new
94-question final test.

## DSH-agent answer evaluation

First produce matched real DSH rollouts for the same question IDs. Then run:

```bash
RUN_ROOT=results/runs/agents/current-source

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

The output contains `report.json`, per-query judgments, and call metadata. It
stores prompt hashes and token usage but never stores an API key.

## Current status

The all-467 aspect artifact is complete. The selected iteration uses rubric v2,
`gpt-5.6-luna`, medium reasoning, and `--max-variant-repairs 2`; it passed both
calibration partitions. The exact frozen setting then passed all predeclared
gates on the one-time 94-question final test. Maintained metric tables are
reported only in [`RESULTS.md`](RESULTS.md); this document defines the method
and commands so result tables cannot drift between files. These are
model-reviewed silver results, not independent human verification.
