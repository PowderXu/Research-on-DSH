# Dataset and answer-quality analysis

This package turns structurally valid support discussions into the local
DocsQA-Repo benchmark and validates the frozen answer judge under the paper's
weak-supervision assumption. It contains no retrieval implementation and writes
generated artifacts only under `results/` or
`evaluation/dataset/evaluation_data/`.

```text
validate_sources.py               resolve local documentation and linked-QA evidence
materialize_images.py             reproduce images and record provenance
image_evidence.py                 build conservative image-to-text evidence
normalize_dataset.py              construct standalone local QA records
build_aspects.py                  construct and review question-specific aspects
merge_aspect_runs.py              validate and merge the 467 aspect records
build_weak_supervision_splits.py  create the 60/20/20 evaluator overlay
question_partitions.py            validate evaluator partition membership
calibrate_aspect_judge.py         construct controls and apply acceptance gates
describe_benchmark.py             reproduce corpus and evidence statistics
audit_integrity.py                check duplicate and split-overlap diagnostics
llm_runtime.py                    shared credential, usage, and percentile helpers
rubrics/                          hash-frozen domain-neutral rules
test_*.py                         deterministic contracts and regression tests
```

The weak-supervision assumption is that most platform-selected answers, together
with their resolved internal documentation, are sufficiently correct and
complete to resolve the original question. These are noisy positive references,
not human gold labels.

## Reproduce the paper artifacts

Run commands from the repository root with `PYTHONPATH=evaluation:.`.

The reported dataset is v14. Its exact generation prompt is not a versioned
input in the current tree, so reproduce it from the retained frozen work state;
this makes no model call:

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

Rebuild the deterministic evaluator overlay:

```bash
PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m dataset_analysis.build_weak_supervision_splits \
  --questions evaluation/dataset/evaluation_data/normalized/questions.jsonl \
  --output evaluation/dataset/evaluation_data/normalized/weak_supervision_split.json \
  --seed 20260901
```

Validate the frozen final-test report without an API call:

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

`--report-only` requires the retained calibration cases, per-candidate labels,
and judge-call metadata. It verifies their hashes and recomputes the report; it
does not call a model. A mismatch requires a new output directory.

The current `rubrics/normalization_v1.json` is the later v15 clean-generation
rubric. Running it is a new dataset-construction experiment and must use new
work and output directories; it is not an exact reproduction of the paper's
v14 package.

Full commands, contracts, and maintained numbers are in:

- [`../../docs/NORMALIZED_DATASET.md`](../../docs/NORMALIZED_DATASET.md)
- [`../../docs/ASPECT_EVALUATION.md`](../../docs/ASPECT_EVALUATION.md)
- [`../../docs/RESULTS.md`](../../docs/RESULTS.md)
