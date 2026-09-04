# Dataset and answer normalization

This package converts structurally valid support discussions into a local,
standalone QA package. It does not implement retrieval or optimize the
answer-evaluation rule.

```text
validate_sources.py   resolve internal documentation and linked-QA evidence
materialize_images.py reproduce images and record their provenance
image_evidence.py     convert reproducible pixels to conservative local text
normalize_dataset.py  build standalone answers, claims, requirements, and evidence IDs
describe_benchmark.py reproduce corpus and evidence statistics
audit_integrity.py    check duplicate and split-overlap diagnostics
llm_runtime.py        shared credential, usage, and percentile helpers
rubrics/              versioned normalization and final-answer judging rules
```

The paper's v14 normalized package is materialized from retained local work
state without a model call:

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

This produces 467 accepted local-evidence records from 556 structurally
eligible sources. The normalization score and `>0.90` gate belong only to
dataset construction; they are not the final-answer metric and are not used by
the aspect-rule optimizer.

The current clean-generation rubric is `rubrics/normalization_v1.json`.
Running it is a new dataset-construction experiment and must use a new work
directory. General aspect-rule optimization is documented separately in
[`../rule_optimization/README.md`](../rule_optimization/README.md).
