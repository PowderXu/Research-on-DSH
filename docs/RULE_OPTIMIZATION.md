# Frozen aspect annotations

The current benchmark uses one pool of 467 questions and the frozen
`aspects.jsonl` from the pinned dataset release. It does not create train,
validation, or test partitions and does not run an optimizer.

Questions and reference answers remain separate files. The evaluator joins
annotation records by `question_id`; agents cannot access reference answers
or scoring annotations.

Deterministic aspect schema, coverage, and evidence-binding helpers are in
`evaluation/dataset_analysis/aspect_construction.py`. The original construction
instructions are retained in
`evaluation/dataset_analysis/rubrics/aspect_construction_initial.md` for provenance.
They are not an evaluation-time optimization workflow.

## Historical preparation of the existing annotations

The released 1,926 aspects were prepared before the unpartitioned design.
That completed experiment used SkillOpt and internal 280/94/93 cohorts. Its
recorded selection and held-out results describe that historical annotation
experiment only; they do not define groups in the current dataset.

The previous executable optimizer, split loader, and configuration have been
removed. Original code remains in Git history and a local source backup.
Existing run artifacts under `results/runs/rule-optimization/` are historical
provenance; current evaluation reads the pinned dataset instead.

These are model-generated silver annotations. The earlier optimization scores
do not establish human agreement, and the already-inspected question pool is
not fresh confirmatory evidence. See [historical results](RESULTS.md).
