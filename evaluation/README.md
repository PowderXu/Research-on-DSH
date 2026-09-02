# Evaluation

The evaluation package has four explicit responsibilities:

```text
dataset/           build and verify immutable corpora and QA records
dataset_analysis/  validate evidence, normalize answers, and test judge weak supervision
kbbench/           implement shared retrievers and trajectory-ranking metrics
tests/             automated correctness and contract tests, grouped by component
```

Python tests are separate from implementation code:

```text
tests/
├── dataset/           corpus construction, source extraction, and data contracts
├── dataset_analysis/  normalization, aspects, splits, and judge controls
└── kbbench/           retrieval and scoring
```

DSH Python tests live in `../dsh_plugin/tests/{backend,agent_eval}/`.
The pytest configuration discovers only these two test roots. The benchmark
runners in `dataset_analysis/`, `kbbench/`, and `../dsh_plugin/agent_eval/`
remain implementation code; they are not part of the software-test directories.

Ownership is one-way: dataset construction never imports a retriever, retrieval
never mutates the dataset, and generated work is written only under
`../results/`.

The default source package is
`dataset/evaluation_data/combined/{corpus,questions,splits}`. The stricter
answer-generation package is `dataset/evaluation_data/normalized/`. See
[the normalized-data contract](../docs/NORMALIZED_DATASET.md) and
[the evaluation protocol](../docs/EVALUATION_PROTOCOL.md).

The normalized package retains the historical physical 33/73/361 split used by
existing retrieval and agent artifacts. Judge weak supervision instead uses
`dataset/evaluation_data/normalized/weak_supervision_split.json`, a deterministic
project-stratified overlay across all 467 questions: 280
`calibration_train`, 93 `calibration_validation`, and 94 `final_test`. This is
an evaluator-development split only. It does not partition agent execution or
change the questions and tools visible to an evaluated system.

The completed live selection used two iterations. With
`--max-variant-repairs 1`, train passed and validation failed only balanced-case
coverage (0.9462 < 0.98). With `--max-variant-repairs 2`, train and validation
passed; that setting was frozen and passed all gates on final test (macro-F1
0.9496, balanced-case rate 0.9894). The selected artifacts are
`../results/runs/dataset-analysis/aspect-calibration-v2-{train-r2,validation-r2,final-test}/`.
This evaluator result does not change the exploratory status of retrieval or
agent measurements on the reused 361-question physical `test` partition.

Install and verify from the repository root:

```bash
python3 -m venv evaluation/.venv
evaluation/.venv/bin/pip install -e "./evaluation[retrieval,judge,test]"
PYTHONPATH=evaluation evaluation/.venv/bin/python -m dataset.scripts.verify
evaluation/.venv/bin/python -m pytest -c evaluation/pyproject.toml
```

The pytest command above runs both evaluation and DSH Python tests. To run only
evaluation tests, append `evaluation/tests`; to run only one component, append
its directory, such as `evaluation/tests/kbbench`. These are software checks,
not new real-model benchmark runs.

Current consolidated numbers are maintained only in
[`../docs/RESULTS.md`](../docs/RESULTS.md).

`kbbench/` is shared infrastructure, not the reported agent benchmark. Actual
integrated DSH execution and answer evaluation live under
`../dsh_plugin/agent_eval/`.
