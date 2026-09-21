# Evaluation

```text
dataset/            build and verify the four pinned corpora and QA records
dataset_analysis/   validate evidence and construct the normalized local package
kbbench/            shared retrieval implementations and ranking metrics
tests/              local software checks (ignored by Git)
```

Dataset construction never imports a retriever, retrieval never mutates the
dataset, and every generated artifact is written under `../results/` or the
ignored `dataset/evaluation_data/` tree.

The normalized package contains one pool of 467 questions, with no train,
validation, or test partitions. `questions.jsonl` contains only question inputs;
`answers.jsonl` contains reference answers, evidence labels, and grading metadata.
Both files use `question_id` as their join key. Evaluators join them internally;
agent prompts receive the question and permitted documentation.

The dataset is published separately at https://github.com/PowderXu/docsqa-data.
Download the pinned commit from the benchmark root:

```bash
PYTHONPATH=evaluation:. evaluation/.venv/bin/python -m dataset.scripts.download_dataset
```

The lock is `dataset/templates/dataset_source.json`. Private downloads use your
existing `gh auth login`; cached packages are hash-verified without network
access. The corpus is decompressed into the ordinary evaluation package.

Install and verify from the repository root:

```bash
python3 -m venv evaluation/.venv
evaluation/.venv/bin/pip install -e './evaluation[all]'
PYTHONPATH=evaluation:. evaluation/.venv/bin/python -m dataset.scripts.verify
```

Software-check sources stay local and are ignored by Git. When present,
run the Python checks with
`evaluation/.venv/bin/python -m pytest -c evaluation/pyproject.toml` and the
TypeScript checks with `npm run --prefix dsh_plugin verify:dsh`.
These optional development checks are not required to run the benchmark.

See [the protocol](../docs/EVALUATION_PROTOCOL.md), [aspect annotations](../docs/ASPECT_ANNOTATIONS.md),
and [maintained results](../docs/RESULTS.md).
