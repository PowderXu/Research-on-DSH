# Evaluation

```text
dataset/            build and verify the four pinned corpora and QA records
dataset_analysis/   validate evidence and construct the normalized local package
rule_optimization/  optimize one general aspect-construction rule with SkillOpt
kbbench/            shared retrieval implementations and ranking metrics
tests/              deterministic software and contract tests
```

Dataset construction never imports a retriever, retrieval never mutates the
dataset, and every generated artifact is written under `../results/` or the
ignored `dataset/evaluation_data/` tree.

The normalized package retains 467 records. Its historical physical split
(33/73/361) reproduces existing development experiments. Rule optimization has
an independent deterministic 60/20/20 split (280/94/93) generated under the
run directory. That split changes neither agent inputs nor agent behavior.

Install and verify from the repository root:

```bash
python3 -m venv evaluation/.venv
evaluation/.venv/bin/pip install -e './evaluation[all]'
evaluation/.venv/bin/pip install -e \
  'git+https://github.com/microsoft/SkillOpt.git@51d0a4d96e88558c84dee637f98e24e3fb2d1547#egg=skillopt'
PYTHONPATH=evaluation:. evaluation/.venv/bin/python -m dataset.scripts.verify
evaluation/.venv/bin/python -m pytest -c evaluation/pyproject.toml
```

The pytest command covers `evaluation/tests/` and `../dsh_plugin/tests/`.
TypeScript DSH checks run through `npm run --prefix dsh_plugin verify:dsh`.
These are software checks, not real-model benchmark runs.

The second install uses the official SkillOpt v0.2.0 source commit. Its
published wheel omits Markdown prompts required by the aggregation stage; the
wrapper detects that incomplete package before making model calls.

See [the protocol](../docs/EVALUATION_PROTOCOL.md), [rule optimization](../docs/RULE_OPTIMIZATION.md),
and [maintained results](../docs/RESULTS.md).
