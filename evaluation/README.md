# Evaluation

This area has one-way ownership:

```text
dataset/  builds and verifies the immutable dataset
kbbench/  evaluates ranked retrieval against that built dataset
tests/    validates both contracts
```

`dataset/` never imports a retriever. `kbbench/` consumes
`dataset/data/{corpus,questions,splits}` and writes only to caller-selected
output/cache paths under `results/`.

Install the evaluation environment from the repository root:

```bash
python3 -m venv evaluation/.venv
evaluation/.venv/bin/pip install -e "./evaluation[retrieval,test]"
```

Then verify the frozen data and evaluation code independently:

```bash
evaluation/.venv/bin/python evaluation/dataset/verify.py
evaluation/.venv/bin/python -m pytest -c evaluation/pyproject.toml evaluation/tests
```
