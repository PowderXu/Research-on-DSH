# Evaluation commands and results

This is the stable result index. Raw trajectories, indexes, predictions, and
generated reports are intentionally not committed; runs write below
`results/runs/`.

## One-time setup

```bash
python3 -m venv evaluation/.venv
evaluation/.venv/bin/pip install -e "./evaluation[retrieval,test]"
npm ci --prefix dsh_plugin
npm run --prefix dsh_plugin setup:dsh-profile
npm run --prefix dsh_plugin verify:dsh
evaluation/dataset/prepare_raw.sh
```

For Neo4j, also install the graph extras and start Neo4j Community:

```bash
evaluation/.venv/bin/pip install -e "./evaluation[all]"
```

The commands below assume Neo4j is reachable at `bolt://127.0.0.1:7687` and
`NEO4J_PASSWORD` is set.

## Evaluation 1: plugin/retrieval performance

This evaluation makes no LLM calls. It measures ranked pages and warm retrieval
latency using the already-built dataset.

```bash
DSH_RG_PATH=$(node -p "require('./dsh_plugin/node_modules/@vscode/ripgrep').rgPath")

evaluation/.venv/bin/python -m kbbench.plugin_eval \
  --dataset-dir evaluation/dataset/data \
  --repo-root evaluation/dataset/raw/github-docs \
  --output-dir results/runs/retrieval/dsh-three-arm \
  --cache-dir results/cache/github_docs \
  --corpus-revision c34e3dccad00f61133c799d20e7d1208a0e6cc92 \
  --dsh-rg "$DSH_RG_PATH" \
  --split test \
  --ingest
```

Outputs: `report.json` and `per_query.jsonl` in the selected output directory.
`--ingest` rebuilds only this benchmark's namespaced Neo4j labels.

### Current result

The latest completed 246-question retrieval-foundation run predates the final
three-arm wrapper, so it validates the hybrid/graph architecture but is not a
completed filesystem-vs-hybrid-vs-Neo4j plugin ranking.

| Retrieval method | Recall@10 | Hit@10 | nDCG@10 | p50 |
|---|---:|---:|---:|---:|
| BM25 | 0.381 | 0.415 | 0.243 | 3.80 ms |
| HNSW | 0.470 | 0.524 | 0.306 | 17.33 ms |
| BM25 + HNSW RRF | 0.493 | 0.537 | 0.337 | 17.34 ms |
| Hybrid + conditional explicit-link expansion | 0.503 | 0.545 | 0.349 | 19.42 ms |
| Hybrid + always-on explicit-link expansion | 0.518 | 0.561 | 0.355 | 19.49 ms |
| Pure routing + BM25 | 0.288 | 0.321 | 0.196 | 3.83 ms |

## Evaluation 2: integrated DSH-agent performance

Run each arm with the same model and identical fixed test IDs. This direct runner
uses the authored skill as-is; it performs no skill optimization.

```bash
export KBBENCH_OPENAI_ENV_FILE=/absolute/path/to/.env

for arm in fs hybrid neo4j; do
  PYTHONPATH=evaluation:. evaluation/.venv/bin/python -m dsh_plugin.agent_eval.runner \
    --arm "$arm" \
    --split test \
    --output-dir "results/runs/agents/$arm"
done
```

Aggregate the paired rollouts without repeating model calls:

```bash
PYTHONPATH=evaluation:. evaluation/.venv/bin/python -m dsh_plugin.agent_eval.report \
  --fs results/runs/agents/fs \
  --hybrid results/runs/agents/hybrid \
  --neo4j results/runs/agents/neo4j \
  --out-dir results/runs/agents/report
```

Outputs: `report.md`, `report.json`, and `per_query.jsonl`.

### Current result

The latest real-model integration check used five paired, single-page questions
with `gpt-5-mini`. Primary retrieval metrics use the agent's final ordered
sources.

| DSH arm | Recall@10 | Hit@10 | nDCG@10 | p50 end-to-end | Tokens / QA |
|---|---:|---:|---:|---:|---:|
| Filesystem | 0.600 | 0.600 | 0.377 | 35.01 s | 193,230 |
| Hybrid | 0.600 | 0.600 | 0.312 | 29.82 s | 36,793 |
| Neo4j | 0.200 | 0.200 | 0.126 | 18.41 s | 15,367 |

Do not rank the arms from this pilot. Five cases are insufficient, none is a
linked multi-page question, and the Neo4j agent made no `techdocs_expand` call.

## Validation, not benchmark performance

```bash
npm run --prefix dsh_plugin verify:dsh
evaluation/.venv/bin/python evaluation/dataset/verify.py
evaluation/.venv/bin/python -m pytest -c evaluation/pyproject.toml
```

These commands check manifests, contracts, frozen data, scoring, and direct DSH
rollout/report logic. Passing tests must not be reported as benchmark accuracy.
