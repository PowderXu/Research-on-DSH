# Evaluation commands and results

This file is the stable result index. Raw trajectories, indexes, predictions,
and generated reports are intentionally not committed; new runs write below
`results/runs/`.

## One-time setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
npm ci
npm run setup:dsh-profile
npm run verify:dsh
scripts/prepare_raw_github_docs.sh
```

For the Neo4j arm, also install graph dependencies and start Neo4j Community:

```bash
.venv/bin/pip install -r requirements-graph.txt
```

The commands below assume Neo4j is reachable at `bolt://127.0.0.1:7687` and
`NEO4J_PASSWORD` is set.

## Evaluation 1: plugin/retrieval performance

This evaluation is independent of DSH model calls. It measures ranked pages and
warm retrieval latency only.

```bash
DSH_RG_PATH=$(node -p "require('@vscode/ripgrep').rgPath")

.venv/bin/python -m kbbench.github_docs_plugin_eval \
  --dataset-dir dataset/github_docs_kb_benchmark/data \
  --repo-root data/github-docs \
  --output-dir results/runs/retrieval/dsh-three-arm \
  --cache-dir data/evaluation_cache/github_docs \
  --corpus-revision c34e3dccad00f61133c799d20e7d1208a0e6cc92 \
  --dsh-rg "$DSH_RG_PATH" \
  --split test \
  --ingest
```

Output: `results/runs/retrieval/dsh-three-arm/report.json` and
`per_query.jsonl`. `--ingest` rebuilds only this benchmark's namespaced Neo4j
labels.

### Current result

The latest completed full retrieval-foundation run used all 246 held-out
questions. It predates the final DSH three-arm wrapper, so it validates the
hybrid/graph architecture but is not a completed filesystem-vs-hybrid-vs-Neo4j
plugin comparison.

| Retrieval method | Recall@10 | Hit@10 | nDCG@10 | p50 |
|---|---:|---:|---:|---:|
| BM25 | 0.381 | 0.415 | 0.243 | 3.80 ms |
| HNSW | 0.470 | 0.524 | 0.306 | 17.33 ms |
| BM25 + HNSW RRF | 0.493 | 0.537 | 0.337 | 17.34 ms |
| Hybrid + conditional explicit-link expansion | 0.503 | 0.545 | 0.349 | 19.42 ms |
| Hybrid + always-on explicit-link expansion | 0.518 | 0.561 | 0.355 | 19.49 ms |
| Pure routing + BM25 | 0.288 | 0.321 | 0.196 | 3.83 ms |

Result boundary: hybrid retrieval clearly improved over BM25/routing in this
run. Explicit-link expansion added a small measured lift. The full current DSH
three-arm command above still needs a fresh run before publishing a plugin-arm
ranking.

## Evaluation 2: integrated DSH-agent performance

Run each arm separately with the same model and frozen test IDs. The adapter
starts the hybrid/Neo4j backend automatically; the Neo4j graph must already be
ingested by Evaluation 1.

```bash
export KBBENCH_OPENAI_ENV_FILE=/absolute/path/to/.env

.venv/bin/python scripts/skillopt_github_docs_eval.py \
  --config evaluation/skillopt/github_docs_dsh/configs/fs.yaml \
  --skill dsh-techdocs-plugin/skills/fs/initial_skill.md \
  --split test \
  --test_env_num 246 \
  --out_root results/runs/agents/fs

.venv/bin/python scripts/skillopt_github_docs_eval.py \
  --config evaluation/skillopt/github_docs_dsh/configs/hybrid.yaml \
  --skill dsh-techdocs-plugin/skills/hybrid/initial_skill.md \
  --split test \
  --test_env_num 246 \
  --out_root results/runs/agents/hybrid

.venv/bin/python scripts/skillopt_github_docs_eval.py \
  --config evaluation/skillopt/github_docs_dsh/configs/neo4j.yaml \
  --skill dsh-techdocs-plugin/skills/neo4j/initial_skill.md \
  --split test \
  --test_env_num 246 \
  --out_root results/runs/agents/neo4j
```

Aggregate the three paired rollout sets separately from model execution:

```bash
.venv/bin/python -m kbbench.github_docs_dsh_system_report \
  --fs results/runs/agents/fs \
  --hybrid results/runs/agents/hybrid \
  --neo4j results/runs/agents/neo4j \
  --out-dir results/runs/agents/report
```

Output: `results/runs/agents/report/report.md`, `report.json`, and
`per_query.jsonl`.

### Current result

The latest real-model DSH integration check used five paired, single-page
questions with `gpt-5-mini`. Primary retrieval metrics below use the agent's
final ordered sources.

| DSH arm | Recall@10 | Hit@10 | nDCG@10 | p50 end-to-end | Tokens / QA |
|---|---:|---:|---:|---:|---:|
| Filesystem | 0.600 | 0.600 | 0.377 | 35.01 s | 193,230 |
| Hybrid | 0.600 | 0.600 | 0.312 | 29.82 s | 36,793 |
| Neo4j | 0.200 | 0.200 | 0.126 | 18.41 s | 15,367 |

Do not rank the arms from this pilot. Five cases are insufficient, none is a
linked multi-page question, and the Neo4j agent made zero `techdocs_expand`
calls. The result verifies the real DSH execution paths and exposes efficiency
differences; it does not measure graph advantage.

## Validation (not a performance evaluation)

```bash
npm run verify:dsh
.venv/bin/python scripts/verify_package.py
PYTHONPATH=. .venv/bin/python -m pytest
```

These commands check manifests, tool/skill contracts, frozen data, scoring, and
rollout/report logic. Their pass count must never be reported as benchmark
performance.
