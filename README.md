# DSH DocsQA plugin benchmark

This repository is a DSH-only foundation for building and evaluating knowledge-
base plugins over large repositories of small, linked Markdown files. It keeps
three things in scope:

1. native DeepSeek Harness plugins and arm-specific skills;
2. no-model retrieval evaluation and integrated DSH-agent evaluation;
3. dataset, plugin, evaluation, and result documentation.

The reference corpus is GitHub Docs: 3,740 canonical pages pinned at commit
`c34e3dccad00f61133c799d20e7d1208a0e6cc92`. The benchmark contains 328 real
GitHub Community questions and 421 accepted-answer page citations, split into
55 train, 27 validation, and 246 held-out test questions.

## DSH arms

| Arm | Retrieval capability | DSH skill |
|---|---|---|
| Filesystem | bounded `grep`, `glob`, and `read` over raw Markdown | `skills/fs/initial_skill.md` |
| Hybrid | BM25 + HNSW + reciprocal-rank fusion | `skills/hybrid/initial_skill.md` |
| Neo4j | hybrid seeds plus explicit, bounded Neo4j expansion | `skills/neo4j/initial_skill.md` |

The filesystem arm uses official DSH filesystem tools. The hybrid and Neo4j
arms expose native `techdocs_search`, `techdocs_fetch`, and (for Neo4j)
`techdocs_expand` tools backed by a local service. Only the retrieval plugin and
its matching skill change between arms.

## Evaluation layers

- **Plugin/retrieval evaluation:** no LLM; measures Recall, Hit, nDCG, latency,
  index construction, and graph statistics for the retrieval implementations.
- **Integrated DSH-agent evaluation:** real DSH episodes; additionally measures
  tokens, end-to-end latency, tool calls, failures, and graph-tool use.

Do not combine their latency numbers. The first isolates retrieval; the second
measures the complete agent system.

## Setup and verification

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
npm ci
npm run setup:dsh-profile
npm run verify:dsh
.venv/bin/python scripts/verify_package.py
PYTHONPATH=. .venv/bin/python -m pytest
```

Neo4j evaluation additionally needs `requirements-graph.txt` and a local Neo4j
Community instance. The normalized corpus is committed; the filesystem arm
also needs the pinned raw repository:

```bash
scripts/prepare_raw_github_docs.sh
```

## Repository structure

```text
dataset/github_docs_kb_benchmark/  frozen corpus, questions, qrels, splits, evaluator
dsh-techdocs-plugin/               native DSH tools and three retrieval skills
evaluation/harness/                matched DSH patches and answer schema
evaluation/skillopt/               frozen agent splits and per-arm configurations
kbbench/                           retrieval backends, DSH rollout adapter, scoring
scripts/                           dataset setup, validation, agent evaluation, SkillOpt
tests/                             plugin, dataset, retrieval, and agent contracts
docs/                              dataset, plugin, evaluation, and result documentation
results/                           generated-output contract; raw runs are ignored
```

Start with [docs/RESULTS.md](docs/RESULTS.md) for separate run commands and the
current results. Design details are in [docs/DATASET_DESIGN.md](docs/DATASET_DESIGN.md),
[docs/PLUGIN_DESIGN.md](docs/PLUGIN_DESIGN.md), and
[docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md).
