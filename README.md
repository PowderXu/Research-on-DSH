# DSH DocsQA plugin benchmark

This is a strictly DeepSeek Harness foundation for building and evaluating
knowledge-base plugins over large repositories of small, linked Markdown files.
The repository has four top-level areas:

```text
docs/         dataset, plugin, evaluation, and result documentation
results/      generated-output contract; run contents are ignored
evaluation/   dataset construction plus KB/retrieval evaluation
dsh_plugin/   DSH package, skills, profile, backend, and agent evaluation
```

The reference dataset contains 3,740 canonical GitHub Docs pages pinned at
commit `c34e3dccad00f61133c799d20e7d1208a0e6cc92`, 328 real GitHub Community
questions, and 421 accepted-answer page citations. The fixed split is 55 train,
27 validation, and 246 held-out test questions.

## Three DSH arms

| Arm | Retrieval capability | Authored skill |
|---|---|---|
| Filesystem | bounded DSH filesystem search and read | `skills/fs/initial_skill.md` |
| Hybrid | BM25 + HNSW + reciprocal-rank fusion | `skills/hybrid/initial_skill.md` |
| Neo4j | hybrid seeds plus bounded typed graph expansion | `skills/neo4j/initial_skill.md` |

The filesystem arm uses official DSH filesystem tools. Hybrid and Neo4j expose
native `techdocs_search`, `techdocs_fetch`, and optional `techdocs_expand` tools
through a local backend. The three authored skill files are versioned directly;
there is no SkillOpt training or optimization layer.

## Evaluation layers

- Plugin/retrieval evaluation makes no model calls and measures Recall, Hit,
  nDCG, warm latency, build time, and graph statistics.
- Integrated DSH-agent evaluation runs real DSH episodes and additionally
  measures model tokens, end-to-end latency, tool calls, failures, and graph use.

Do not combine the two latency scopes. See [docs/RESULTS.md](docs/RESULTS.md) for
separate run commands and current results.

## Setup and verification

```bash
python3 -m venv evaluation/.venv
evaluation/.venv/bin/pip install -e "./evaluation[retrieval,test]"
npm ci --prefix dsh_plugin
npm run --prefix dsh_plugin setup:dsh-profile
npm run --prefix dsh_plugin verify:dsh
evaluation/.venv/bin/python evaluation/dataset/verify.py
evaluation/.venv/bin/python -m pytest -c evaluation/pyproject.toml
```

Neo4j evaluation additionally needs `evaluation/requirements-graph.txt` and a
local Neo4j Community instance. Filesystem and agent evaluation need the pinned
raw docs checkout:

```bash
evaluation/dataset/prepare_raw.sh
```

Design details are in [docs/DATASET_DESIGN.md](docs/DATASET_DESIGN.md),
[docs/PLUGIN_DESIGN.md](docs/PLUGIN_DESIGN.md), and
[docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md).
