# DocsQA-Repo

DocsQA-Repo is a dataset and benchmark for agents that answer real technical
support questions from large repositories of small, linked Markdown/MDX files.
It evaluates both the retrieval trajectory and the final grounded answer. The
DSH plugins in this repository are baseline systems, not the research
contribution.

```text
docs/         paper, benchmark design, protocol, and maintained result tables
evaluation/   dataset construction, normalization, rule optimization, retrieval scoring, tests
dsh_plugin/   three DeepSeek Harness baseline configurations
results/      ignored local runs, model outputs, indexes, and caches
```

## Dataset

The source collection combines GitHub Docs, Prisma, Supabase, and Tailwind CSS:

- 798 answered public support discussions in the fixed candidate manifest;
- 556 structurally eligible question/answer packages;
- 467 normalized local-evidence records retained for evaluation;
- 4,860 searchable pages from pinned documentation repositories.

Each normalized record contains the real question, a standalone answer derived
from the accepted answer and its internal documentation, local document IDs and
paths, normalized claims, and image text where required. Live web pages and
image-vector retrieval are excluded from evaluation.

The historical physical split is kept for reproducibility: 33 construction, 73
validation, and 361 development/evaluation records. It is not a sealed system
test because it was used during development.

## Benchmark

Trajectory retrieval reports Recall@10, Hit@10, nDCG@10, p50 latency, model
tokens, and validity failures. Final-answer evaluation uses question-specific
weighted aspects and reports Weighted Aspect Coverage (WAC), following the
[BRIGHT-Pro paper](https://arxiv.org/abs/2605.04018) and its
[official evaluator](https://github.com/yale-nlp/Bright-Pro/blob/main/agentic_retrieval/scripts_evaluation/judge.py),
plus unsupported-claim and citation diagnostics.

The general aspect-construction rule is optimized separately from all evaluated
agents. The 467 normalized records are divided into an exact deterministic
60/20/20 optimization split: 280 train, 94 validation, and 93 held-out test.
`gpt-5.6-luna` applies the shared rule to construct per-question aspects;
`gpt-5.6-sol` proposes bounded rule edits through SkillOpt 0.2.0. Validation
selects the rule, and test is evaluated only after the rule is frozen. This
optimizes textual instructions, not model parameters or retrieval plugins.

## DSH baselines

| Arm | Retrieval capability |
|---|---|
| Filesystem | LLM-guided search and reads over the local Markdown/MDX tree |
| Hybrid | BM25 and HNSW rankings fused with reciprocal-rank fusion |
| Neo4j-capable | Hybrid seeds plus optional bounded typed graph expansion |

All three use the same corpus, language model, non-retrieval configuration,
answer contract, and scoring code. Generated plugin data remains under ignored
`dsh_plugin/plugin/data/` and `results/` paths.

## Setup and verification

```bash
python3 -m venv evaluation/.venv
evaluation/.venv/bin/pip install -e './evaluation[all]'
evaluation/.venv/bin/pip install -e \
  'git+https://github.com/microsoft/SkillOpt.git@51d0a4d96e88558c84dee637f98e24e3fb2d1547#egg=skillopt'
npm ci --prefix dsh_plugin
npm run --prefix dsh_plugin setup:dsh-profile
npm run --prefix dsh_plugin verify:dsh
evaluation/.venv/bin/python -m pytest -c evaluation/pyproject.toml
```

Prepare the pinned repositories and build the dataset:

```bash
evaluation/dataset/scripts/prepare_raw.sh
PYTHONPATH=evaluation:. evaluation/.venv/bin/python -m dataset.scripts.build_all
PYTHONPATH=evaluation:. evaluation/.venv/bin/python -m dataset.scripts.verify
```

The exact normalization, retrieval, agent, and rule-optimization commands are
kept in the component documents below. Generated corpora and model outputs are
not committed; the repository contains code, contracts, and consolidated
tables only.

## Documentation

- [Dataset design](docs/DATASET_DESIGN.md)
- [Normalized dataset](docs/NORMALIZED_DATASET.md)
- [Benchmark and evaluation protocol](docs/EVALUATION_PROTOCOL.md)
- [Aspect-rule optimization](docs/RULE_OPTIMIZATION.md)
- [DSH baseline design](docs/PLUGIN_DESIGN.md)
- [Current results](docs/RESULTS.md)
- [Paper draft (Markdown)](docs/PAPER_DRAFT.md)
- [Paper draft (LaTeX)](docs/PAPER_DRAFT.tex)
