# DSH DocsQA

DocsQA is a DeepSeek Harness plugin foundation for answering questions over
large repositories of small, linked Markdown/MDX documents. The repository has
four top-level ownership areas:

```text
docs/         paper, method supplements, protocol, and current results
dsh_plugin/   DSH tools, skills, backend, harness profile, and agent runner
evaluation/   dataset construction, label quality, retrieval, and tests
results/      ignored local runs and caches
```

## Current pipeline

```text
public documentation repositories + real support QAs
  -> evaluation/dataset/             build immutable local corpora and qrels
  -> evaluation/dataset_analysis/    validate, normalize, build aspects, check weak supervision
  -> evaluation/kbbench/             shared retrieval and scoring implementation
  -> dsh_plugin/agent_eval/          run the actual DSH harness and grade answers
```

The combined pre-normalization source package contains GitHub Docs, Tailwind
CSS, Prisma, and Supabase. It has 5,392 source-document records and 556
structurally eligible questions. Normalization removes empty navigation/source
rows; the evaluated corpus shared by all retrieval arms therefore has 4,860
searchable documents and 467 questions. Images are converted once to local text
evidence and are not searched as image vectors.

The dataset keeps its historical physical split of 33 construction, 73
validation, and 361 legacy-`test` questions so earlier retrieval and agent
artifacts remain reproducible. Judge development uses a separate,
evaluator-only 60/20/20 overlay across all 467 records: 280 calibration-train,
93 calibration-validation, and 94 final-test questions. This overlay changes
neither the agent's inputs nor its behavior; it prevents judge-rule selection
from using the final evaluator check. It is a prospective evaluator holdout,
not a claim that these records were absent from earlier dataset construction.
The selected `gpt-5.6-luna` evaluator-validation configuration uses
`--max-variant-repairs 2`; it passed calibration train and validation, then
passed every declared gate on the frozen final test (macro-F1 0.9496, balanced
case rate 0.9894). This validates the silver judge protocol only. Retrieval and
agent results on the reused 361-question physical `test` partition remain
exploratory.

Generated artifacts never live beside source code. All runs use
`results/runs/`, all caches use `results/cache/`, and only the consolidated
tables in [docs/RESULTS.md](docs/RESULTS.md) are maintained as current reports.

## DSH retrieval candidates

| Arm | Capability |
|---|---|
| Filesystem | Bounded DSH filesystem search and read |
| Hybrid | BM25 + HNSW with reciprocal-rank fusion |
| Neo4j | Hybrid seeds plus bounded typed graph expansion |

The three arms expose the same DocsQA capability and differ only in their
retrieval tools and matching authored skill. Runtime corpora, indexes, graph
files, and traces stay under ignored `dsh_plugin/plugin/data/` or `results/`.
`evaluation/kbbench/` supplies the retrievers and ranking metrics used by the
backend and trajectory scorer. The reported system benchmark is
`dsh_plugin/agent_eval/`, which measures installed skills, plugin/tool use,
multi-turn behavior, final answers, tokens, and end-to-end latency.

## Setup and verification

```bash
python3 -m venv evaluation/.venv
evaluation/.venv/bin/pip install -e "./evaluation[retrieval,graph,judge,test]"
npm ci --prefix dsh_plugin
(cd dsh_plugin && node -p "require('@vscode/ripgrep').rgPath")
npm run --prefix dsh_plugin setup:dsh-profile
npm run --prefix dsh_plugin verify:dsh
evaluation/.venv/bin/python -m pytest -c evaluation/pyproject.toml
```

Python correctness tests live in `evaluation/tests/{dataset,dataset_analysis,kbbench}/`
and `dsh_plugin/tests/{backend,agent_eval}/`. TypeScript plugin tests remain in
`dsh_plugin/plugin/test/`. The pytest command runs all Python tests;
`verify:dsh` runs the TypeScript checks. These do not rerun the paper's
real-model experiments.

To review tracked implementation changes without test files:

```bash
git diff HEAD -- evaluation dsh_plugin \
  ':(exclude,glob)**/test_*.py' \
  ':(exclude,glob)**/*.test.ts'
```

Git diff omits untracked files. List new files separately with
`git ls-files --others --exclude-standard`.

Prepare or refresh the four pinned documentation checkouts with:

```bash
evaluation/dataset/scripts/prepare_raw.sh
```

Build the source package before running its verifier; then materialize or build
the normalized package as described in [Current results](docs/RESULTS.md):

```bash
PYTHONPATH=evaluation:. evaluation/.venv/bin/python -m dataset.scripts.build_all
PYTHONPATH=evaluation:. evaluation/.venv/bin/python -m dataset.scripts.verify
```

The MiniLM embedding model must be downloaded once before a run that uses
`--local-files-only`. Generated corpora, model caches, graph artifacts, and
per-query outputs are intentionally not committed. Consequently, a clean clone
can run the construction code, but cannot reproduce the exact maintained
numbers without the corresponding frozen release artifacts. The present local
workspace retains those artifacts; a publication release must package their
derived, license-compatible forms and hashes.

## Documentation

- [Dataset design](docs/DATASET_DESIGN.md)
- [Normalized dataset](docs/NORMALIZED_DATASET.md)
- [Plugin design](docs/PLUGIN_DESIGN.md)
- [Evaluation protocol](docs/EVALUATION_PROTOCOL.md)
- [Aspect evaluation and weak supervision](docs/ASPECT_EVALUATION.md)
- [Current results](docs/RESULTS.md)
- [Workshop/formal-paper draft](docs/PAPER_DRAFT.md)
