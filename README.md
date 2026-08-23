# GitHub Docs knowledge-base benchmark

This repository is now the standalone benchmark for a knowledge base built from
many small, linked Markdown files. It packages the GitHub Docs corpus, 328 real
support questions, four agent configurations, retrieval/agent evaluators,
external retrieval skills, a PyPI-backed SkillOpt adapter, and reproducible
evaluation entry points.

## Benchmark contract

- Corpus: 3,740 canonical `github/docs` pages at commit
  `c34e3dccad00f61133c799d20e7d1208a0e6cc92`.
- Questions: 328 public GitHub Community questions with accepted-answer Docs
  links, yielding 421 canonical page qrels.
- Splits: 55 train, 27 validation, 246 untouched test.
- Primary metrics: Recall@10, Hit@10, nDCG@10, and p50 latency.
- Primary agent ranking: the final ordered `sources` returned by the agent.
  First-visible tool/backend documents are diagnostics only.
- Factual slices: intent, evidence category, explicit-link evidence structure,
  and qrel count. The heuristic `graph_opportunity` label is excluded.

The qrels are sparse accepted-answer citations. A retrieved page without a qrel
has zero measured gain, but is unjudged rather than proven irrelevant.

## Evaluation arms

| Arm | Harness | Retrieval capability | Instruction artifact |
|---|---|---|---|
| DSH filesystem | DeepSeek Harness | bounded repository/index-page filesystem search | `dsh-techdocs-plugin/skills/fs/initial_skill.md` |
| DSH hybrid | DeepSeek Harness | BM25 + HNSW + reciprocal-rank fusion | `dsh-techdocs-plugin/skills/hybrid/initial_skill.md` |
| DSH Neo4j | DeepSeek Harness | the same hybrid seeds plus deterministic Neo4j structural expansion | `dsh-techdocs-plugin/skills/neo4j/initial_skill.md` |
| Codex + FastCtx | Codex | FastCtx MCP `grep`, `glob`, and `read` over the same raw Markdown | `codex-techdocs-plugin/skills/github-docs-fastctx/SKILL.md` |

For an agent-level comparison, keep the model, question order, non-KB tools,
answer schema, token limit, and machine fixed. Only the KB plugin and its
retrieval-specific skill may vary. Retrieval-only experiments and agent-level
experiments are reported separately.

## Reference result tables

The deterministic retrieval run evaluated all 246 held-out test questions.
These scores compare KB architectures, not DSH and Codex as agents.

| Method | Recall@10 | Hit@10 | nDCG@10 | p50 retrieval |
|---|---:|---:|---:|---:|
| BM25 | 0.381 | 0.415 | 0.243 | 3.80 ms |
| HNSW | 0.470 | 0.524 | 0.306 | 17.33 ms |
| BM25 + HNSW RRF | 0.493 | 0.537 | 0.337 | 17.34 ms |
| Hybrid + conditional explicit-link expansion | 0.503 | 0.545 | 0.349 | 19.42 ms |
| Hybrid + always-on explicit-link expansion | 0.518 | 0.561 | 0.355 | 19.49 ms |
| Pure routing + BM25 | 0.288 | 0.321 | 0.196 | 3.83 ms |

Hybrid retrieval is the strongest foundation in this run. Explicit Markdown-
link expansion adds a small measured lift at about two milliseconds of median
local retrieval time. Sparse accepted-answer qrels mean an unjudged retrieved
page is not necessarily irrelevant, and this result alone does not justify
always-on graph traversal in production.

The real-agent integration pilot used five single-qrel questions. Its primary
metrics score each agent's final ordered sources.

| Arm | Hit@10 | nDCG@10 | p50 end-to-end | Tokens / QA |
|---|---:|---:|---:|---:|
| DSH filesystem | 0.600 | 0.377 | 35.01 s | 193,230 |
| DSH hybrid | 0.600 | 0.312 | 29.82 s | 36,793 |
| DSH Neo4j | 0.200 | 0.126 | 18.41 s | 15,367 |
| Codex + FastCtx | 0.200 | 0.200 | 34.95 s | 146,444 |

Do not rank the systems from this pilot. Five questions are insufficient, none
requires linked multi-page traversal, the Neo4j agent made no expansion call,
and Codex used `gpt-5.4-mini` while DSH used `gpt-5-mini`. The table verifies
the four execution paths; it is not evidence of a general harness ranking.
Raw trajectories, predictions, optimization checkpoints, and diagnostics are
intentionally not committed. New runs write beneath `results/runs/`.

## Quick verification

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
npm ci
npm run setup:dsh-profile
npm run verify:dsh
.venv/bin/python scripts/verify_package.py
PYTHONPATH=. .venv/bin/python -m pytest
```

The root and headless-profile lockfiles pin FastCtx `0.2.5` and the complete
DSH package family at `0.1.1-rc.2`. `npm run verify:dsh` rejects mismatches
between the root runtime, plugin manifest/lockfile, installed packages, and
headless profile.

Microsoft SkillOpt is installed from PyPI as the exact `skillopt==0.2.0`
dependency included by the `test`, `optimizer`, and `all` extras. Its source is
not copied into this repository.

The normalized corpus is bundled, so retrieval evaluation does not require a
second repository checkout. The filesystem and Codex/FastCtx arms need raw
Markdown; prepare the pinned source with:

```bash
scripts/prepare_raw_github_docs.sh
```

Install `requirements-graph.txt` and run Neo4j Community locally for the graph
arm. Neo4j Community is free; `neo4j-graphrag` is the existing retrieval
library, while this project defines the technical-document schema and bounded
expansion policy.

Custom OpenAI-compatible gateways are optional and are not shipped in this
repository. Codex uses its isolated default configuration unless
`github-docs-codex-fastctx` receives `--codex-home <dir>` and that directory
contains a provider-only `config.toml` (for example, an OpenCode gateway
configuration). DSH uses the official OpenAI endpoint unless
`DSH_OPENAI_BASE_URL` is set; `OPENAI_BASE_URL` is accepted as a fallback
alias. Keep provider credentials in environment variables, never in the
benchmark configuration or result files.

## Repository structure

```text
dataset/github_docs_kb_benchmark/   canonical portable corpus, qrels, splits, evaluator
evaluation/github_docs_v2/          compatibility symlinks to the canonical dataset
evaluation/harness/                 DSH/Codex patches, skills, and answer schema
evaluation/skillopt/                frozen SkillOpt splits and arm configurations
kbbench/                            retrievers, services, runners, scoring, SkillOpt adapter
dsh-techdocs-plugin/                DSH service, provider, tool, and arm-specific skill plugins
codex-techdocs-plugin/              matched Codex + FastCtx retrieval skill
dsh_home/                           pinned headless DSH profile and lockfile
scripts/                            setup, verification, training, evaluation, and analysis CLIs
tests/                              Python contract and integration tests
docs/                               evaluation protocol and implementation plans
results/                            generated-run contract; only its README is tracked
```

The normalized corpus and frozen question/split files are tracked under
`dataset/github_docs_kb_benchmark/`. The raw `github/docs` checkout is created
on demand under the ignored `data/github-docs/` path. Retrieval indexes,
Neo4j state, model trajectories, optimization checkpoints, and reports are
also generated locally and are not committed. There is no archived benchmark
or copied third-party source tree in the current repository.

See [`dataset/github_docs_kb_benchmark/README.md`](dataset/github_docs_kb_benchmark/README.md)
for the portable dataset contract, [`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md)
for the fair-comparison boundary, and [`results/README.md`](results/README.md)
for the generated-output layout.
