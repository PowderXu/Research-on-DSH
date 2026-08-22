# DocsQA -- GitHub-Style Docs KB benchmark

This repository is now the standalone benchmark for a knowledge base built from
many small, linked Markdown files. It packages the GitHub Docs corpus, 328 real
support questions, four agent configurations, retrieval/agent evaluators,
SkillOpt overlays, and the existing result evidence.

The previous SWE-bench coding-agent experiment is preserved under
[`archive/swebench_agent_trial`](archive/swebench_agent_trial/README.md); it is
not part of the default commands.

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

## Four configurations

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

## Layout

```text
dataset/github_docs_kb_benchmark/   portable corpus, questions, qrels, evaluator
kbbench/                            retrieval, plugin service, agent runners
dsh-techdocs-plugin/                DSH service/provider/skill plugins
codex-techdocs-plugin/              matched Codex + FastCtx skill
evaluation/harness/                 arm-specific DSH patches and answer schema
evaluation/skillopt/                frozen SkillOpt splits and configurations
results/reference/                  imported retrieval and real-agent evidence
results/optimization/               SkillOpt trajectories and selected skills
results/diagnostics/                failed gates and development-only evidence
archive/swebench_agent_trial/       superseded standalone benchmark
```

See [`docs/EVALUATION_PROTOCOL.md`](docs/EVALUATION_PROTOCOL.md) for the fair
comparison boundary and [`docs/RESULTS.md`](docs/RESULTS.md) for what the
existing results do—and do not—show.
