# Codex + FastCtx GitHub Docs protocol

## Purpose

Add a real Codex filesystem-retrieval arm to the frozen GitHub Docs pilot
without changing the questions, qrels, corpus, cutoffs, or DSH results.

The arm is `codex-fastctx`:

- actual Codex CLI 0.147.0;
- FastCtx 0.2.5 as an invocation-scoped local MCP server;
- only FastCtx `grep`, `glob`, and `read` exposed for filesystem retrieval;
- read-only Codex sandbox and zero shell commands in compliant episodes;
- the same filesystem retrieval policy as the DSH `fs` arm, translated to
  Codex skill frontmatter and FastCtx tool names;
- one fresh ephemeral Codex task per question;
- structured answer and ordered source output.

The source skill is
`codex-techdocs-plugin/skills/github-docs-fastctx/SKILL.md`. The evaluator
installs it under the pinned docs repository's `.agents/skills` directory and
also supplies its exact text as invocation-scoped developer instructions. The
second delivery path makes policy loading auditable even though Codex skill
activation is not emitted as a normal MCP tool call.

## Skill parity

The DSH and Codex skills have identical numbered strategy and failure-recovery
rules:

1. preserve exact query signals;
2. use `index.md` metadata for bounded routing when the domain is clear;
3. treat frontmatter only as a routing clue;
4. grep the smallest promising subtree;
5. verify candidate evidence before citing;
6. treat index pages as routers unless their bodies answer the question;
7. cite canonical pages instead of redirect aliases.

Only harness-specific tool names and Codex-compatible frontmatter differ.

## Frozen pilot

The five IDs are `108045`, `122713`, `180308`, `26749`, and `57244`, matching
the existing DSH system report. The corpus is the pinned 3,740-page GitHub Docs
repository and the qrels are unchanged accepted-answer documentation links.

Run the arm:

```bash
PYTHONPATH=. ../.venv/bin/python -m kbbench.github_docs_codex_fastctx_eval \
  --output results/runs/github_docs_codex_fastctx_paired5_v1 \
  --codex-bin "$(command -v codex)" \
  --fastctx-bin node_modules/.bin/fastctx \
  --model gpt-5.4-mini \
  --reasoning-effort low
```

The default invocation passes `--ignore-user-config` for isolation. To use an
optional OpenAI-compatible gateway, pass `--codex-home /path/to/isolated-home`;
the runner activates it only when that directory contains `config.toml`, then
records `provider_mode=custom_codex_home` without copying the configuration.
The supplied home should contain provider configuration only so the benchmark
does not inherit unrelated tools or instructions.

Generate the four-arm report:

```bash
PYTHONPATH=. ../.venv/bin/python -m kbbench.github_docs_dsh_system_report \
  --fs results/runs/skillopt_github_docs_fs_microopt1 \
  --hybrid results/runs/skillopt_github_docs_hybrid_microopt2 \
  --neo4j results/runs/skillopt_github_docs_neo4j_microopt1 \
  --codex-fastctx results/runs/github_docs_codex_fastctx_paired5_v1 \
  --out-dir results/runs/github_docs_four_arm_paired5_v1
```

## Metrics

The report preserves the original visible-result Recall@10, Hit@10, and
nDCG@10. These metrics rank distinct Markdown documents in the order search
tools first exposed them. It also reports citation-ranked Hit@10 and nDCG@10,
which rank the final evidence selected by the agent. Both views are required:
a broad early glob can lower visible-result ranking even when the final answer
cites a relevant page.

Latency is end-to-end agent p50. Tokens include fresh input, cached input, and
output. Tool-call counts are trace-derived. Every official Codex episode must
have at least one FastCtx call, no FastCtx errors, and zero shell commands.

## Reference result location

The consolidated historical table is in the repository root `README.md`. Raw
pilot outputs are intentionally not bundled.

## Interpretation boundary

This is not a harness-only causal comparison. The existing DSH trajectories
used `gpt-5-mini`; Codex with ChatGPT authentication rejected that model, so
the Codex arm used the supported `gpt-5.4-mini`. Replacing global Codex
authentication with API-key login only for this run would mutate user account
state and was not done. A formal harness comparison requires rerunning the DSH
arms on `gpt-5.4-mini`, or another model supported identically by both
harnesses.

Five questions are insufficient for ranking claims, no question in this slice
is `multi_page_linked`, the Neo4j profile made no expansion calls, and current
qrels are incomplete accepted-answer citations rather than exhaustive human
relevance judgments. The result demonstrates the integration and exposes its
failure modes; it does not establish a general FastCtx, DSH, or graph ranking.
