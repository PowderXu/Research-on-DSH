# Codex treatment

Codex itself is an external CLI and is not vendored in this repository. The
benchmark vendors every custom Codex-side treatment that it controls and
invokes Codex through `kbbench/full_agent_eval.py`.

The official Codex CLI guide covers installation and sign-in:
https://developers.openai.com/codex/cli/

The original trial used Codex CLI `0.147.0`. A reproduction should record its
actual `codex --version`; pass a non-default executable with
`--codex-bin /absolute/path/to/codex`.

## Arms

| Arm | Custom Codex treatment |
|---|---|
| `C3-S` | The checked-in `techdocs-research` skill plus the optional technical-doc MCP; normal shell-led repository search. |
| `C3-SF` | `C3-S` plus FastCtx `grep` and `glob`, with shell `rg`, `grep`, and `find` prohibited for repository discovery. |
| `C3-SF-Q` | `C3-SF` plus the checked-in quiet narration developer instruction. |

The runner copies `techdocs-research/SKILL.md` into each clean task worktree.
For FastCtx arms it appends one search-policy paragraph. For the quiet arm it
loads `silent_narration_instructions.txt` as the invocation's developer
instructions.

Every Codex arm uses non-interactive `codex exec --json` with:

- an ephemeral session;
- the task worktree as the working directory;
- workspace-write sandboxing and no interactive approvals;
- user configuration ignored;
- apps, browser use, and computer use disabled, with web access prohibited by
  the registered task prompt;
- the registered model and reasoning effort;
- an invocation-scoped technical-doc MCP configuration; and
- for `C3-SF` arms, an invocation-scoped FastCtx MCP configuration exposing
  only `grep` and `glob`.

The Codex trace parser records reported aggregate token usage, completed tool
events, repository-search calls and result characters, progress/final text
characters, generated patches, and wall time. It does not expose authoritative
per-internal-model-request context, which is why the causal token-gap analysis
remains unresolved.
