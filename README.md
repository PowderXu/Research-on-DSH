# Small Codex–DSH agent trial

This directory packages the current small real-model comparison between Codex
and DeepSeek Harness (DSH). It contains two related studies over the same five
public coding tasks:

1. a repository-search tool comparison; and
2. a progress-narration policy comparison.

The package is deliberately conservative. It records a persistent token and
latency gap, but it does **not** claim to have identified the mechanism that
causes that gap.

This directory is standalone: copy or clone it as its own repository and run
all commands below from this directory. It includes the Python runner and
official-score wrapper, the DSH technical-document plugin, the frozen five-task
manifest, registered protocols, and compact observed-result artifacts. It does
not depend on files elsewhere in the original development repository.

## Read the arm names first

An **arm** is one benchmark configuration applied to every selected task. Arm
IDs use this pattern:

```text
<harness><integration-design>-<activation/search treatment>-<optional narration treatment>
```

| Symbol | Meaning |
|---|---|
| `C` | Codex harness |
| `D` | DeepSeek Harness (DSH) |
| `3` | The third registered KB-integration design, called the **D3 revision**. It keeps a full coding agent and adds the optional technical-document KB composition. It does not mean three tools or three model calls. |
| `S` | **Skill-selected**: the model sees a skill and may choose the one composite KB tool. KB retrieval is optional. |
| `F` | **FastCtx** repository search is added to the Codex `S` arm; only FastCtx `grep` and `glob` are exposed for repository discovery. |
| `Q` | **Quiet** Codex narration treatment: suppress intermediate user-facing prose. |
| `N` | **Narrated** DSH treatment: require one short progress sentence before tool calls. |

Therefore, the arms in this package expand as follows:

| Arm | Plain-English meaning |
|---|---|
| `C3-S` | Codex, D3 integration design, skill-selected optional KB, normal shell-led repository search, default narration. |
| `C3-SF` | `C3-S` with FastCtx replacing shell `rg`/`grep`/`find` for repository discovery. |
| `C3-SF-Q` | `C3-SF` with quiet intermediate narration. |
| `D3-S` | DSH, D3 integration design, matched skill-selected optional KB, native DSH filesystem-search plugin, default narration. |
| `D3-S-N` | `D3-S` with the forced short-narration treatment. |

Other recurring abbreviations are **KB** (knowledge base), **MCP** (Model
Context Protocol, the tool interface used by Codex), and **FS** (filesystem).
All arms retain normal coding abilities such as reading and editing files,
running shell commands, and testing. `Resolved` means the generated patch
passed the official SWE-bench evaluator for that task.

## Headline result

In the original five-task tool pilot, all three agents resolved the same four
tasks:

| Arm | Resolved | Mean tokens/task | Mean agent latency | Repository search |
|---|---:|---:|---:|---|
| `C3-S` | 4/5 | 412,038 | 95.8 s | Codex shell search |
| `C3-SF` | 4/5 | 373,847 | 90.4 s | FastCtx MCP `grep`/`glob` |
| `D3-S` | 4/5 | 203,589 | 36.2 s | DSH filesystem-search plugin |

FastCtx reduced Codex tokens by 9.3%, but DSH still used 45.5% fewer tokens
than FastCtx Codex. Therefore, replacing shell search with a structured search
tool did not explain the remaining gap.

The narration experiment also failed to explain it. Across five independent
rollouts of the five tasks, suppressing Codex progress narration removed 98.6%
of visible progress text but changed total tokens by **+1.0%**, not downward.

The best-supported current statement is:

> DSH accumulated substantially less input context than Codex in this pilot.
> Search-tool output size and visible progress narration do not explain most of
> the difference. The exact contribution of model-call count, context carried
> per call, tool-result replay, prompt/tool-schema size, and batching remains
> unresolved.

See [TOKEN_GAP.md](TOKEN_GAP.md) for the evidence boundary and next required
instrumentation.

## Public benchmark provenance

The tasks come from
[SWE-bench Verified](https://www.swebench.com/SWE-bench/guides/datasets/), a
public 500-instance subset of SWE-bench that was screened by software engineers.
SWE-bench instances are real GitHub issue-resolution tasks. A prediction is a
repository patch, and the official evaluator applies that patch and runs tests
inside a Docker environment. See the
[official evaluation guide](https://www.swebench.com/SWE-bench/guides/evaluation/)
and the
[SWE-bench harness reference](https://github.com/SWE-bench/SWE-bench/blob/main/docs/reference/harness.md).

This trial is **not** the complete 500-task benchmark. It uses five Django tasks
from the frozen manifest
[`swebench_fastctx_pilot5_v1.json`](config/swebench_fastctx_pilot5_v1.json).
The runnable standalone copy flattens each task's stratum and difficulty from
the original parent manifest so no external file is required. The unmodified
child registration is retained as
[`swebench_fastctx_pilot5_v1.original.json`](config/swebench_fastctx_pilot5_v1.original.json).
They are the first five tasks in the deterministic, stratum-balanced schedule
of the repository's 27-task documentation-aware Django subset:

| Instance | Selection stratum | Upstream difficulty |
|---|---|---|
| `django__django-11239` | code only | `<15 min fix` |
| `django__django-12741` | latent documentation/code | `<15 min fix` |
| `django__django-13741` | directly documentation-related | `<15 min fix` |
| `django__django-12209` | code only | `<15 min fix` |
| `django__django-13109` | latent documentation/code | `<15 min fix` |

The sample was not randomly drawn and is too small for a general Codex-versus-
DSH capability claim. Repeating these five tasks measures stochastic stability
on these tasks, not generalization to 25 independent tasks.

Gold patches were used offline when constructing the parent strata. Gold patch
content, test patches, solution URLs, and gold-derived prose were not exposed
to either agent.

## Matched controls and remaining differences

`C3-SF` and `D3-S` preserve full coding-agent capability: file editing, shell
commands, testing, and the optional KB remain available. The comparison is not
"Codex with coding tools" versus "DSH with only a KB."

The requested model identifier was `gpt-5.4-mini` with low reasoning effort in
every arm. Web access was disabled. Every task started from a clean detached
worktree at the public SWE-bench base commit.

The Codex and DSH harnesses still have different system prompts, agent loops,
context policies, and account/API paths. Those are part of the harness
treatment; they are not controlled away.

## What this trial does and does not test

The three-arm tool pilot made zero technical-doc KB calls in every arm. It tests
repository search and harness orchestration, **not KB retrieval accuracy**.

Across the later 25-rollout narration study, `D3-S` made four KB calls in total
(`0.16` per task); the other three arms made none. This is too sparse to support
a KB-performance claim.

For the detailed studies, read:

- [TOOL_COMPARISON.md](TOOL_COMPARISON.md)
- [NARRATION_COMPARISON.md](NARRATION_COMPARISON.md)
- [TOKEN_GAP.md](TOKEN_GAP.md)

## Package contents

- `kbbench/`: executable Python runner, retrieval service, trace parsers, and
  official SWE-bench matrix scorer.
- `codex/`: checked-in Codex skill, narration treatment, and an explicit map of
  the Codex-side invocation policy.
- `dsh-techdocs-plugin/`: the DSH plugin used by `D3-S` and `D3-S-N`, including
  its unit tests.
- `dsh_home/profiles/headless/`: reproducible headless DSH profile that loads
  the local plugin.
- `config/`: frozen task manifest and the registered tool/narration protocols.
- `evidence/`: compact reports and machine-readable aggregate results from the
  original run.
- `scripts/prepare_data.py`: downloads and validates the public dataset and
  Django mirror.
- `scripts/check_setup.py`: checks the local runtime before paid model calls.

Bundled evidence:

- [tool protocol](config/full_agent_fastctx_d3s_protocol_v2.json)
- [three-arm pilot registration](config/fastctx_pilot5_protocol_v1.json)
- [narration protocol](config/full_agent_narration_ablation_protocol_v1.json)
- [tool-pilot report](evidence/TOOL_PILOT_RESULTS.md)
- [tool-pilot aggregate JSON](evidence/tool_pilot_results.json)
- [five-trial narration report](evidence/NARRATION_5TRIAL_RESULTS.md)

Full raw model trajectories from the original private run are intentionally not
distributed. A reproduction creates `run_state.json`, prediction patches,
per-task raw traces, and `score_matrix_state.json` under its own `results/`
directory.

## Prerequisites

Run commands from the root of this standalone directory, `bench_small_trial/`.

Required software:

- Python 3.10 or later;
- Node.js 22.19 or later (or Node 24);
- Docker Desktop or Docker Engine;
- an installed and authenticated Codex CLI;
- an OpenAI API key for the DSH model-provider path;
- `git`, `npm`, and `zstd`.

Codex is intentionally not vendored. Install it and sign in following the
[official Codex CLI guide](https://developers.openai.com/codex/cli/), then run
`codex --version`. See [codex/README.md](codex/README.md) for the exact
Codex-side treatment bundled here.

The original trial environment used Python `3.10.15`, Codex CLI `0.147.0`,
DSH `0.1.0-rc.6`, FastCtx `0.2.5`, Node.js `22.22.0`, and the
`gpt-5.4-mini` model identifier with low reasoning effort. Direct Python
packages, DSH, and FastCtx are pinned by this package. Codex and the model
service are external, so record their versions in any reproduction and report
deviations from the original environment.

Create an isolated Python environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

The direct Python dependencies used by the runner and official scorer are
version-pinned in both `pyproject.toml` and `requirements.txt`.

Install the pinned DSH and FastCtx dependencies:

```bash
npm install
npm run setup:dsh-profile
npm run test:plugin
```

The direct runtime versions are pinned in `package.json`: DSH
`0.1.0-rc.6` and FastCtx `0.2.5`. The DSH release has a large transitive
dependency tree, so its first installation can take several minutes. This
package does not vendor `node_modules`.

The runner reads `OPENAI_API_KEY` from the environment or from `--env-file`.
Do not commit an API key.

The expected local public-data paths are:

```text
data/swebench_verified/test.parquet
data/swebench_verified/repos/django.git
```

Download the official test split, create a mirror containing the historical
Django commits, and verify the five manifest IDs:

```bash
.venv/bin/python scripts/prepare_data.py
```

The manifest fixes the five instance IDs. A newly serialized Parquet file may
have a different byte hash across library versions even when its records are
equivalent; preserve the checked-in manifest and verify that all five IDs are
present.

Export the API key used by the DSH OpenAI provider, confirm that Codex is
authenticated, start Docker, and run the readiness check:

```bash
export OPENAI_API_KEY="your-key"
codex --version
docker --version
.venv/bin/python scripts/check_setup.py
```

The setup checker is read-only. Without flags, it deliberately fails if the API
key, data, Docker CLI, Codex CLI, DSH runtime, FastCtx runtime, or DSH profile
is missing. Its `--dsh-only` mode, used below, omits the Codex and FastCtx
requirements. Do not commit `.env`, credentials, downloaded data, generated
runs, or `node_modules`; the included `.gitignore` excludes them.

## Run the repository-search tool study

First validate the complete DSH-only path. This runs the registered `D3-S` arm
over all five tasks; it is a real benchmark run, not a one-task smoke test:

```bash
.venv/bin/python scripts/check_setup.py --dsh-only

PYTHONPATH=. .venv/bin/python -m kbbench.full_agent_eval \
  --arm-set d3s-only \
  --output results/repro_small_d3s_only \
  --max-paid-usd 20
```

Score that standalone run with the official SWE-bench evaluator:

```bash
PYTHONPATH=. .venv/bin/python -m kbbench.swebench_score_matrix \
  --manifest config/swebench_fastctx_pilot5_v1.json \
  --predictions-dir results/repro_small_d3s_only/predictions \
  --output-dir results/repro_small_d3s_only/official_pilot5 \
  --arms D3-S \
  --run-prefix repro-small-d3s-only \
  --arm-workers 1
```

Completion validates the DSH agent, plugin/profile, technical-doc service,
five-task manifest, prediction export, and official scoring path without
requiring Codex or FastCtx. Keep this validation output separate from the
comparison results below.

Next run the paired FastCtx Codex and DSH arms:

```bash
PYTHONPATH=. .venv/bin/python -m kbbench.full_agent_eval \
  --arm-set fastctx-d3s \
  --output results/repro_small_tool_pair \
  --max-paid-usd 20
```

Then run the original shell-search Codex reference separately:

```bash
PYTHONPATH=. .venv/bin/python -m kbbench.full_agent_eval \
  --arm-set c3s-pilot \
  --output results/repro_small_codex_shell \
  --max-paid-usd 20
```

The local paths shown in the prerequisites are runner defaults. If `codex` is
not on `PATH`, add `--codex-bin /absolute/path/to/codex`. Other runtime and data
paths can likewise be overridden with the corresponding CLI flags.

The paired run intentionally executes `D3-S` again: only that run alternates
`C3-SF` and `D3-S` by task, so the standalone validation must not be substituted
into the comparison. The `C3-S` reference is separate because the original
study added it after the alternating pair. That timing difference is a known
limitation.

## Score the tool study

Start Docker, then score the pair:

```bash
PYTHONPATH=. .venv/bin/python -m kbbench.swebench_score_matrix \
  --manifest config/swebench_fastctx_pilot5_v1.json \
  --predictions-dir results/repro_small_tool_pair/predictions \
  --output-dir results/repro_small_tool_pair/official_pilot5 \
  --arms C3-SF,D3-S \
  --run-prefix repro-small-tool-pair \
  --arm-workers 2
```

Score the original Codex arm:

```bash
PYTHONPATH=. .venv/bin/python -m kbbench.swebench_score_matrix \
  --manifest config/swebench_fastctx_pilot5_v1.json \
  --predictions-dir results/repro_small_codex_shell/predictions \
  --output-dir results/repro_small_codex_shell/official_pilot5 \
  --arms C3-S \
  --run-prefix repro-small-codex-shell \
  --arm-workers 1
```

Official scoring pulls Linux images and can require substantial disk space.
Agent wall-clock latency reported by this study excludes Docker scoring time.

## Run the narration study

One narration trial runs all four arms over the same five tasks:

```bash
PYTHONPATH=. .venv/bin/python -m kbbench.full_agent_eval \
  --arm-set narration-ablation \
  --output results/repro_small_narration_v1 \
  --max-paid-usd 20
```

Score it with:

```bash
PYTHONPATH=. .venv/bin/python -m kbbench.swebench_score_matrix \
  --manifest config/swebench_fastctx_pilot5_v1.json \
  --predictions-dir results/repro_small_narration_v1/predictions \
  --output-dir results/repro_small_narration_v1/official_pilot5 \
  --arms C3-SF,C3-SF-Q,D3-S,D3-S-N \
  --run-prefix repro-small-narration-v1 \
  --arm-workers 2
```

To reproduce the five-trial stability study, repeat both commands with new
output directories and run prefixes ending in `v2` through `v5`. Never reuse an
output directory for a different arm set; the runner rejects incompatible
state. Existing completed cases are resumable, and `--retry-failed` reruns only
nonzero-return-code cases while preserving failed-attempt metadata.

## Metrics

- **Resolved:** official SWE-bench patch resolution after Docker tests.
- **Fresh input:** non-cached input tokens reported by the model path.
- **Cached input:** prompt-cache read tokens; these still count in aggregate
  token throughput but are normally cheaper than fresh input.
- **Output:** all model output tokens reported by the path, not merely the final
  prose.
- **Total tokens:** fresh input + cached input + output, summed over the complete
  task trajectory.
- **Latency:** agent wall time from invocation through patch completion,
  excluding official Docker scoring.
- **Progress characters:** visible non-final assistant text characters. This is
  a character counter, not a token counter.

Token totals are comparable arithmetic aggregates, but the fresh/cached
decomposition comes from different telemetry surfaces: Codex's final
`turn.completed` usage and DSH's per-message API usage. This distinction is one
reason the token mechanism remains an open question.
