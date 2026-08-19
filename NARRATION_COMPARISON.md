# Progress-narration policy comparison

> **Arm key:** `C` = Codex, `D` = DSH, `3` = the D3 full-agent optional-KB
> integration design, `S` = skill-selected KB, `F` = FastCtx, `Q` = quiet
> Codex, and `N` = narrated DSH. See the
> [complete notation guide](README.md#read-the-arm-names-first).

## Question

Is visible progress narration a primary cause of the Codex-versus-DSH token and
latency difference?

This is more accurately described as a **preamble-policy ablation** than a pure
text-deletion experiment. The treatment changes an instruction given to the
model, so it can change narration, tool selection, tool arguments, number of
steps, returned evidence, and the final answer.

## Arms

```text
C3-SF    Codex + FastCtx, default Codex narration
C3-SF-Q  Codex + FastCtx, no intermediate user-facing prose
D3-S     DSH + filesystem-search plugin, default DSH narration
D3-S-N   DSH + same plugin, one short progress sentence before tool calls
```

Within Codex, the only intended treatment was the quiet developer instruction.
Within DSH, the only intended treatment was the
[`narration-ablation.mjs`](dsh-techdocs-plugin/narration-ablation.mjs)
system-prompt plugin. The task, model, reasoning effort, repository-search
backend, coding capabilities, optional KB, and validation requirements were
held fixed within each harness.

## Trial structure

- Five distinct SWE-bench Verified Django tasks.
- Five independent real-model trials of those same tasks.
- 25 judged patches per arm; 100 judged patches total.
- Official SWE-bench Docker scoring.
- No infrastructure, evaluator, empty-patch, or ambiguous failures.

The 25 observations per arm are repeated rollouts over five tasks, not 25
independent tasks.

## Official resolution

| Trial | `C3-SF` | `C3-SF-Q` | `D3-S` | `D3-S-N` |
|---|---:|---:|---:|---:|
| 1 | 4/5 | 4/5 | 2/5 | 4/5 |
| 2 | 4/5 | 4/5 | 4/5 | 4/5 |
| 3 | 5/5 | 4/5 | 4/5 | 3/5 |
| 4 | 4/5 | 5/5 | 4/5 | 3/5 |
| 5 | 4/5 | 4/5 | 3/5 | 4/5 |
| **Aggregate** | **21/25** | **21/25** | **17/25** | **18/25** |

Codex default and Codex quiet had identical aggregate resolution, with two
default-only and two quiet-only paired successes. DSH normal and DSH narrated
differed by one aggregate success. These small, unstable differences do not
establish an accuracy benefit for either narration policy.

## Efficiency and behavior

All figures below are means per task over 25 rollouts.

| Metric | `C3-SF` | `C3-SF-Q` | `D3-S` | `D3-S-N` |
|---|---:|---:|---:|---:|
| Fresh input tokens | 39,422 | 35,583 | 19,468 | 15,754 |
| Cached input tokens | 415,631 | 424,274 | 182,784 | 104,243 |
| Output tokens | 4,287 | 4,183 | 2,196 | 1,851 |
| Total tokens | 459,340 | 464,040 | 204,449 | 121,848 |
| Agent latency | 83.4 s | 96.2 s | 32.4 s | 24.4 s |
| Progress messages | 7.88 | 0.12 | 0.00 | 8.12 |
| Progress characters | 1,665 | 23 | 0 | 455 |
| Final-answer characters | 1,077 | 1,114 | 692 | 704 |
| Filesystem searches | 11.60 | 13.20 | 5.68 | 4.96 |
| Search-result characters | 36,629 | 34,128 | 9,991 | 9,456 |
| All recorded tool events | 19.80 | 20.40 | 17.12 | 14.36 |
| Recorded model steps | unavailable | unavailable | 14.64 | 9.32 |
| Multi-tool steps | unavailable | unavailable | 2.56 | 4.36 |
| KB calls | 0 | 0 | 0.16 | 0 |

## Codex treatment effect

`C3-SF-Q` versus `C3-SF`:

- progress characters: **-98.6%**;
- resolution: no aggregate change;
- output tokens: **-2.4%**;
- total tokens: **+1.0%**; and
- mean latency: **+15.4%**.

The quiet instruction successfully suppressed visible progress prose. It did
not save total tokens or latency. Quiet Codex performed slightly more filesystem
searches and total tool events, while returning fewer search-result characters.

This shows that the quiet arm was not simply the default trajectory with text
removed. Its behavior changed.

## DSH treatment effect

`D3-S-N` versus `D3-S`:

- progress characters: `+455` per task;
- total tokens: **-40.4%**;
- output tokens: **-15.7%**;
- mean latency: **-24.8%**;
- recorded model steps: **-36.3%**; and
- multi-tool steps: **+70.3%**.

Requiring narration did not merely add prose. It induced a different DSH
trajectory with fewer model steps and more tool batching. The reduction cannot
be interpreted as the token price of narration; it is a broad policy effect.

## What progress characters measure

For Codex, the evaluator collects completed `agent_message` items and treats
all non-final messages as progress text. It sums Python string lengths, not
tokens. See
[`_codex_response_trace`](kbbench/full_agent_eval.py#L368).

For DSH, progress text comes from assistant messages containing tool calls, and
the final non-tool message is counted separately.

Progress characters exclude tool results, tool-call arguments, hidden reasoning,
file patches, and the final answer. They cannot be subtracted directly from
aggregate tokens.

## What the ablation supports

Supported:

- Visible Codex progress narration was almost completely removed.
- The Codex token/latency gap remained.
- Adding DSH narration changed its entire trajectory and reduced observed work.
- Narration policy can influence tools and steps; visible text is not an
  isolated additive cost.

Not supported:

- that narration and tool calls always came from the same model response;
- that suppressing narration removed or did not remove a model invocation;
- that narration history accounts for a known number of cached tokens;
- that DSH narration generally improves coding accuracy; or
- that the narration policy explains the Codex-versus-DSH token gap.

Codex `exec --json` exposes aggregate turn usage and item events but not a
response ID and token breakdown for every internal model call. Exact Codex
model-step counts in these results are therefore unavailable.

## Evidence

- Registered protocol:
  [`full_agent_narration_ablation_protocol_v1.json`](config/full_agent_narration_ablation_protocol_v1.json)
- Five-trial report:
  [`NARRATION_5TRIAL_RESULTS.md`](evidence/NARRATION_5TRIAL_RESULTS.md)

The standalone package omits the original private raw trajectories. Running
the narration commands in [README.md](README.md) generates a complete new
`run_state.json`, raw traces, predictions, and official score matrix for each
trial.
