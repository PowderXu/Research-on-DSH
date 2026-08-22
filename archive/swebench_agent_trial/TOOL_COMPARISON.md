# Repository-search tool comparison

> **Arm key:** `C` = Codex, `D` = DSH, `3` = the D3 full-agent optional-KB
> integration design, `S` = skill-selected KB, and `F` = FastCtx search. See
> the [complete notation guide](README.md#read-the-arm-names-first).

## Standalone DSH validation

Before running the cross-harness comparison, the package can exercise the
complete public benchmark path with only `D3-S`:

```bash
.venv/bin/python scripts/check_setup.py --dsh-only
PYTHONPATH=. .venv/bin/python -m kbbench.full_agent_eval \
  --arm-set d3s-only \
  --output results/repro_small_d3s_only \
  --max-paid-usd 20
```

This is a real five-task run. The matching official scoring command is in the
[README](README.md#run-the-repository-search-tool-study). It validates that a
reader can run the DSH harness, plugin, KB service, prediction export, and
SWE-bench evaluator before introducing Codex. Its output is not included in
the comparison table below; the controlled pair reruns `D3-S` alongside
`C3-SF`.

## Question

Does DSH use fewer tokens and less time mainly because its structured
filesystem-search plugin is more efficient than Codex's ordinary shell-led
repository search?

The treatment added FastCtx 0.2.5 to Codex while preserving Codex's full coding
capabilities and the same optional technical-doc KB:

```text
C3-S   Codex + ordinary shell-led search
C3-SF  Codex + required FastCtx grep/glob MCP
D3-S   DSH   + dsh-tool-fs-search
```

`C3-SF` and `D3-S` prohibited shell `rg`, `grep`, and `find` for repository
discovery. Shell remained available for tests, builds, version-control
inspection, and other non-search operations. The runner audited compliance.

## Controls

The three arms used:

- the same five SWE-bench Verified Django issues;
- `gpt-5.4-mini` with low reasoning effort;
- clean detached worktrees at the same base commits;
- full coding-agent capabilities;
- web disabled;
- the same optional `techdocs-research` skill and one composite KB contract;
- the same KB result limit of 8 and evidence budget of 2,200 tokens; and
- official SWE-bench Docker scoring.

The harnesses' system prompts, native tool inventories, context policies, and
agent loops were not identical. Those differences are the subject of the
Codex-versus-DSH comparison.

## Results

All values except `Resolved` are means over five tasks.

| Metric | `C3-S` | `C3-SF` | `D3-S` |
|---|---:|---:|---:|
| Official resolved | 4/5 | 4/5 | 4/5 |
| Fresh input tokens | 37,435 | 32,753 | 24,428 |
| Cached input tokens | 369,382 | 336,461 | 177,050 |
| Output tokens | 5,221 | 4,634 | 2,112 |
| Total tokens | 412,038 | 373,847 | 203,589 |
| Agent latency | 95.8 s | 90.4 s | 36.2 s |
| Structured filesystem calls | 0.0 | 14.4 | 6.8 |
| Structured search-result characters | 0 | 21,384 | 12,761 |
| Shell-search calls | 3.8 | 0.0 | 0.0 |
| Shell-search characters | 10,290 | 0 | 0 |
| All recorded tool events | 20.6 | 22.4 | 17.8 |
| Technical-doc KB calls | 0 | 0 | 0 |

The official evaluator resolved exactly the same four task IDs for all three
arms. The only common failure was `django__django-11239`.

## Contrasts

### FastCtx Codex versus original Codex

Relative to `C3-S`, `C3-SF` used:

- 9.3% fewer total tokens;
- 5.7% less agent wall time;
- 8.9% fewer cached-input tokens; and
- more repository-search calls and more returned search characters.

FastCtx produced a modest efficiency improvement, but not by simply returning
less search text. It returned about twice as many search-result characters as
the original shell searches.

### DSH versus FastCtx Codex

Relative to `C3-SF`, `D3-S` used:

- 45.5% fewer total tokens;
- 59.9% less agent wall time;
- 47.4% fewer cached-input tokens; and
- 52.8% fewer structured-search calls.

The mean total-token gap was 170,258 tokens/task. Its observed components were:

| Component | `C3-SF` minus `D3-S` | Share of total gap |
|---|---:|---:|
| Fresh input | 8,325 | 4.9% |
| Cached input | 159,411 | 93.6% |
| Output | 2,522 | 1.5% |

This identifies **where the accounting gap appears**—mostly cached input—but
not why DSH accumulated less of it.

## Interpretation

The tool hypothesis is only partially supported:

- A structured repository-search tool made Codex somewhat cheaper and faster.
- It did not remove the much larger DSH efficiency advantage.
- Raw search-result size does not explain the result: FastCtx Codex returned
  more search characters than original Codex, yet used fewer total tokens.
- DSH also made fewer total tool events, but current Codex telemetry does not
  expose exact per-model-call boundaries, so the study cannot establish whether
  the remaining gap is caused by fewer inference passes, less context per pass,
  or both.

Because no arm invoked the KB, this experiment says nothing about the accuracy
of BM25, vector, link-graph, or composite KB retrieval. It only shows how the
optional KB coexisted with each full coding harness while repository search was
varied.

## Validity limits

- Five tasks are an integration/effect-direction pilot, not a powered
  benchmark.
- All tasks are from Django and four were resolved by every arm.
- `C3-SF` and `D3-S` were alternated by task, but `C3-S` was run afterward.
- The requested model ID and reasoning effort matched, but Codex and DSH used
  different authentication/accounting paths.
- Agent wall time can include local process, tool, and service noise.
- Search calls and returned characters do not measure the semantic usefulness
  of the returned evidence.

## Evidence

- Registered tool protocol:
  [`full_agent_fastctx_d3s_protocol_v2.json`](config/full_agent_fastctx_d3s_protocol_v2.json)
- Registered three-arm pilot:
  [`fastctx_pilot5_protocol_v1.json`](config/fastctx_pilot5_protocol_v1.json)
- Original result report:
  [`TOOL_PILOT_RESULTS.md`](evidence/TOOL_PILOT_RESULTS.md)
- Machine-readable aggregate results:
  [`tool_pilot_results.json`](evidence/tool_pilot_results.json)

The standalone package omits the original private raw trajectories. Running
the commands in [README.md](README.md) creates new run states, traces,
predictions, and official score matrices under `results/`.
