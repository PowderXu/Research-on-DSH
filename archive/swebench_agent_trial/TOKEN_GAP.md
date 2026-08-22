# Token-gap status: unresolved

> **Arm key:** `C3-SF` is Codex with skill-selected optional KB and FastCtx;
> `D3-S` is the matched DSH skill-selected arm with native filesystem search.
> See the [complete notation guide](README.md#read-the-arm-names-first).

## Current conclusion

The small trial has located the Codex–DSH accounting difference primarily in
**accumulated input tokens, especially cached input**, but it has not identified
the system mechanism that causes DSH to accumulate less context.

For the original tool pilot:

```text
C3-SF total: 373,847 tokens/task
D3-S  total: 203,589 tokens/task
Gap:          170,258 tokens/task
```

Of that observed gap:

```text
fresh input difference:   8,325  ( 4.9%)
cached input difference: 159,411 (93.6%)
output difference:        2,522  ( 1.5%)
```

This decomposition answers **where** the reported tokens differ. It does not
answer **why**.

## Explanations weakened by the current evidence

### 1. Shell search alone

Replacing Codex shell search with FastCtx reduced total tokens by 9.3%, but DSH
still used 45.5% fewer tokens than FastCtx Codex at the same 4/5 resolution.
The repository-search backend is therefore not a sufficient explanation.

### 2. Raw search-result size alone

FastCtx Codex received more search-result characters than original Codex but
used fewer tokens. In the narration trial, quiet Codex received fewer search
characters yet used slightly more total tokens. Character volume from search
results is not a sufficient predictor.

### 3. Visible narration tokens alone

Quiet Codex removed 98.6% of progress characters but total tokens increased
1.0%. Narrated DSH emitted more progress text while total tokens decreased
40.4%. The direct price of visible prose is not the primary explanation.

### 4. Final-answer length alone

Final answers were a small part of the full trajectory. Quiet Codex produced a
slightly longer final answer despite slightly fewer output tokens overall. The
gap is dominated by input processing, not the final summary.

## Remaining plausible mechanisms

The following are hypotheses, not findings:

1. **Different number of model calls.** More inference passes cause prior
   context to be processed repeatedly.
2. **Different context per call.** Codex may retain or serialize more system
   prompt, tool schemas, assistant commentary, tool calls, tool results, plans,
   or file evidence on each pass.
3. **Different batching.** DSH may issue more tools from one model response,
   reducing round trips. The DSH narration treatment demonstrated that batching
   can change materially, but exact Codex step counts are unavailable.
4. **Different tool-result replay or trimming.** The harnesses may preserve,
   compact, truncate, or re-render earlier tool results differently.
5. **Different reasoning-state continuation.** The two API/account paths may
   preserve or replay reasoning state differently even with the same requested
   model identifier and reasoning effort.
6. **Different static context.** Native system prompts, plugin instructions,
   tool definitions, and schemas may differ in size and cache behavior.
7. **Telemetry-semantic differences.** Total tokens use the same arithmetic,
   but Codex reports one aggregate `turn.completed` usage object while DSH is
   summed from per-message API usage. Fresh-versus-cached labels should not be
   treated as proof that both paths serialize identical inputs.
8. **Stochastic trajectory variation.** Only five distinct tasks were used,
   and repeated trajectories varied substantially.

## Why the current trace cannot settle it

DSH session traces expose per-step boundaries and per-message usage. Codex's
non-interactive JSONL trace exposes item events and one aggregate usage object
for the completed turn. It does not expose, for every internal model request:

- a stable response/request ID;
- fresh, cached, reasoning, and output tokens;
- the exact serialized input;
- component-level input token counts;
- whether adjacent narration and tool-call events share one response; or
- an authoritative model-call count.

Consequently, the attractive equation

```text
more rounds × larger context per round = more cached-input usage
```

is plausible but not yet verified from the current Codex telemetry.

## Instrumentation required for the next experiment

For every model request in both harnesses, record:

| Field | Purpose |
|---|---|
| request and response IDs | establish exact model-call boundaries |
| start/end timestamps | separate model latency from tool latency |
| fresh and cached input tokens | locate repeated processing |
| output and reasoning tokens | separate generation from input cost |
| serialized-input hash and byte/token count | detect context growth and replay |
| system/developer prompt tokens | measure static harness overhead |
| tool-definition/schema tokens | measure plugin/tool inventory overhead |
| prior assistant-message tokens | measure narration/history carryover |
| prior tool-call tokens | measure action-history carryover |
| tool-result tokens | measure evidence replay and trimming |
| output item types | distinguish commentary, tool calls, reasoning, and final answer |
| tools emitted per response | measure batching and parallelism |

The most useful derived table would be:

| Metric | Codex | DSH |
|---|---:|---:|
| model calls/task | unknown now | available |
| mean fresh input/call | unknown | computable with enhanced export |
| mean cached input/call | unknown | computable with enhanced export |
| mean tool-result tokens retained/call | unknown | unknown |
| tools/model call | inferred only | available |
| context growth by call | unknown | partially reconstructable |

## Recommended next ablation

Use the same five public tasks for diagnosis before spending on a larger run:

1. Keep `C3-SF` and `D3-S` on their current matched model, reasoning effort,
   FastCtx/DSH structured search, and full coding capability.
2. Add per-model-request instrumentation rather than another prompt treatment.
3. Run at least five paired repetitions.
4. Decompose total input into static prompt, tool schemas, history, tool results,
   and other state.
5. Test the two factors suggested by the trace:
   - fixed maximum tools per model response; and
   - matched tool-result retention/truncation.
6. Treat accuracy as a guardrail: a token reduction counts as an improvement
   only if official patch resolution and required validation do not regress.

Only after that diagnostic run should the study claim whether DSH's advantage
comes from fewer calls, smaller context per call, more batching, different
history retention, or a combination.

## What may be stated now

Safe statement:

> On five public SWE-bench Verified Django tasks, DSH matched the three-arm
> tool pilot's 4/5 resolution while using substantially fewer aggregate tokens
> and less agent wall time. FastCtx explained a modest part of the Codex cost,
> and visible narration did not explain the remaining gap. Most of the observed
> token difference was reported as cached input, but the causal source remains
> unresolved because Codex lacks per-model-call context telemetry in this run.

Unsafe statements include:

- “DSH is generally twice as efficient as Codex.”
- “DSH wins because it uses fewer reasoning rounds.”
- “Codex narration causes the token gap.”
- “The DSH KB improves SWE-bench accuracy.”
- “Cached tokens alone explain latency.”
