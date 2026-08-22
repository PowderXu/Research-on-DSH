# Five-trial narration ablation

## Scope

- Five independent real-model trials, including the original trial.
- Five SWE-bench Verified Django tasks per trial.
- Four arms per task: C3-SF, C3-SF-Q, D3-S, and D3-S-N.
- Model: `gpt-5.4-mini`, low reasoning effort.
- Official SWE-bench Docker scoring; no infrastructure, ambiguous, empty-patch, or evaluator failures.
- 25 judged patches per arm, 100 judged patches in total.

The 25 observations per arm are repeated rollouts over five distinct tasks. They measure stochastic stability on this pilot, not generalization to 25 independent tasks.

## Official resolution by trial

| Trial | C3-SF | C3-SF-Q | D3-S | D3-S-N |
|---|---:|---:|---:|---:|
| 1 | 4/5 | 4/5 | 2/5 | 4/5 |
| 2 | 4/5 | 4/5 | 4/5 | 4/5 |
| 3 | 5/5 | 4/5 | 4/5 | 3/5 |
| 4 | 4/5 | 5/5 | 4/5 | 3/5 |
| 5 | 4/5 | 4/5 | 3/5 | 4/5 |
| **Aggregate** | **21/25 (84%)** | **21/25 (84%)** | **17/25 (68%)** | **18/25 (72%)** |

Paired resolution flips:

- C3-SF versus C3-SF-Q: default-only 2, quiet-only 2, both resolved 19, neither resolved 2.
- D3-S versus D3-S-N: normal-only 2, narrated-only 3, both resolved 15, neither resolved 5.

## Aggregate efficiency

All token and latency figures are means per task over 25 rollouts. Latency is agent wall time, excluding official Docker scoring.

| Arm | Fresh input | Cached input | Output | Total tokens | Mean latency | Median latency | Progress chars | Model steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C3-SF | 39,422 | 415,631 | 4,287 | 459,340 | 83.4s | 78.1s | 1,665 | unavailable |
| C3-SF-Q | 35,583 | 424,274 | 4,183 | 464,040 | 96.2s | 85.3s | 23 | unavailable |
| D3-S | 19,468 | 182,784 | 2,196 | 204,449 | 32.4s | 30.8s | 0 | 14.64 |
| D3-S-N | 15,754 | 104,243 | 1,851 | 121,848 | 24.4s | 24.1s | 455 | 9.32 |

Direct within-harness treatment effects:

| Contrast | Resolution | Total tokens | Output tokens | Latency | Progress chars |
|---|---:|---:|---:|---:|---:|
| C3-SF-Q minus C3-SF | 0 points | +1.0% | -2.4% | +15.4% | -98.6% |
| D3-S-N minus D3-S | +4 points | -40.4% | -15.7% | -24.8% | +455/task |

DSH behavioral counters explain why the narrated treatment did not merely add prose:

| Counter | D3-S | D3-S-N | Change |
|---|---:|---:|---:|
| Model steps/task | 14.64 | 9.32 | -36.3% |
| Multi-tool steps/task | 2.56 | 4.36 | +70.3% |
| Filesystem-search calls/task | 5.68 | 4.96 | -12.7% |
| Tool-only steps/task | 13.64 | 0.20 | -98.5% |
| Mixed text-plus-tool steps/task | 0.00 | 8.12 | treatment activated |

## Interpretation

The five-trial evidence rejects visible progress narration as the primary cause of the Codex/DSH token and latency gap:

1. Suppressing Codex progress narration worked mechanically, reducing progress text by 98.6%, but it did not improve resolution, total tokens, or latency. Mean total tokens rose 1.0% and latency rose 15.4%.
2. Forcing short DSH narration increased visible progress text, yet total tokens fell 40.4% and latency fell 24.8%. The treatment changed the entire trajectory: D3-S-N used fewer model rounds and more multi-tool steps per round.
3. Therefore the relevant mechanism is orchestration behavior induced by the prompt, not the token cost of narration characters themselves.
4. The original D3-S result of 2/5 was a stochastic low draw. Across five trials D3-S ranged from 2/5 to 4/5 and resolved 17/25 overall.
5. D3-S-N's 18/25 versus D3-S's 17/25 is too small and unstable to establish an accuracy advantage. The paired arms disagreed on only five rollouts, split 3 versus 2.

## Task stability

| Task | C3-SF | C3-SF-Q | D3-S | D3-S-N |
|---|---:|---:|---:|---:|
| django-11239 | 1/5 | 3/5 | 0/5 | 0/5 |
| django-12209 | 5/5 | 4/5 | 3/5 | 3/5 |
| django-12741 | 5/5 | 5/5 | 5/5 | 5/5 |
| django-13109 | 5/5 | 5/5 | 5/5 | 5/5 |
| django-13741 | 5/5 | 4/5 | 4/5 | 5/5 |

This task concentration is the main remaining validity limit. A larger set of distinct tasks is required before claiming that either narration policy improves general coding accuracy.

## Cost

- Trial 1: $0.3385 metered DSH API usage.
- Trial 2: $0.3488.
- Trial 3: $0.3198.
- Trial 4: $0.3249.
- Trial 5: $0.3220.
- **Total: $1.6539.**

Codex was run through the existing authenticated plan and recorded as $0 direct API usage by this harness.
