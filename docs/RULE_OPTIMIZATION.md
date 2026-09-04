# Aspect-rule optimization

## Purpose

The benchmark needs question-specific criteria for judging final agent answers.
Following BRIGHT-Pro's fine-grained design, these criteria are called
**aspects**: atomic, weighted answer requirements tied to evidence. Unlike
BRIGHT-Pro, DocsQA-Repo does not yet have human-authored aspects for every
question. It therefore generates **silver aspects** with an LLM from one shared
general rule.

Only that general rule is optimized. The process does not optimize an answer
agent, a retrieval skill, a plugin, model weights, per-question prompts, or the
deterministic WAC formula.

## Assumption and optimization signal

The weak-supervision assumption is that most GitHub discussions marked
answered have an accepted answer that is correct and sufficiently complete
when read with the internal documentation it explicitly cites. The normalized
source answer is therefore treated as a noisy positive, not human gold.

For each record, `gpt-5.6-luna` applies the current shared rule once. It returns:

- atomic aspects with importance and criticality;
- mappings to existing requirement, claim, and local evidence IDs; and
- `full`, `partial`, or `none` source-answer coverage for each aspect.

Deterministic code validates all IDs, requires every critical user requirement
to have a critical aspect, and gives document evidence no credit unless it has
a local path. It then computes

```text
coverage = sum(aspect importance * support value) / sum(aspect importance)
support value: full = 1, partial = 0.5, none = 0
```

A record passes when the aspect schema is valid and coverage is at least 0.80.
The optimization target is validation pass rate. A frozen rule is eligible for
later answer evaluation only when at least 80% of validation records pass.
This is a weak-supervision compatibility criterion, not proof of human
agreement.

## Pipeline

```text
467 normalized source records
          |
          +--> 60% train (280) ----------+
          |                               |
          |                         Luna applies rule
          |                               |
          |                     deterministic pass/score
          |                               |
          |                       Sol proposes bounded
          |                       ADD/DELETE/REPLACE edits
          |                               |
          +--> 20% validation (94) <------+  keep only improvement
          |
          +--> freeze best global rule
                         |
                         +--> 20% held-out test (93), once
                         |
                         +--> generate aspects for all 467 records
                                      |
                                      +--> deterministically bind exact
                                           evidence IDs from mapped claims
                                      |
                                      +--> freeze for agent scoring
```

The split is deterministic and project-stratified. Duplicate source questions
stay together. Train trajectories are visible to the optimizer, validation
selects the best rule, and test feedback is never returned to optimization.
The split exists only for rule optimization; evaluated zero-shot agents receive
no benchmark answers, aspects, or optimization examples.

## SkillOpt integration

The implementation uses Microsoft SkillOpt 0.2.0 as a library instead of
reimplementing its optimizer. The custom environment is
[`evaluation/rule_optimization/adapter.py`](../evaluation/rule_optimization/adapter.py).
It supplies the dataset loader, Luna rollout, deterministic score, and
conversation trace expected by SkillOpt. The thin wrapper in
[`run.py`](../evaluation/rule_optimization/run.py) registers the environment in
the process-local registry exposed by SkillOpt 0.2.0; it does not modify or
vendor SkillOpt.

The models have separate roles:

| Role | Frozen model | Function |
|---|---|---|
| Target/evaluator | `gpt-5.6-luna`, medium reasoning | Apply the current rule to one source QA and produce aspects plus source coverage |
| Optimizer | `gpt-5.6-sol`, xhigh reasoning | Reflect across scored train trajectories and propose general rule edits |
| Deterministic scorer | Python | Validate IDs and compute coverage/pass without an LLM |

SkillOpt's validation gate keeps an edited rule only when its validation pass
rate is higher than the current rule's. The full configuration is
[`config.yaml`](../evaluation/rule_optimization/config.yaml), and the editable
starting rule is [`initial_rule.md`](../evaluation/rule_optimization/initial_rule.md).

This follows SkillOpt's separation of a frozen target model, an optimizer model,
scored trajectories, bounded textual edits, and validation selection. It does
not claim that SkillOpt itself validates the benchmark assumption. See the
[SkillOpt paper](https://arxiv.org/abs/2605.23904), its
[official repository](https://github.com/microsoft/SkillOpt), and the
[official custom-benchmark guide](https://github.com/microsoft/SkillOpt/blob/main/docs/guide/new-benchmark.md).

## Handoff to final-answer evaluation

After the rule is selected and the held-out test is run:

1. freeze the exact rule, model settings, split manifest, and hashes;
2. apply that rule to each benchmark question to create frozen aspects;
3. bind exact evidence IDs mechanically from each aspect's mapped normalized
   claims, without changing the aspect text, claims, weights, or criticality;
4. give an answer judge the question, candidate agent answer, permitted local
   evidence, and frozen aspects;
5. let the judge assign aspect support labels; and
6. compute Weighted Aspect Coverage (WAC) deterministically from those labels
   and all frozen aspect weights.

This final score follows the standard weighted coverage formula used by
[BRIGHT-Pro](https://arxiv.org/abs/2605.04018) and its
[official evaluator](https://github.com/yale-nlp/Bright-Pro/blob/main/agentic_retrieval/scripts_evaluation/judge.py):
`sum(w_i * c_i) / sum(w_i)`, with `c_i` in `{0, 0.5, 1}`. Unlike the
weak-supervision source-coverage score above, final WAC grades an evaluated
agent answer against the frozen aspects.

The source answer is used while optimizing and generating aspects. It is never
shown as a reference answer to the evaluated agent. Aspects are never rebuilt
from an evaluated agent's answer.

The run wrapper materializes this handoff as `frozen_aspects.jsonl` after the
held-out rule test. It marks the artifact ineligible for agent judging if any of
the 467 records is missing or structurally invalid.

Evidence binding is deterministic because these values are opaque local
identifiers rather than semantic decisions. It uses only exact evidence IDs
already attached to the claims selected by the aspect constructor. It never
uses fuzzy path matching, live URLs, or outside knowledge. Raw model output is
retained separately so transcription errors and repairs remain auditable.

The current design uses one frozen target model for aspect construction and no
additional reviewer or generated counterexamples. Independent human validation
of source-answer completeness and generated aspect quality is future work.

## Reproduce

From the repository root:

```bash
evaluation/.venv/bin/pip install -e './evaluation[rule-optimization]'
evaluation/.venv/bin/pip install -e \
  'git+https://github.com/microsoft/SkillOpt.git@51d0a4d96e88558c84dee637f98e24e3fb2d1547#egg=skillopt'
export KBBENCH_OPENAI_ENV_FILE=/absolute/path/to/private.env
PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m rule_optimization.run \
  --config evaluation/rule_optimization/config.yaml
```

The source installation is necessary because the published SkillOpt 0.2.0
wheel omits Markdown prompts used by its aggregation stage. The commit is the
official v0.2.0 tag target; the benchmark neither copies nor rewrites those
prompts.

Generated split files, trajectories, candidate rules, and model outputs are
written to `results/runs/rule-optimization/` and remain untracked. A reported
run must retain its generated config, split manifest, initial rule, best frozen
rule, validation results, held-out test result, and token/latency summary.

## Completed reference run

The authorized Sol/Luna run completed on 2026-09-03.

| Stage | Result |
|---|---:|
| Initial-rule validation | 90/94 (95.74%) |
| Best-rule validation | 91/94 (96.81%) |
| Frozen held-out test | 88/93 (94.62%) |
| Raw all-record compatibility pass | 444/467 (95.07%) |
| Final frozen aspect records | 467/467 |

SkillOpt accepted one of five candidate updates; the selected rule came from
step 3. It added general instructions for separating independently scorable
procedural roles, preserving epistemic scope, and copying evidence identifiers
exactly. During the all-record freeze, deterministic claim binding corrected
44 evidence fields across the 22 records that had failed raw structural
validation. Locally reproduced image-derived text remains an allowed evidence
type under the dataset protocol. No API or JSON-parsing failures occurred.

The frozen artifact contains 1,926 aspects (mean 4.12 per question) and is
eligible for downstream agent judging. Optimization used 8,044,182 model
tokens across 911 calls; the held-out test used 553,359 tokens across 93 calls;
and all-record aspect generation used 2,832,463 tokens across 467 calls. Total
usage was 11,430,004 tokens across 1,471 calls. These results establish
compatibility with the weak-supervision assumption; they do not measure human
agreement or final agent-answer quality.
