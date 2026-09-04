# General-rule optimization

This module optimizes one global natural-language rule for constructing
question-specific answer-evaluation aspects. It does not tune an agent, alter
retrieval, or create per-question rules.

The normalized accepted source answer is weak supervision. For each record,
the frozen target model applies the current rule once to construct aspects and
rate how fully the source answer covers them. Deterministic code rejects invalid
IDs or missing critical-requirement coverage and computes importance-weighted
source-answer coverage. SkillOpt reflects on train trajectories, proposes
bounded edits to the shared rule, and accepts an edit only when validation pass
rate improves. A record passes when its deterministic weighted coverage is at
least 0.80; the frozen rule is eligible for use only when at least 80% of
validation records pass.

## Models and split

- target/aspect constructor: `gpt-5.6-luna`, medium reasoning;
- rule optimizer: `gpt-5.6-sol`, xhigh reasoning;
- data: the 467 accepted records in normalization work `normalization-v14`;
- split: deterministic, project-stratified 60% train, 20% validation, 20% held-out test;
- duplicate source records stay in one split;
- validation selects the rule; the wrapper opens test once only after the
  validation pass rate reaches 0.80.

The generated splits, trajectories, candidate rules, and test outputs are under
`results/runs/rule-optimization/` and are intentionally not committed.

## Run

Install the rule-optimization dependency from `evaluation/`:

```bash
python -m pip install -e '.[rule-optimization]'
python -m pip install -e \
  'git+https://github.com/microsoft/SkillOpt.git@51d0a4d96e88558c84dee637f98e24e3fb2d1547#egg=skillopt'
```

The second command uses the official v0.2.0 source commit because the
published wheel does not include the Markdown prompts required at runtime.
The local wrapper checks for those files before any model call.

Point the credential bridge at an env file containing `OPENAI_API_KEY`, then run
the official SkillOpt engine through the local registration wrapper:

```bash
export KBBENCH_OPENAI_ENV_FILE=/path/to/.env
python -m rule_optimization.run \
  --config evaluation/rule_optimization/config.yaml
```

For a low-cost integration check, override the run size without changing the
recorded benchmark configuration:

```bash
python -m rule_optimization.run \
  --config evaluation/rule_optimization/config.yaml \
  --cfg-options train.num_epochs=1 env.limit=6 \
  evaluation.sel_env_num=4 evaluation.test_env_num=4 env.workers=2
```

The frozen rule is written by SkillOpt to the configured output directory as
`best_skill.md`. After the held-out rule test, the wrapper applies that frozen
rule to all 467 normalized records. Exact evidence identifiers are then bound
mechanically from each aspect's mapped normalized claims; this changes no
semantic field and records every binding repair. The resulting
`frozen_aspects.jsonl` is the input expected by the DSH final-answer judge. A
paper result is valid only when all 467 aspect records are present and it
records the config, split manifest, initial rule, frozen rule, eligibility
report, and held-out test summary from the same run.
