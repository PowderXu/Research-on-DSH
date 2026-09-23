# Frozen aspect annotations

The benchmark uses 1,926 fixed answer aspects for one pool of 467 questions.
The annotations are downloaded as `aspects.jsonl` from the pinned dataset
release. Each aspect describes an answer requirement, its importance and
criticality, and its evidence mappings.

Questions and reference answers remain separate files. The evaluator joins
annotation records by `question_id`; agents cannot access reference answers
or scoring annotations.

## Evaluation

Every system is judged against the same aspect descriptions, weights, and
evidence mappings. The LLM judge labels coverage of each aspect in the current
answer; deterministic code computes Weighted Aspect Coverage (WAC). Fixed
annotations do not mean preassigned answer scores or deterministic LLM judgments.

The evaluator is `dsh_plugin/agent_eval/aspect_judge.py`. It reads the released
annotations and cannot add aspects or change their weights during scoring.
Construction schemas, coverage checks, and evidence-binding helpers remain in
`evaluation/dataset_analysis/aspect_construction.py`; they are separate from
the final-answer judge. See [the evaluation protocol](EVALUATION_PROTOCOL.md).

## Annotation provenance and limits

The existing labels were prepared offline with `gpt-5.6-luna` from normalized
questions, reference answers, requirements, claims, and local evidence. Their
shared construction rule was selected in an earlier annotation experiment;
[archived preparation details](https://github.com/PowderXu/Research-on-DSH/blob/c886f9c/docs/RULE_OPTIMIZATION.md)
and the ignored local run artifacts preserve that history. Original construction
instructions remain in
`evaluation/dataset_analysis/rubrics/aspect_construction_initial.md`.

These are model-generated silver annotations, not expert-verified gold labels.
Structural validity does not establish semantic correctness or completeness.
The question pool has been inspected during development, so current results
remain exploratory. Earlier annotation cohorts do not partition today's dataset.
