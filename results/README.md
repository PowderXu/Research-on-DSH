# Result provenance

- `reference/retrieval_v2`: 246-question deterministic retrieval evaluation,
  normalized to factual qrel/link slices.
- `reference/agent_pilot5/normalized`: current four-arm, final-source-scored
  pilot report.
- `reference/agent_pilot5/{four_arm,dsh_three_arm,codex_fastctx}`: original
  imported reports and raw Codex traces.
- `optimization`: SkillOpt development trajectories and selected skill files.
- `diagnostics`: failed gates and legacy development analyses.

Imported artifacts may contain absolute paths from the machine on which they
were produced. Those strings are provenance only; active code, configuration,
and default commands do not depend on them.
