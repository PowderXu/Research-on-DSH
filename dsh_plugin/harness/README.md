# DSH agent harness

This directory contains the common answer schema and matched DSH patches used
by integrated agent evaluation.

- `docsqa_dsh_common.patch.yml` disables unrelated tools for DocsQA.
- `docsqa_fs_system.patch.yml` enables DSH filesystem tools and the FS skill.
- `docsqa_hybrid_system.patch.yml` enables native DocsQA tools and the hybrid skill.
- `docsqa_neo4j_system.patch.yml` additionally exposes `docsqa_expand`.

Every arm returns the same JSON schema and runs through the same pinned headless
DSH profile. The per-arm skill may differ because its available tools differ.
For a fair run, keep question IDs, model, timeout, machine, and non-KB plugin
inventory identical.
