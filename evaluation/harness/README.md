# GitHub Docs agent harness

This directory holds the answer schema and DSH patches used by the GitHub Docs
agent evaluation. The three DSH arms share `github_docs_dsh_common.patch.yml`
and load exactly one arm-specific system or skill patch:

- `github_docs_fs_*`: filesystem search;
- `github_docs_hybrid_*`: BM25 + HNSW + RRF backend;
- `github_docs_neo4j_*`: hybrid seeds plus Neo4j expansion.

The corresponding external skill files live under
`dsh-techdocs-plugin/skills/`. SkillOpt changes those Markdown overlays without
editing the tool contract or backend implementation.

Codex uses `answer_schema.json` with
`codex-techdocs-plugin/skills/github-docs-fastctx/SKILL.md`; its runner is
`kbbench.github_docs_codex_fastctx_eval`.

Before a paired agent comparison, verify that every arm uses the same question
IDs, requested and actual model, answer schema, timeout, non-KB plugin
inventory, and machine. The primary ranking is the final ordered `sources`
array. Backend-visible documents are a diagnostic only.
