# DSH technical-docs plugin

This package provides the native DeepSeek Harness side of the benchmark:

- `@kbbench/dsh-techdocs`: `techdocs_search`, `techdocs_fetch`, and optional
  `techdocs_expand` tools over the local GitHub Docs backend;
- `@kbbench/dsh-techdocs/skill-fs`: filesystem retrieval skill;
- `@kbbench/dsh-techdocs/skill-hybrid`: BM25 + HNSW retrieval skill;
- `@kbbench/dsh-techdocs/skill-neo4j`: hybrid plus explicit graph-expansion
  skill.

`cordis.patch.yml` installs one isolated plugin group. Evaluation patches enable
the matching tool inventory and exactly one skill per arm. An administrator may
supply `skillPath`; the loader validates frontmatter, arm identity, required
tool names, required sections, and size before registration.

The Node plugin is deliberately thin. Retrieval and graph construction live in
the local Python backend, while DSH owns tool registration, skill selection,
context exposure, session traces, and agent execution.

```bash
npm test
```

See `SERVICE_CONTRACT.md`, `GRAPH_SCHEMA.md`, and
`../../docs/PLUGIN_DESIGN.md`.
