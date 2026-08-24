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
npm ci --prefix dsh_plugin
npm run --prefix dsh_plugin setup:dsh-profile
npm run --prefix dsh_plugin verify:dsh
```

Run these commands from the repository root. The outer DSH install provides the
host-side peer graph; `setup:dsh-profile` installs this package's build tools,
builds it, and connects it to the pinned profile.

## TypeScript package layout

```text
src/index.ts            runtime plugin entry and Cordis configuration
src/tools.ts            typed DSH tool definitions
src/service.ts          abortable HTTP boundary
src/evidence.ts         scoped evidence normalization and rendering
src/skill-loader.ts     validated Markdown-skill registration
src/skill-*.ts          one public plugin entry per evaluation arm
test/*.test.ts          strict TypeScript tests
skills/*/*.md           authored model instructions
lib/*.js                generated DSH runtime entries and shared chunks
lib/types/*.d.ts        generated consumer declarations
```

The build is deliberately two-stage. `tsc -b` is the only TypeScript
transformer and owns declarations in `lib/types/`; `tsdown` reads that emitted
JavaScript and creates the published ESM entries in `lib/`. The package's
`main`, `types`, and conditional `exports` never point at `src/`.

See `SERVICE_CONTRACT.md`, `GRAPH_SCHEMA.md`, and
`../../docs/PLUGIN_DESIGN.md`.
