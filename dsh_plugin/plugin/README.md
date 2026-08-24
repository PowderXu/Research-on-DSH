# DSH DocsQA plugin

DocsQA is the public documentation-question-answering capability; it is not the
name of one retrieval algorithm. This package exposes that stable capability
while filesystem, hybrid, and Neo4j remain candidate implementations:

- `@kbbench/dsh-docsqa`: `docsqa_search`, `docsqa_fetch`, and optional
  `docsqa_expand` tools over the local GitHub Docs backend;
- `@kbbench/dsh-docsqa/skill-fs`: filesystem retrieval skill;
- `@kbbench/dsh-docsqa/skill-hybrid`: BM25 + HNSW retrieval skill;
- `@kbbench/dsh-docsqa/skill-neo4j`: hybrid plus explicit graph-expansion
  skill.

`cordis.patch.yml` installs one plugin group. Evaluation patches enable
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
scripts/clean-runtime.ts remove only obsolete top-level runtime bundles
skills/*/*.md           authored model instructions
lib/*.js                generated DSH runtime entries and shared chunks
lib/types/*.d.ts        generated consumer declarations
```

The build is deliberately two-stage. `tsc -b` is the only TypeScript
transformer and owns declarations in `lib/types/`; `tsdown` reads that emitted
JavaScript and creates the published ESM entries in `lib/`. The package's
`main`, `types`, and conditional `exports` never point at `src/`.

Continue with the [service contract](SERVICE_CONTRACT.md), the
[Neo4j graph schema](GRAPH_SCHEMA.md), and the
[system-level plugin design](../../docs/PLUGIN_DESIGN.md). The parent
[DSH README](../README.md#documentation-map) explains how their responsibilities
differ and which one to read first.
