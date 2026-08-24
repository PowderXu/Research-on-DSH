# DeepSeek Harness plugin system

Everything specific to DSH is contained here. The checked-in source and
generated/runtime boundaries are:

```text
dsh_plugin/
├── package.json                 DSH install/build/verification orchestration
├── plugin/
│   ├── src/                     typed Cordis plugins, tools, client, evidence
│   ├── test/                    TypeScript contract and registration tests
│   ├── skills/{fs,hybrid,neo4j}/ authored Markdown instructions
│   ├── lib/                     generated JS, declarations, shared chunks
│   ├── tsconfig.json            tsc-owned production emission
│   ├── tsconfig.test.json       strict no-emit source/test checking
│   ├── tsdown.config.ts         public ESM entry bundling only
│   ├── cordis.patch.yml         package insertion into the DSH profile
│   └── package.json             compiled exports and DSH peer contract
├── backend/                     local HTTP service for hybrid and Neo4j
├── dsh_home/                    pinned headless DSH profile
├── harness/                     common/model/per-arm patches and answer schema
├── agent_eval/                  real-model runner and paired report generator
└── scripts/                     typed build/dependency verification
```

The authored skills are `plugin/skills/{fs,hybrid,neo4j}/initial_skill.md`.
They are loaded directly during integrated evaluation. No SkillOpt optimizer,
training script, candidate-skill directory, or duplicated optimizer split is
part of this repository.

## Build and runtime workflow

The package follows DSH's tsc-first convention:

```text
src/*.ts --tsc--> lib/types/*.js + lib/types/*.d.ts
                       |
                       +--tsdown--> lib/index.js
                                    lib/skill-fs.js
                                    lib/skill-hybrid.js
                                    lib/skill-neo4j.js
                                    lib/<shared-chunk>.js
                                              |
                                              +--> headless DSH profile
```

- Relative source imports use explicit `.ts` extensions; TypeScript rewrites
  runtime imports during emission.
- `Config` is a Schemastery schema, `apply` receives a typed Cordis `Context`,
  and each `inject` list contains only the DSH service actually consumed.
- Every model-facing tool uses `defineTool`, typed parameters, a mandatory
  output schema/renderer, cooperative cancellation, and a declared timeout.
- DSH loads `lib/*.js`; consumers receive declarations from `lib/types/*.d.ts`.
  Source files and generated build metadata are not package entrypoints.

This matches the official DSH guidance for
[plugin structure](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/develop/basic/index.md),
[typed tools](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/tools/README.md),
and [package publication](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/develop/basic/publish.md).

Install and verify the DSH runtime from the repository root:

```bash
npm ci --prefix dsh_plugin
npm run --prefix dsh_plugin setup:dsh-profile
npm run --prefix dsh_plugin verify:dsh
```

`setup:dsh-profile` expands to three ordered operations: install the plugin's
development dependencies without running lifecycle scripts, build with
`tsc` then `tsdown`, and install the built local package into the headless
profile. `verify:dsh` checks compiled exports, declarations, DSH dependency
versions, the installed profile package, strict types, and all plugin tests.

Agent execution also imports the retrieval implementation from
`evaluation/kbbench/`, but all DSH orchestration, profile configuration, service
adaptation, trajectories, and agent reporting stay in this directory.
