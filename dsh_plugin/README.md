# DeepSeek Harness plugin system

Everything specific to DSH is contained here. The checked-in source and
generated/runtime boundaries are:

```text
dsh_plugin/
├── package.json                 DSH install/build/verification orchestration
├── plugin/
│   ├── src/                     typed Cordis plugins, tools, client, evidence
│   ├── test/                    TypeScript contract and registration tests
│   ├── scripts/                 typed runtime-bundle cleanup
│   ├── skills/{fs,hybrid,neo4j}/ authored Markdown instructions
│   ├── data/{fs,hybrid,neo4j}/  local arm-owned corpora and databases
│   ├── lib/                     generated JS, declarations, shared chunks
│   ├── tsconfig.json            tsc-owned production emission
│   ├── tsconfig.test.json       strict no-emit source/test checking
│   ├── tsdown.config.ts         public ESM entry bundling only
│   ├── cordis.patch.yml         package insertion into the DSH profile
│   └── package.json             compiled exports and DSH peer contract
├── backend/                     local stores, KGGen adapter, and HTTP service
├── dsh_home/                    pinned headless DSH profile
├── harness/                     common/model/per-arm patches and answer schema
├── agent_eval/                  real-model runner, frozen-aspect judge, reports
├── tests/
│   ├── backend/                 Python store, graph, and service tests
│   └── agent_eval/              Python runner, judge, and report tests
└── scripts/                     typed build/dependency verification
```

Python test source is kept under `tests/`, separate from the backend and agent
implementation. The TypeScript package keeps its existing `plugin/test/`
directory. From the repository root, run DSH Python tests with:

```bash
evaluation/.venv/bin/python -m pytest -c evaluation/pyproject.toml dsh_plugin/tests
```

## Documentation map

The following files describe different boundaries of DocsQA. They are
human-facing design documents: DSH does not load them and changing their prose
does not change runtime behavior.

| Document | What it explains | Read it when |
|---|---|---|
| [`plugin/SERVICE_CONTRACT.md`](plugin/SERVICE_CONTRACT.md) | The wire contract between the TypeScript DSH adapter and the local Python backend: health, search, graph expansion, and evidence fetch endpoints; request fields; response envelope; URI scope; evidence budget; and error shape. | Implementing a new backend, changing a tool payload, or debugging an adapter/backend mismatch. |
| [`plugin/GRAPH_SCHEMA.md`](plugin/GRAPH_SCHEMA.md) | The Neo4j v2 representation: projects, documents, sections, text-enriched image units, KGGen entities/predicates/claims, typed structural relationships, indexes, degree bounds, and evidence provenance. | Changing graph ingestion or expansion while keeping graph behavior query-blind and evidence-backed. |
| [`../docs/PLUGIN_DESIGN.md`](../docs/PLUGIN_DESIGN.md) | The system-level architecture: DocsQA as the stable capability, filesystem/hybrid/Neo4j as candidate implementations, DSH plugin and skill roles, build/profile boundaries, tool flow, and evaluation invariants. | Understanding how the entire DSH package is composed or introducing another retrieval candidate. |
| [`../docs/RULE_OPTIMIZATION.md`](../docs/RULE_OPTIMIZATION.md) | The benchmark's question-specific aspect construction, shared-rule optimization, frozen-rule boundary, and WAC handoff. | Understanding how answer-evaluation aspects are created before agent scoring. |

These documents deliberately do not duplicate one another:

```text
PLUGIN_DESIGN.md       selects components and defines their responsibilities
        |
        +-- SERVICE_CONTRACT.md defines the DSH <-> backend boundary
        |
        +-- GRAPH_SCHEMA.md     defines the Neo4j-only data/traversal boundary
```

For a first read, start with `PLUGIN_DESIGN.md`, then read
`SERVICE_CONTRACT.md`. Read `GRAPH_SCHEMA.md` only when working on the Neo4j
candidate. The `../../docs/PLUGIN_DESIGN.md` spelling seen inside
`plugin/README.md` is just a relative path: from `dsh_plugin/plugin/`, go up to
`dsh_plugin/`, go up again to the repository root, and then enter `docs/`.

The actual runtime inputs are `plugin/package.json`, compiled `plugin/lib/*.js`
exports, `plugin/cordis.patch.yml`, the selected Markdown skill, and the matching
patch under `harness/`.

The authored retrieval skills are `plugin/skills/{fs,hybrid,neo4j}/initial_skill.md`.
They are loaded directly during integrated evaluation. They are never optimized
with the benchmark's separate aspect-rule optimizer.

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
- Before bundling, a typed cleanup step removes only tsdown-owned top-level
  `lib/*.js` artifacts. It preserves `lib/types` and TypeScript's incremental
  state, while preventing an obsolete hashed shared chunk from entering a
  later package.

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

`evaluation/kbbench/` supplies shared retrieval and scoring code and can be used
for no-model implementation checks. The paper's reported system evaluation is
`agent_eval/`: it runs the initialized `dsh_home/` profile and measures skill
loading, plugin/tool orchestration, multi-turn model behavior, final answers,
tokens, and end-to-end latency.

## Local plugin data

Each arm owns a separate runtime store below `plugin/data/`:

```text
data/<arm>/
├── documents/   exact normalized corpus text and generated .ignore
├── corpus/      prepared corpus, manifest, and evaluation splits
├── indexes/     embeddings and sparse/dense index caches
├── assets/      derived image assets
├── artifacts/   neutral retrieval units and optional KGGen graph JSON
├── traces/      local service and agent traces
└── database/    Neo4j files; created only for the neo4j arm
```

Every arm directory contains a negating `.gitignore`: Git stores the directory
placeholder, while corpora, model artifacts, databases, and traces remain local.
Prepare all three stores from the frozen normalized dataset:

```bash
for arm in fs hybrid neo4j; do
  PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
    -m dsh_plugin.backend.prepare_plugin_data \
    --arm "$arm" \
    --source-dataset evaluation/dataset/evaluation_data/normalized
done
```

Preparation materializes only the normalized corpus `rendered_text` at each
canonical `source_path`, plus a generated root `.ignore`. It does not copy raw
repository pages, assets, or metadata into the searchable workspace. The
optional `--source-documents` path records provenance only;
`--copy-documents` is a deprecated compatibility flag with no copying effect.
Existing canonical pages may be refreshed, but stale extra files or directories
are rejected, never deleted automatically. Use a clean destination or explicitly
relocate stale data before preparing it again.

The generated `.ignore` explicitly re-includes the corpus directories and pages
so official filesystem searches can traverse them beneath the Git-ignored
plugin-data directory. This is a search-visibility policy, not a security or
filesystem-access boundary: tool glob overrides can bypass file exclusions.
The exact on-disk inventory and content checks are therefore required as well.

This makes local image-derived text and all other searchable content identical
across filesystem, hybrid, and Neo4j. The hybrid and Neo4j paths additionally
emit the same query-blind neutral retrieval-unit and image-provenance records.
Image-derived text is already part
of its owning page before BM25/HNSW chunking; it is not indexed again as a
standalone image vector. Neo4j projects the occurrence and asset provenance
into the plugin-owned graph schema without changing the shared text index.

At startup, before model episodes, the integrated runner validates the workspace bytes
and inventory and performs a no-model health check through the installed
official DSH filesystem tools. It checks searchable and glob-visible document
counts and corpus-only search/read probes, without using questions, qrels, or
answer labels. Every arm writes the successful startup check to
`<output-dir>/preflight.json`; a failed check prevents model execution. This
startup work is excluded from per-episode agent latency and token metrics.

The filesystem-visibility repair must be evaluated in fresh matched runs under
`results/runs/agents/fs-visibility-fix/pilot/` and then
`results/runs/agents/fs-visibility-fix/full/`. Historical filesystem results
were affected by Git-ignore directory-skipping and are not a clean comparison
of retrieval algorithms; see [`../docs/RESULTS.md`](../docs/RESULTS.md).

KGGen is optional and runs offline in an isolated environment because its
dependency range conflicts with the benchmark embedding environment:

```bash
python3 -m venv dsh_plugin/.venv-kggen
dsh_plugin/.venv-kggen/bin/pip install \
  -r dsh_plugin/backend/requirements-kggen.txt
PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m dsh_plugin.backend.prepare_plugin_data \
  --arm neo4j \
  --source-dataset evaluation/dataset/evaluation_data/normalized \
  --run-kggen \
  --kggen-python dsh_plugin/.venv-kggen/bin/python
```

This writes `plugin/data/neo4j/artifacts/kggen_graph.json`. Extraction and
alias clustering come from KGGen; the adapter adds stable IDs and requires an
evidence unit plus source excerpt for every claim. It never reads questions,
answers, or qrels.

The default extraction model is `openai/gpt-5.6-luna`. The scalable adapter
batches retrieval units by project, checkpoints each raw KGGen response, uses
SemHash for entity/predicate aliases, and rejects claims whose cited unit has
only token overlap with the extracted relation. `--kggen-max-cost-usd` is an
optional operator guard; no cost ceiling is applied unless it is explicitly
passed.

Start the plugin-owned Neo4j service and ingest the prepared records:

```bash
docker compose -f dsh_plugin/backend/compose.neo4j.yml up -d
PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m dsh_plugin.backend.ingest_neo4j
```

This service binds browser port `7475` and Bolt port `7688` to `127.0.0.1`
only, avoiding the benchmark's older global Neo4j container and network
exposure of the local-development database. Its `/data` and `/logs` mounts resolve
to `plugin/data/neo4j/database` and `plugin/data/neo4j/traces/neo4j`. The ingest
command creates text/full-text indexes, all schema-v2 document/image-text nodes
and structural edges, and the semantic nodes and KGGen
claims when `kggen_graph.json` exists.
