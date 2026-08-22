# DSH Technical-Docs KB

This is a zero-paid-API scaffold for a DeepSeek Harness technical-document retrieval bundle. It follows DSH's service-definition/provider/consumer capability pattern and intentionally separates technical resources from conversational memories.

## D3 capability composition

The installable package contains several independently loadable DSH plugins:

| Role | Package subpath | Responsibility |
|---|---|---|
| Service Definition | `@kbbench/dsh-techdocs/capability` | Provider-neutral `ctx.techdocs.retrieve()` seam |
| Provider | `@kbbench/dsh-techdocs/provider-repo` | Local HTTP adapter over the shared BM25/HNSW/link-graph backend |
| Routing Service Definition | `@kbbench/dsh-techdocs/routing-capability` | Provider-neutral `ctx.techdocsRouting.decide()` seam |
| Routing Provider | `@kbbench/dsh-techdocs/routing-rules` | Deterministic shared rule policy |
| Model-facing Consumer | `@kbbench/dsh-techdocs/tool-composite` | One `techdocs_composite` tool, schema-matched to the Codex MCP tool |
| Skill Consumer | `@kbbench/dsh-techdocs/skill-techdocs` | Thin `techdocs-research` playbook |
| Lifecycle Consumer | `@kbbench/dsh-techdocs/consumer-coding` | Automatic `agent/pre-step` routing, retrieval, and evidence injection |

The service and provider plugins are never selected by the model. A DSH preset loads them. The model may select the skill and composite tool, while the lifecycle consumer is invoked deterministically by DSH.

`D3-S` enables the service, repository provider, composite tool, and skill. `D3-A` enables the same components plus routing and the lifecycle consumer. Both keep the normal coding-agent plugin inventory available.

The legacy bundle entry point exposes two model-facing tools:

- `techdocs_search`: one composite, citation-ready retrieval operation;
- `techdocs_fetch`: bounded passage loading for URIs returned by search.

The bundle does not implement retrieval algorithms in Node. A local KB service owns OpenViking resource navigation, disk-backed dense+BM25 retrieval, optional Neo4j structural expansion, local reranking, and benchmark traces. Repository Markdown links and linked code are first-class, commit-pinned evidence rather than LLM-generated relationships. Graph expansion is disabled by default because it must earn its use on relation-matched evaluation rather than being assumed beneficial. See `SERVICE_CONTRACT.md` and `GRAPH_SCHEMA.md`.

## GitHub Docs arm-specific skills

The GitHub Docs three-arm evaluation uses external Markdown skills rather than
embedding trainable instructions in JavaScript:

| Arm | Plugin | Default artifact |
|---|---|---|
| Filesystem | `@kbbench/dsh-techdocs/skill-fs` | `skills/fs/initial_skill.md` |
| BM25 + HNSW | `@kbbench/dsh-techdocs/skill-hybrid` | `skills/hybrid/initial_skill.md` |
| Neo4j GraphRAG | `@kbbench/dsh-techdocs/skill-neo4j` | `skills/neo4j/initial_skill.md` |

Each entry accepts an administrator-provided `skillPath`. This is the isolated
candidate overlay used by SkillOpt; the plugin validates the arm, stable tool
names, required sections, and size before registering it. Corpus routes,
frontmatter, and `index.md` children remain retrieval evidence and are not copied
into the skill prompt.

The D3 automatic route and the paired Codex `UserPromptSubmit` hook consume the same `routing-rules.json`. This prevents the harnesses from silently using different task classifiers.

## Development

```sh
npm test
DSH_HOME=../.dsh-trial node_modules/.bin/dsh --profile headless \
  --patch ../evaluation/harness/dsh_d3_mount_smoke.patch.yml --help
```

The pure contract tests need only Node.js. A live DSH test additionally requires the exact `0.1.0-rc.6` package family and a reachable KB service.

For a scored KB-only turn, apply `trial-kb-minimal.patch.yml` after the model-provider patch. It disables unrelated coding-agent tools and prompt sections while preserving `techdocs_search` and `techdocs_fetch`. The held-out stress ablation is recorded in `../results/dsh_kubernetes_stress_v1/report.json`.

## Install after the live runtime is available

```sh
dsh plugin --profile default add ./dsh-techdocs-plugin
dsh --profile default --dump-config
```

Do not run this treatment alongside the official OpenViking auto-recall bundle during a scored turn: both would inject overlapping retrieval context. The official bundle is a separate baseline under the same corpus, model, context budget, and machine.
