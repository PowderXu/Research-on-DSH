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

The D3 automatic route and the paired Codex `UserPromptSubmit` hook consume the same `routing-rules.json`. This prevents the harnesses from silently using different task classifiers.

## Development

```sh
cd ..
npm run test:plugin
DSH_HOME=./dsh_home node_modules/.bin/dsh --profile headless --help
```

The pure contract tests need Node.js and the dependencies installed by the
standalone package's `npm install`. A live DSH trial additionally requires the
headless profile installation and reachable task-local KB service described in
the top-level `README.md`.

## Install after the live runtime is available

```sh
node_modules/.bin/dsh plugin --profile default add ./dsh-techdocs-plugin
node_modules/.bin/dsh --profile default --dump-config
```

Do not run this treatment alongside the official OpenViking auto-recall bundle during a scored turn: both would inject overlapping retrieval context. The official bundle is a separate baseline under the same corpus, model, context budget, and machine.
