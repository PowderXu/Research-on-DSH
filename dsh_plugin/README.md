# DeepSeek Harness plugin system

Everything specific to DSH is contained here:

```text
plugin/       native tool and three authored skill plugins
backend/      local HTTP service used by hybrid and Neo4j tools
dsh_home/     pinned headless DSH profile
harness/      common/model/per-arm evaluation patches and answer schema
agent_eval/   direct real-model runner and paired report generator
scripts/      DSH dependency consistency verification
```

The authored skills are `plugin/skills/{fs,hybrid,neo4j}/initial_skill.md`.
They are loaded directly during integrated evaluation. No SkillOpt optimizer,
training script, candidate-skill directory, or duplicated optimizer split is
part of this repository.

Install and verify the DSH runtime from the repository root:

```bash
npm ci --prefix dsh_plugin
npm run --prefix dsh_plugin setup:dsh-profile
npm run --prefix dsh_plugin verify:dsh
```

Agent execution also imports the retrieval implementation from
`evaluation/kbbench/`, but all DSH orchestration, profile configuration, service
adaptation, trajectories, and agent reporting stay in this directory.
