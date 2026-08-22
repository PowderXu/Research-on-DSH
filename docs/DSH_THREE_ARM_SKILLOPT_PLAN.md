# DSH three-arm skill plugins and SkillOpt improvement plan

## Decision

The corrected experiment will contain two explicitly separate evaluation layers.

1. **Provider retrieval evaluation** calls each retrieval provider through a DSH-loaded plugin contract, without an LLM or skill. It measures the KB implementations themselves.
2. **DSH system evaluation** runs a real DSH agent with the tool inventory and skill appropriate to that arm. It measures how well the complete DSH composition uses the KB.

This separation is required. A natural-language skill cannot affect a deterministic search function that is invoked directly by Python. SkillOpt only becomes meaningful when a target agent receives the candidate skill, runs the task, and produces a scored trajectory.

The three system arms are:

| Arm | DSH retrieval capability | Arm-specific skill |
|---|---|---|
| `dsh_fs` | Official `@deepseek-ai/dsh-tool-fs-search` `glob`/`grep` tools plus bounded file reading | Filesystem and `index.md` navigation skill |
| `dsh_hybrid` | BM25 + HNSW + RRF technical-doc plugin, with metadata-aware indexing | Hybrid query, metadata, and evidence-selection skill |
| `dsh_neo4j` | Neo4j GraphRAG plugin using the same lexical/dense foundation plus typed traversal | Conditional graph-use and path-verification skill |

The final system comparison may use different optimized skills because the arms expose different retrieval affordances. A separate shared/neutral-skill calibration is still necessary to distinguish retrieval-backend effects from skill effects.

## Implementation status (2026-08-21)

The plugin-level system path and SkillOpt adapter are now implemented. This is no longer only a proposed layout.

| Component | Status | Evidence |
|---|---|---|
| Frozen split | Complete | 55 train, 27 validation, and the original 246 test questions; seed `20260821` |
| Three external skills | Complete | `dsh-techdocs-plugin/skills/{fs,hybrid,neo4j}/initial_skill.md` |
| DSH profiles | Complete | common patch plus arm patches in `evaluation/harness/` |
| Hybrid plugin service | Complete | BM25 + HNSW + RRF at `/v1/search` and evidence fetch at `/v1/fetch` |
| Neo4j plugin service | Complete | the same hybrid seed search plus explicit `/v1/expand` over typed, bounded relations |
| SkillOpt adapter | Complete | real DSH subprocess rollout, candidate overlay, trace capture, external qrel scoring, and fail-closed validation |
| Microsoft SkillOpt pin | Complete | vendored at commit `da06b157cb9878e378663ee1ecf429c83fe1a8f9` |
| Unit/contract tests | Complete | 57 Node tests and 15 Python tests |
| Full frozen real-model test | Not run | requires completion of arm-specific optimization and a deliberate final budget allocation |

The system evaluator now invokes DSH and its loaded tools. The older `kbbench/github_docs_three_arm_eval.py` remains a provider/direct-backend research runner and must not be used as evidence for the final DSH system comparison.

The explicit graph interface was also corrected during real-agent testing. When graph expansion is a separate `techdocs_expand` operation, `techdocs_search` no longer advertises an `allow_graph` argument that the backend would ignore. This keeps traversal observable and prevents the agent from mistaking seed retrieval for graph execution.

## Why the direct three-arm runner is not the system benchmark

`kbbench/github_docs_three_arm_eval.py` implements filesystem query planning, subprocess search, hybrid retrieval, Neo4j ingestion, Cypher expansion, and result ranking inside the evaluator. It does not load DSH or invoke the installed DSH plugin composition.

That runner is a useful direct-backend prototype, but it must not be described as a DSH plugin evaluation. Its reusable retrieval code should move behind provider plugins or backend services. The evaluator should retain only:

- dataset and split loading;
- invoking a pinned DSH profile or plugin benchmark endpoint;
- collecting ranked document IDs and traces;
- Recall, Hit, nDCG, latency, token, and failure measurement;
- paired significance tests and report generation.

## Correct runtime topology

### Layer 1: provider retrieval evaluation

```text
Frozen questions and qrels
          |
          v
Evaluation driver
          |
          v
DSH plugin host / deterministic benchmark workflow
          |
          v
ctx.techdocs.retrieve(request)
          |
          +-- filesystem provider
          +-- BM25 + HNSW provider
          `-- Neo4j GraphRAG provider
          |
          v
Typed ranked IDs + retrieval trace
          |
          v
Evaluator computes Recall / Hit / nDCG / p50
```

No model or skill participates in this layer. It is the clean architecture comparison.

### Layer 2: DSH system and skill evaluation

```text
Frozen user question
          |
          v
Pinned DSH headless agent
  + common model and context settings
  + arm-specific retrieval plugins
  + candidate arm-specific skill
          |
          v
Real tool trajectory
          |
          +-- tool requests and results
          +-- ranked/visible evidence IDs
          +-- final answer and citations
          +-- latency and token usage
          `-- failures and abstention
          |
          v
External scorer (qrels never enter agent context)
```

This is the layer optimized by SkillOpt.

## Plugin composition

The current `TechdocsCapability` and `ProviderSelector` are the correct foundation. The retrieval algorithms should be supplied as independently loadable providers rather than selected by `if arm == ...` in the evaluator.

### Shared plugins

| Plugin | Responsibility |
|---|---|
| `techdocs-capability` | Provider-neutral `ctx.techdocs.retrieve()` and typed result contract |
| `techdocs-observer` | Records requests, ranked evidence, visible evidence, failures, and timings |
| `techdocs-corpus-metadata` | Parses the pinned Markdown/frontmatter representation shared by all arms |
| `techdocs-benchmark-driver` | Deterministic no-model invocation used only by provider evaluation |
| `techdocs-skill-loader` | Loads one Markdown skill from an administrator-selected path and registers it with `ctx.skills` |

### Provider plugins

| Provider | Implementation boundary |
|---|---|
| `provider-fs` | Reuse the official DSH filesystem-search package and its packaged ripgrep core. Do not maintain a second custom subprocess parser in the evaluator. |
| `provider-hybrid` | Own query embedding, BM25, HNSW, RRF, metadata boosts/filters, passage selection, and ranked results. |
| `provider-neo4j` | Own the official `neo4j-graphrag` retriever, graph query, typed traversal, and provenance. Graph ingestion is a separate build command, not evaluator logic. |

### DSH profiles

Each scored profile keeps the same DSH version, target model, system prompt outside the skill, context limits, timeouts, and result limits.

```text
github-docs-fs
  common DSH QA composition
  @deepseek-ai/dsh-tool-fs-search
  bounded read tool
  techdocs-observer
  skill-fs

github-docs-hybrid
  common DSH QA composition
  techdocs-capability
  provider-hybrid
  techdocs-search/fetch consumer
  techdocs-observer
  skill-hybrid

github-docs-neo4j
  common DSH QA composition
  techdocs-capability
  provider-neo4j
  techdocs-search/expand/fetch consumer
  techdocs-observer
  skill-neo4j
```

The provider retrieval layer should use one common request/result schema. The system layer intentionally exposes the natural tool inventory of each deployable arm; it therefore measures whole-system performance, not only retriever quality.

## Corpus metadata is evidence, not skill text

The GitHub Docs corpus contains important structured evidence in every page's frontmatter and especially in `index.md` pages. The skill must teach the agent how to use that evidence, but it must not copy the complete route tree into the prompt.

Putting thousands of page titles and routes in a skill would:

- become stale whenever the repository changes;
- consume context on every question;
- encourage memorized routing rather than evidence-backed retrieval;
- let SkillOpt overfit to specific document paths;
- blur the boundary between the corpus/index and the agent policy.

### Metadata to parse at the pinned repository revision

At minimum, preserve:

| Source | Fields or relations |
|---|---|
| File identity | canonical document ID, repository path, commit, whether the file is `index.md` |
| Frontmatter | `title`, `shortTitle`, `intro`, `contentType`, `layout`, `versions`, `category`, `includedCategories` |
| Routing | `children`, parent directory/index, `introLinks`, relevant carousel targets |
| Aliases | `redirect_from` mapped to the canonical document |
| Sections | heading text, anchor, ordering, variant-conditioned flag |
| Markdown links | source page/section, target page/anchor, anchor text, local link context |
| Composition | reusable IDs, variables, and version/Liquid conditions |
| Exact technical signals | CLI flags, environment variables, API paths, configuration keys, and code-shaped identifiers |

All arms receive the same source facts at the same commit, but each architecture represents them differently:

- the filesystem arm reads raw Markdown/frontmatter and uses `index.md` files as navigational documents;
- the hybrid arm adds normalized metadata fields to lexical/dense indexing and uses metadata as soft boosts or explicit user-supported filters;
- the Neo4j arm materializes routes, sections, links, reusables, aliases, versions, and code entities as typed nodes/relationships with provenance.

Metadata must not become an irreversible routing gate. If a route or version hypothesis is wrong, ordinary lexical/dense retrieval must remain able to recover the relevant page.

## Skill artifacts

The canonical trainable artifact for every arm will be a Markdown file. Instructions must no longer be embedded only as a JavaScript string in `skill-techdocs.mjs`.

Implemented source layout:

```text
dsh-techdocs-plugin/
  skill-loader.mjs
  skills/
    fs/initial_skill.md
    hybrid/initial_skill.md
    neo4j/initial_skill.md

kbbench/skillopt_github_docs/
    adapter.py
    credentials.py
    dataloader.py
    rollout.py
    scorer.py
    validation.py

evaluation/skillopt/
  github_docs_v2_split/
    train/items.json
    val/items.json
    test/items.json
    manifest.json
  github_docs_dsh/
    configs/
      fs.yaml
      hybrid.yaml
      neo4j.yaml

scripts/
  skillopt_github_docs_train.py
  skillopt_github_docs_eval.py
```

The loader reads an administrator-controlled `skillPath` during profile startup, parses the stable identity metadata, and registers the body through `ctx.skills.register`. SkillOpt candidates are mounted through an isolated temporary overlay. A rollout must never edit the installed plugin or the source skill in place.

### Protected and trainable regions

Each skill should use this shape:

```markdown
---
name: github-docs-<arm>
description: <stable activation description>
arm: <fs|hybrid|neo4j>
version: 1
---

# GitHub Docs retrieval

## Contract

Stable tool names, evidence rules, and safety constraints.

## Strategy

The SkillOpt-trainable retrieval and verification instructions.

## Failure recovery

The SkillOpt-trainable recovery and abstention instructions.
```

The validation hook must reject a candidate that:

- changes `name`, `arm`, stable tool names, or the evidence/citation contract;
- exceeds the configured skill token/line limit;
- contains secrets or external network instructions;
- includes a held-out test query, qrel document ID, or copied accepted answer;
- asks the agent to bypass tool-call, scope, timeout, or evidence rules;
- changes any backend or evaluator setting.

Only strategy and recovery language is optimized.

## Initial skill design by arm

### Filesystem and `index.md` navigation skill

The initial filesystem skill should teach the model to:

1. Preserve exact identifiers from the question, including flags, paths, API names, environment variables, and quoted error text.
2. Use a likely top-level `content/<domain>/index.md` and progressively narrower `index.md` files when the domain is clear.
3. Read frontmatter `children`, `introLinks`, title, short title, intro, category, and redirect aliases as routing clues.
4. Search exact phrases/identifiers with the official `grep` tool; avoid one broad regex that produces hundreds of unrelated matches.
5. Open the candidate page and relevant heading before treating a grep match as evidence.
6. Treat `index.md` primarily as a router unless it contains the answer itself.
7. Fall back from a wrong route to repository-wide search rather than repeatedly exploring the same subtree.
8. Cite the canonical page rather than an obsolete `redirect_from` alias.

The skill must not contain a hardcoded list of all GitHub Docs domains. The live `index.md` structure is the source of truth.

### BM25 + HNSW skill

The initial hybrid skill should teach the model to:

1. Form a concise retrieval query while preserving exact identifiers, product/version cues, and the user's actual constraint.
2. Use titles, short titles, intros, headings, routes, content types, versions, and aliases as retrieval signals.
3. Treat inferred route/version metadata as a soft preference unless the user explicitly constrained it.
4. Inspect the returned section and evidence signals rather than trusting only the document title.
5. Make at most one materially different reformulation when the first evidence set is weak.
6. Fetch only URIs returned by search and prefer the smallest passage that answers the question.
7. Abstain when the evidence does not support the requested claim.

The skill should not attempt to manually reproduce BM25, vector search, or RRF; those algorithms belong to the provider.

### Neo4j GraphRAG skill

The initial graph skill should teach the model to:

1. Start from lexical/dense seed retrieval rather than performing unbounded traversal.
2. Request graph expansion only for a relationship-bearing information need: linked prerequisites, configuration-to-procedure, version/variant dependencies, shared reusable content, redirects, or multi-page evidence.
3. Prefer contextual Markdown-link edges whose anchor text, source section, and local context match the question.
4. Use route, reusable, and shared-code edges only when their degree is bounded and their provenance is visible.
5. Treat high-degree route/category nodes as weak evidence, not proof of relevance.
6. Verify every expanded page against its actual passage before citing it.
7. Stop after the bounded expansion when it adds no relevant evidence; do not use graph traversal merely because the backend supports it.
8. Preserve the path explanation in traces: seed page, edge type, target page, and supporting context.

Community detection is not added to this skill or backend unless a separate benchmark demonstrates broad synthesis questions that local typed traversal cannot answer.

## SkillOpt integration

Use the official [Microsoft SkillOpt](https://github.com/microsoft/SkillOpt) research workflow, pinned to an exact commit. SkillOpt treats the Markdown skill as trainable text, obtains target-agent trajectories, proposes bounded edits, and accepts candidates through held-out validation gating.

SkillOpt is vendored at the pinned commit above and is loaded by the two wrapper scripts. The project runtime is `../.venv/bin/python`; `.venv-skillopt` is not the active runtime. The wrappers register the local `github_docs_dsh` adapter without modifying the vendored SkillOpt source.

SkillOpt 0.2.0 currently has a configuration-flattening conflict when an inherited structured YAML contains an `env:` mapping and the flattened configuration also expects `env` to be a scalar adapter name. The checked-in arm configurations are intentionally flat YAML files to avoid that upstream incompatibility.

### Custom benchmark adapter

Following SkillOpt's [new benchmark interface](https://github.com/microsoft/SkillOpt/blob/main/docs/guide/new-benchmark.md), add a `github_docs_dsh` environment with:

1. `SplitDataLoader`: reads the frozen GitHub Docs train/validation/test JSONL files.
2. `rollout`: mounts one candidate skill overlay, launches the selected DSH headless profile, records the complete tool/answer trajectory, and removes the overlay.
3. `EnvAdapter`: exposes build, rollout, task-type, and inherited reflection behavior.
4. `scorer`: normalizes evidence IDs and computes task scores outside the agent.
5. one YAML configuration per arm, all inheriting the same base target-model and optimizer settings.

Every non-empty rollout must persist the conversation and a structured trace containing:

```text
question_id
corpus_revision
harness/profile/model identifiers
skill SHA-256 and SkillOpt step
tool calls, arguments, statuses, and latency
ranked IDs returned or made visible
IDs actually opened/fetched
graph paths, when present
final citations
input/output/cached tokens
wall-clock latency
hard score, soft score, and failure reason
```

The qrels and score are appended only after the DSH process finishes. They are never placed in the target model's context.

### Independent optimization

Run three separate SkillOpt jobs:

```text
initial_skill_fs      -> best_skill_fs
initial_skill_hybrid  -> best_skill_hybrid
initial_skill_neo4j   -> best_skill_neo4j
```

Do not optimize one universal skill across mixed arms. A rule that is useful for `grep` can be harmful for dense retrieval, and graph-specific instructions are meaningless when graph tools are absent.

All jobs must use:

- the same target model and reasoning effort;
- the same train/validation/test question IDs;
- the same maximum task/tool time;
- the same skill-size limit and candidate edit budget;
- the same optimizer model unless cost evidence requires a documented change;
- validation gating enabled;
- no test evaluation during candidate selection.

### SkillOpt scoring and gate

For each task, record:

```text
hard = 1 only when the run is protocol-valid and Hit@10 = 1

soft = 0.50 * nDCG@10
     + 0.30 * Recall@10
     + 0.20 * Hit@10
```

The aggregate validation gate is lexicographic:

1. reject protocol violations, leakage, missing traces, timeouts, or malformed output;
2. do not accept a statistically or materially worse mean retrieval soft score;
3. require improvement in the registered primary validation score;
4. use lower p50 end-to-end latency, fewer tool calls, and fewer tokens only as tie-breakers;
5. never exchange a meaningful accuracy regression for lower latency or token use.

Because GitHub Docs questions can have one qrel, report nDCG and Recall separately even if they correlate with Hit. Category-level metrics are mandatory; one overall scalar is used only for SkillOpt's validation gate.

## Evaluation categories

Question categories must be assigned from real corpus/question evidence before optimization and frozen with the split.

| Category | Required behavior |
|---|---|
| `direct_exact` | Exact name, error, flag, API route, or phrase should locate one page |
| `semantic_paraphrase` | User wording differs substantially from documentation wording |
| `index_route_navigation` | `index.md` hierarchy or frontmatter children materially narrows the target area |
| `redirect_or_anchor` | Correct resolution requires an obsolete path, redirect, or section anchor |
| `version_or_variant` | Product/version/Liquid condition affects which passage is valid |
| `reusable_composition` | Relevant evidence is inserted through reusable Markdown content |
| `linked_multi_page` | Answer needs multiple pages connected by explicit contextual links |
| `typed_graph_relation` | A route, reusable, code entity, dependency, or procedure relation is necessary and query-relevant |

Do not label a question as graph-required merely because the gold page has a graph edge. A graph category needs gold evidence showing that relationship traversal is part of the information need.

The current public GitHub Discussions-derived set is useful, but many labels identify one page. Before claiming a GraphRAG advantage, the benchmark must contain enough real multi-page or relationship-bearing questions with multi-document qrels.

## Experimental matrix

### Provider benchmark: KB architecture

Run the complete frozen test set through:

| Cell | Model | Skill | Provider |
|---|---|---|---|
| `P-FS` | none | none | filesystem provider |
| `P-HYBRID` | none | none | BM25 + HNSW provider |
| `P-NEO4J` | none | none | Neo4j GraphRAG provider |

This establishes the actual retrieval performance and must be completed before system claims.

### DSH system benchmark: skill and orchestration

For causal interpretation, use three skill states:

| State | Meaning |
|---|---|
| `S0-neutral` | Minimal logically equivalent guidance; only tool names differ where required |
| `S1-expert` | The initial arm-specific skills described above |
| `S2-skillopt` | The validation-selected `best_skill.md` for that arm |

The full factorial has nine cells (`3 arms x 3 skill states`). If the real-model budget cannot support all nine on the full test set:

1. run all skill states on train/validation during development;
2. run `S0-neutral` and `S2-skillopt` for every arm on the frozen test set;
3. add `S1-expert` test cells only if the remaining budget permits;
4. never substitute mock results for the final benchmark.

The primary deployment comparison is the three `S2-skillopt` cells. The `S0` cells disclose how much of the difference comes from the provider versus the optimized skill.

## Ranking and metric semantics

The three system arms do not naturally emit evidence in the same way, so the observer must define ranking before any run.

- Filesystem: ranked IDs are canonical pages in the order they first become visible in successful `grep` results, with pages explicitly opened/read recorded separately. Duplicate pages collapse at first occurrence.
- Hybrid: ranked IDs are the provider's returned page order; fetched pages are recorded separately.
- Neo4j: ranked IDs are the post-expansion provider order; every graph-added result records its seed and edge path.

Report both:

- `visible_ranked_ids`: evidence the model could see;
- `consumed_ids`: pages the model explicitly opened, fetched, or cited.

Primary retrieval metrics use `visible_ranked_ids`. A secondary evidence-use analysis uses `consumed_ids` and final citations.

### Required metrics

For every arm, skill state, and category report:

- Recall@1, @5, @10, and @20;
- Hit@1, @5, @10, and @20;
- nDCG@10;
- p50 and p95 provider latency;
- p50 and p95 end-to-end DSH latency;
- input, cached-input, and output tokens per question;
- tool calls and failures per question;
- answer/citation score when final-answer evaluation is enabled.

Provider latency and end-to-end latency must not be mixed. Skill selection, model inference, tool serialization, and repeated calls are part of system latency but not provider latency.

## Fairness controls

Freeze and record:

- GitHub Docs commit and normalized corpus hash;
- question/qrel manifest hash and split IDs;
- DSH, plugin, Node, Python, Neo4j, and `neo4j-graphrag` versions;
- target and optimizer model identifiers;
- embedding model and index parameters;
- top-k/result/token/tool-call limits;
- warm/cold cache protocol and arm execution order;
- skill content and SHA-256;
- SkillOpt configuration, seed, accepted/rejected edits, and validation history.

The candidate skill may change only agent instructions. It may not change the tool schema, backend settings, corpus, qrels, evaluator, call limits, or score calculation.

Use alternating or randomized arm order per question to reduce cache and temporal effects. Use paired bootstrap confidence intervals over identical question IDs.

## Budget plan

The previously stated maximum paid-model budget is **USD 20**. Index construction, BM25/HNSW retrieval, local embeddings, Neo4j Community, and deterministic plugin tests should remain local and cost zero paid API dollars.

A practical initial allocation is:

| Work | Maximum |
|---|---:|
| Filesystem skill optimization | $4.50 |
| Hybrid skill optimization | $4.50 |
| Neo4j skill optimization | $4.50 |
| Frozen real-model final comparison | $5.00 |
| Failure/retry reserve | $1.50 |

Before training, run one real, unscored cost-calibration question per profile and calculate the affordable rollout count from actual tokens. Do not estimate cost only from prompt length. The final benchmark must use real model calls; deterministic mock runs are allowed only for adapter and trace validation and must never enter the results.

If the budget is too small, reduce the number of optimization steps or use a smaller frozen train subset. Do not inspect or tune on the frozen test set.

## Real-agent pilot results

These are engineering validation runs, not the final benchmark. They used `gpt-5-mini` with low reasoning and real DSH sessions.

The matched system-performance artifact is `results/github_docs_dsh_system_paired5_v1/report.md`, with machine-readable `report.json` and `per_query.jsonl` beside it. It aggregates only the initial-skill train rollouts and baseline-selection rollouts; candidate-selection trajectories are excluded.

| Run | Result | DSH tool path | Total agent tokens | End-to-end latency |
|---|---:|---|---:|---:|
| Filesystem, question `122713` | Hit@10 `0` | `skill → grep → read → read → read` | 53,080 | 22.46 s |
| Hybrid, question `122713` | Hit@10 `0` | `skill → techdocs_search` | 6,083 | 18.65 s |
| Neo4j, linked question `26686` | Hit@10 `1`, Recall@10 `0.5`, nDCG@10 `0.2641` | `skill → techdocs_search → techdocs_fetch` | 13,777 | 28.13 s |

The first hybrid trial asked for only six results even though the tool supported ten; the qrel appeared at backend rank eight. The initial hybrid and Neo4j skills now explicitly request `limit: 10`, matching the metric contract.

The graph profile did not invoke `techdocs_expand` on question `26686` because the answer was already supported by its seed results. This is intended conditional behavior. A later graph-opportunity trial exposed and fixed the misleading no-op `allow_graph` search argument described above. The model still declined explicit traversal because the retrieved JWT/private-key pages directly answered the question; forcing traversal only to improve a citation-derived qrel would make the operational policy worse.

The official SkillOpt micro-pilots used the same bounded view of three training and two validation questions for every arm. Because SkillOpt evaluates both the baseline and candidate on validation, each job ran seven DSH episodes: three train, two baseline-selection, and two candidate-selection episodes.

| Arm | Train hard / soft | Baseline validation soft | Candidate validation soft | Gate | DSH-agent tokens | Optimizer tokens | Wall time |
|---|---:|---:|---:|---|---:|---:|---:|
| Filesystem | `0.0000 / 0.0000` | `0.4077` | `0.0000` | Reject | 1,453,822 | 4,599 | 212.3 s |
| Hybrid | `0.6667 / 0.4722` | `0.5000` | `0.5000` | Reject tie | 239,294 | 10,132 | 194.6 s |
| Neo4j | `0.6667 / 0.5526` | `0.5000` | `0.5000` | Reject tie | 139,289 | 9,853 | 124.0 s |

All three `best_skill.md` files are byte-identical to their corresponding initial skills. The filesystem candidate materially regressed validation. The hybrid candidate mostly duplicated the existing search/fetch checklist. The Neo4j candidate did not improve the registered validation score. Retaining the initial skills is the correct evidence-based result; these micro-pilots validate independent optimization and gating but do not establish a skill lift.

The filesystem token total is dominated by 1,238,784 cached input tokens from repeated broad repository-tool context. That large difference is a system-level observation, not a provider-retrieval result. The Neo4j micro-pilot made no `techdocs_expand` calls on this five-question view, so it cannot be used to estimate graph-expansion lift; the final link-specific evaluation must contain discriminative traversal cases.

### Existing-qrel limitation

The public questions are real GitHub Community questions, and their qrels are resolved from documentation links in accepted answers. They are not exhaustive human relevance judgments against the current pinned corpus. Documentation can move or change after the discussion was answered.

For example, question `122713` links to historical cancellation anchors that now resolve to a general “get started” page, while the current cancellation/refund evidence lives on different pages. Question `49521` asks about a JWT signature failure, but its single accepted-answer qrel is an installation-access-token page; the agent answered from the current JWT and private-key pages and therefore scored zero.

Consequences:

- qrel Hit/Recall/nDCG remain valid measures of reproducing accepted-answer citations;
- they are not, by themselves, complete measures of current answer correctness;
- SkillOpt must not be encouraged to memorize individual qrels;
- final reporting must separate retrieval score from answer/citation validity and disclose qrel drift;
- a graph claim should be based on the frozen `linked_multi_page`/relationship subsets plus inspected graph paths, not only the overall mean.

## Reproducible commands

Materialize the frozen split:

```bash
../.venv/bin/python scripts/materialize_github_docs_skillopt_split.py
```

Run the contract tests:

```bash
(cd dsh-techdocs-plugin && npm test)
../.venv/bin/python -m pytest -q \
  tests/test_github_docs_dsh_system_report.py \
  tests/test_skillopt_github_docs.py \
  tests/test_github_docs_three_arm_eval.py
```

Run a real arm evaluation. The API key is parsed as data from the configured file; the file is never sourced as shell code:

```bash
KBBENCH_OPENAI_ENV_FILE=/path/to/profile-agent/.env \
../.venv/bin/python scripts/skillopt_github_docs_eval.py \
  --config evaluation/skillopt/github_docs_dsh/configs/hybrid.yaml \
  --skill dsh-techdocs-plugin/skills/hybrid/initial_skill.md \
  --split train \
  --cfg-options test_env_num=1 out_root=results/hybrid_real
```

Run the bounded SkillOpt integration pilot:

```bash
KBBENCH_OPENAI_ENV_FILE=/path/to/profile-agent/.env \
../.venv/bin/python scripts/skillopt_github_docs_train.py \
  --config evaluation/skillopt/github_docs_dsh/configs/hybrid.yaml \
  --train_size 3 --batch_size 3 --minibatch_size 3 \
  --merge_batch_size 3 --sel_env_num 2 --num_epochs 1 --limit 3 \
  --out_root results/skillopt_github_docs_hybrid_microopt
```

`--limit 3` is necessary for a micro-pilot because SkillOpt requires `train_size` to match the loaded training-pool size. It creates an in-memory bounded view and does not rewrite the frozen JSON files. Remove both `--limit 3` and the `train_size=3` override for the configured full training split.

## Implementation sequence

1. **Freeze protocol.** Save corpus revision, question/qrel hashes, categories, splits, metric definitions, and profile invariants.
2. **Extract the provider boundary.** Move filesystem, hybrid, and Neo4j retrieval behind provider plugins/services; remove retrieval algorithms from the evaluator.
3. **Create the common trace contract.** Ensure every provider and DSH tool trajectory emits canonical ranked IDs and timing fields.
4. **Build corpus metadata once per revision.** Parse frontmatter, `index.md` routes, aliases, headings, links, reusables, variables, and variants with provenance.
5. **Create the three DSH profiles.** Verify the loaded plugin inventory and tool schemas from DSH configuration dumps.
6. **Externalize the three initial skills.** Add the skill loader, protected-region validator, token limit, and isolated candidate overlay.
7. **Run provider evaluation.** Establish the no-model KB scores before optimizing agent instructions.
8. **Install and pin SkillOpt.** The current empty virtual environment is not evidence of an installation.
9. **Implement the `github_docs_dsh` SkillOpt adapter.** Start with offline deterministic lifecycle tests, then one real-model end-to-end trace per arm.
10. **Optimize independently on train/validation.** Preserve every trajectory, proposed edit, validation decision, and best-skill artifact.
11. **Run frozen final evaluation.** Evaluate `S0-neutral` and `S2-skillopt` on identical test questions with real models.
12. **Publish decomposed results.** Report provider performance, skill lift, complete-system performance, categories, uncertainty, tokens, and both latency scopes.

## Acceptance criteria

The refactor is complete only when:

- the evaluator contains no BM25, HNSW, ripgrep query planner, Neo4j schema, ingestion, or Cypher ranking implementation;
- every three-arm request crosses an actual DSH plugin/provider boundary;
- DSH configuration evidence shows the intended profile and skill were loaded;
- filesystem runs use the official DSH filesystem-search plugin, not only its binary path;
- hybrid and graph runs use provider plugins through `ctx.techdocs`;
- all skills are versioned Markdown artifacts and candidate overlays do not mutate source files;
- metadata and `index.md` relations are revision-pinned, query-blind, and traceable;
- SkillOpt train/validation/test separation is enforced and test qrels are unavailable to the optimizer;
- final metrics include category-level Recall, Hit, nDCG, and p50 latency;
- the report clearly labels provider-only results separately from DSH+skill system results;
- final reported system results come from real model calls, not mocks or direct Python retrieval.

## Expected interpretation

The provider evaluation answers:

> Which KB architecture retrieves the relevant GitHub Docs pages most effectively under a matched plugin contract?

The SkillOpt system evaluation answers:

> When each DSH composition receives a skill tailored and validation-optimized for its actual tools, which deployable system gives the best accuracy, evidence use, tokens, and latency?

These are related but different claims. Reporting both prevents a strong skill from being mistaken for a strong retriever, and prevents a strong retriever from being penalized because an untuned agent did not know how to use it.
