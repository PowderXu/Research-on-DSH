# DSH plugin design

## Composition

The benchmark uses four native DSH plugin roles:

| Role | DSH component | Responsibility |
|---|---|---|
| Runtime tool plugin | `@kbbench/dsh-techdocs` | register bounded search/fetch/expand tools over a local backend |
| Filesystem skill plugin | `@kbbench/dsh-techdocs/skill-fs` | teach routing through Markdown, frontmatter, and index pages |
| Hybrid skill plugin | `@kbbench/dsh-techdocs/skill-hybrid` | teach query formation, evidence checking, and bounded retry |
| Neo4j skill plugin | `@kbbench/dsh-techdocs/skill-neo4j` | teach when to expand and how to verify graph-added evidence |

The DSH profile loads the runtime and all three skill plugins, then an arm patch
enables exactly the required tools and one skill. The skills remain external
Markdown artifacts so SkillOpt can improve instructions without changing tool
schemas or retrieval code.

## Tool boundary

The hybrid and Neo4j arms use a thin native DSH adapter:

```text
DSH model
  -> techdocs_search / techdocs_fetch / techdocs_expand
  -> native DSH tool registration
  -> local HTTP contract
  -> GitHub Docs retrieval backend
```

The backend returns canonical page IDs, repository paths, line spans, commit
identity, retrieval signals, and an evidence package with a fixed token budget.
The plugin rejects results outside `viking://resources/techdocs` before they can
enter model context.

`techdocs_expand` is explicit rather than hidden inside search. This lets the
agent decide whether a relationship-bearing question warrants traversal and
lets the evaluation measure whether expansion was actually called.

## Retrieval arms

### Filesystem

Uses DSH's official filesystem search/read tools over the pinned raw repository.
`index.md`, frontmatter, route metadata, and links are routing evidence, not
answers. The agent must open and verify the final Markdown passage.

### Hybrid

Chunks rendered Markdown by paragraph boundaries, builds sparse BM25 and dense
HNSW indexes, and fuses document rankings with reciprocal-rank fusion. It
returns citation-ready passages from the top pages.

### Neo4j

Uses the same hybrid seeds, then permits one bounded expansion over a graph
built before evaluation. Graph paths are discovery reasons; a page is not valid
evidence until its passage supports the answer.

## Invariants

- Corpus and graph are pinned to one immutable commit.
- Search never reads qrels or evaluation categories.
- Graph expansion begins only from search-returned seed URIs.
- Every graph-added page preserves seed and typed-path provenance.
- Model-facing evidence is bounded; the larger candidate set stays outside the
  conversation.
- Skills may change behavior but cannot rename tools or embed qrel page IDs.
