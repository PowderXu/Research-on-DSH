# Neo4j schema

The graph is deterministic, query-blind, and namespaced so ingestion does not
delete unrelated Neo4j data.

## Nodes

| Label | Identity | Purpose |
|---|---|---|
| `KBGitHubDocPage` | canonical page ID | page metadata and graph seed |
| `KBGitHubDocChunk` | page ID + chunk number | vector/full-text retrieval unit |
| `KBGitHubDocRoute` | route ID | soft navigation relationship |
| `KBGitHubDocReusable` | reusable identifier | pages sharing included content |
| `KBGitHubDocCodeEntity` | normalized exact identifier | pages mentioning the same flag, environment variable, or code-shaped term |

## Relationships

| Type | Meaning |
|---|---|
| `GH_FROM_PAGE` | chunk belongs to page |
| `GH_LINKS_TO` | explicit contextual Markdown link between pages |
| `GH_IN_ROUTE` | page belongs to a documentation route |
| `GH_INCLUDES` | page includes reusable content |
| `GH_MENTIONS` | page mentions a bounded code-shaped identifier |

`GH_LINKS_TO` stores source section, target anchor, anchor text, and local link
context. Route/reusable/code nodes are degree-bounded during expansion so they
cannot become unrestricted hubs.

Query-time expansion begins from hybrid-returned pages and permits one hop over
the allowlisted relationships. Neighbor chunks are scored against the current
query; path type and seed page remain visible in the result. A path is a
discovery signal, not evidence by itself.
