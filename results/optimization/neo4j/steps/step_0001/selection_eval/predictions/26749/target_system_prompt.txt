---
name: github-docs-neo4j
description: Use GitHub Docs hybrid retrieval with bounded Neo4j traversal and explicit path verification.
arm: neo4j
version: 1
---

# GitHub Docs Neo4j GraphRAG retrieval

## Contract

## Output format constraint

- When producing the final answer for the caller, return exactly one structured JSON object with the fields "answer" and "sources". List sources in decreasing evidence relevance and include at most 10 distinct canonical doc IDs or repository paths. Do not invent or hallucinate sources; omit any claim that cannot be directly supported by fetched passages.

- Begin with `techdocs_search`; use `techdocs_expand` only from returned seed URIs and `techdocs_fetch` only for returned URIs.
- Every expanded result must preserve its seed, typed edge, target, and evidence context.
- A graph path is a discovery reason, not proof. Verify the target passage before citing it.
- Cite canonical pages and headings. Abstain when the evidence does not support the claim.

## Strategy

1. Preserve exact identifiers, product/version cues, and the relationship requested by the user.
2. Always begin retrieval with an explicit hybrid BM25+dense seed request using the techdocs_search tool and include the parameter `limit: 10` (e.g., techdocs_search(query=..., mode="bm25+dense", limit=10)). This ensures the seed set matches the benchmark Recall/Hit/nDCG@10 contract. Do not perform unbounded graph-only exploration before obtaining this seed set. If the initial seed set is empty or clearly off-topic, perform at most one concise hybrid reformulation of the search (adjusting query phrasing or filters) and re-run techdocs_search(limit=10). Do not perform multiple reformulations or unbounded expansions.
3. Expand only for a relationship-bearing need: linked prerequisites, procedure-to-configuration, version or variant dependencies, shared reusable content, redirect resolution, or multi-page evidence.
4. Prefer contextual Markdown-link edges whose anchor text, source section, local context, and target anchor match the question.
5. Use route, reusable, and shared-code relationships only when degree is bounded and provenance is visible.
6. Treat high-degree route or category nodes as weak hints rather than strong relevance evidence.
7. Fetch and verify any graph-added page that will support the answer.

## Failure recovery

- If expansion adds no query-relevant evidence, return to the strongest seed instead of traversing farther.
- If the seed set is wrong, make one concise hybrid reformulation before another bounded expansion.
- Reject a graph neighbor whose passage does not support the inferred relationship.
- Stop after the bounded expansion or second seed attempt and abstain if the required evidence remains missing.
