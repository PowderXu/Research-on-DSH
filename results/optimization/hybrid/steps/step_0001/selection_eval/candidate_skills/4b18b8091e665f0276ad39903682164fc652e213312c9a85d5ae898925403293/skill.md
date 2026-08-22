---
name: github-docs-hybrid
description: Retrieve GitHub Docs with metadata-aware BM25 and dense search, then verify citation-ready evidence.
arm: hybrid
version: 1
---

# GitHub Docs hybrid retrieval

## Contract

- Use `techdocs_search` for retrieval and `techdocs_fetch` only for URIs returned by search.
- Treat returned passages as candidate evidence, not automatic truth.
- Preserve canonical repository paths, headings, versions, and citations. Abstain when evidence is insufficient.

## Strategy

1. Form a concise query that preserves exact identifiers, product/version cues, and the user's material constraint.
2. Always begin by calling the retrieval tool: perform techdocs_search with limit: 10 and the concise query described below. Do not skip or simulate search locally; use the provider's combined BM25/HNSW/reciprocal-rank fusion results. Treat the returned ranked passages and their metadata as candidate evidence that must be inspected before fetching. If the initial search yields weak or irrelevant results, perform at most one materially different reformulation (see Failure recovery) and repeat techdocs_search once; if the second attempt is still insufficient, abstain.
3. Use title, short title, intro, heading, route, content type, version, category, and redirect metadata as retrieval signals.
4. Treat inferred route and version metadata as soft preferences unless the user explicitly supplied that constraint.
5. Inspect the returned section and evidence signals rather than trusting only a page title.
6. Fetch only the smallest additional passage needed to verify an answer or resolve an explicit ambiguity.
7. Cite the canonical page and relevant section.

## Failure recovery

- If the first result set is weak, make at most one materially different reformulation using a documentation synonym while preserving exact identifiers.
- If the top results cover the wrong product or version, add that missing constraint rather than adding generic words.
- Ignore high-scoring passages that do not entail the requested claim.
- Stop and abstain when the bounded second attempt still lacks supporting evidence.

Operational checklist (must follow during each retrieval attempt):

- Compose a concise query that preserves exact identifiers, product/version cues, and any user material constraints.
- Call techdocs_search(query, limit=10) as the first action. Record and inspect top results, their titles, routes, versions, and relevance signals.
- Only call techdocs_fetch for URIs returned by techdocs_search and fetch the smallest passage needed to verify a claim.
- If top results are irrelevant or point to the wrong product/version, perform one reformulation that adds the missing explicit constraint (do not add generic words), then call techdocs_search(limit=10) again.
- After the second search, if supporting evidence is still insufficient, abstain (return NOT FOUND) rather than inventing or inferring unsupported facts.
- Preserve canonical repo paths, headings, version labels, and cite the exact page and section used as evidence.
