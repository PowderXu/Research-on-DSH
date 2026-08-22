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
2. Request `limit: 10` so the evidence package matches the benchmark's Recall/Hit/nDCG@10 contract. Let the provider combine BM25, HNSW, and reciprocal-rank fusion; do not manually imitate those algorithms.
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
