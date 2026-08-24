# Dataset design

## Source corpus

The corpus is the `content/` tree of `github/docs` pinned to commit
`c34e3dccad00f61133c799d20e7d1208a0e6cc92`. The packaged dataset contains
3,740 canonical Markdown pages. Each row preserves the page ID, source path,
title, short title, rendered text, route, content type, versions, outgoing page
IDs, contextual Markdown-link records, reusable-content identifiers, and
variant metadata where available.

The normalized corpus is under `dataset/github_docs_kb_benchmark/data/`.
Raw Markdown is deliberately not duplicated in Git; prepare it under the ignored
`data/github-docs/` directory with `scripts/prepare_raw_github_docs.sh`.

## Questions and relevance judgments

The 328 questions come from public GitHub Community discussions. The query is
the real question title/body with visible documentation URLs removed. Relevance
judgments are canonical GitHub Docs pages linked by the accepted answer:

- 421 total page qrels;
- 263 questions have one relevant page;
- 65 questions have two or more relevant pages;
- maximum qrel count is six.

Accepted-answer citations are sparse judgments. A retrieved page without a qrel
has zero measured gain but is **unjudged**, not proven irrelevant.

## Frozen splits

| Split | Questions | Purpose |
|---|---:|---|
| Train | 55 | skill or retrieval-policy optimization |
| Validation | 27 | parameter and architecture selection |
| Test | 246 | final reporting only |

The split files are disjoint and the held-out test set is inherited from the
original data construction. `graph_opportunity` is intentionally excluded: it
was a model/heuristic label rather than factual evaluation evidence.

## Factual slices

Every result is grouped by:

- intent category;
- evidence category, including direct, section, anchor, linked multi-page, and
  dispersed multi-page questions;
- qrel connectivity (`single`, `linked`, `dispersed`), derived from explicit
  corpus links and qrels;
- qrel count (`1` or `2+`).

The complete portable schema, hashes, and standalone evaluator are documented
in `dataset/github_docs_kb_benchmark/README.md`.
