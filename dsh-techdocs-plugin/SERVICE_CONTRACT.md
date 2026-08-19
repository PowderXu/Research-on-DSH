# KB service contract

The DSH bundle is deliberately thin. Retrieval algorithms and model runtimes live behind a local HTTP service so that DSH orchestration can be tested independently from Qdrant, OpenViking, Neo4j, and Python model dependencies.

## `POST /v1/search`

The request contains the query, a `viking://resources/techdocs` scope, candidate/result limits, evidence budget, and a bounded graph policy. The service must execute these stages and record each one in `trace`:

1. dense and BM25/sparse global retrieval over section-aware Markdown and linked-code chunks;
2. reciprocal-rank fusion and document grouping;
3. OpenViking/repository hierarchy as a soft prior, never a hard gate;
4. optional one-hop deterministic structural-graph expansion from the top seeds, including explicit document-to-document, document-to-section, and document-to-code links;
5. local reranking of the bounded candidate set;
6. evidence packing with stable source IDs and section-level `viking://` URIs.

The response is:

```json
{
  "queryId": "stable trace id",
  "results": [
    {
      "sourceId": "corpus document id",
      "uri": "viking://resources/techdocs/product/doc.md",
      "title": "Document title",
      "section": "heading anchor",
      "repoPath": "docs/product/billing.md",
      "commit": "full git commit SHA",
      "lineStart": 41,
      "lineEnd": 55,
      "targetKind": "document|section|code_file|symbol|snippet",
      "snippet": "evidence-bearing passage",
      "score": 0.91,
      "signals": ["bm25", "dense", "hierarchy", "graph", "rerank"]
    }
  ],
  "trace": {
    "latencyMs": 123,
    "stageCounts": {},
    "paidUsd": 0,
    "indexRevision": "content hash"
  }
}
```

Every result and expanded neighbor must be attributable to a pinned repository commit. A mutable branch name is acceptable for ingestion discovery but not as the identity of a scored index.

## Repository parsing

Parse Markdown with an existing CommonMark/GFM AST implementation. Normalize inline links, reference-style links, anchors, GitHub `blob`/`tree` URLs, and repository-relative paths against the source file. Fragment identifiers resolve to `Section` nodes. Links outside the pinned repository become `ExternalResource` nodes and are not fetched automatically.

Parse repository code with existing language-aware indexers. Exact file links create `LINKS_TO_CODE`; exact, unambiguous symbol references create `REFERENCES_SYMBOL`. Fenced code examples become `Snippet` nodes but must not be asserted as implementations of a symbol unless a file link, line anchor, or exact static-index match supplies evidence.

Repeated navigation links, generated sidebars, broad directory links, and ambiguous short identifiers must not create unrestricted expansion hubs. Preserve each edge's source path, line/anchor, parser, and commit SHA.

## `POST /v1/fetch`

The request contains previously returned in-scope URIs and a token budget. It returns exact passages with stable citations. It must reject memory, skill, filesystem, and out-of-root URIs.

## Budget invariant

Indexing, hybrid retrieval, deterministic graph construction, and local reranking are zero-paid-API stages. Any optional model call must be pre-authorized by the meter, write actual usage to the trace, and stop at the configured operating limit (default `$18`) so the absolute `$20` ceiling retains a safety reserve.
