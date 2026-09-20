# Local service contract

The DSH plugin calls one local service. All responses are wrapped as
`{"result": ...}` and errors as `{"error":{"message":"..."}}`.

## `GET /health`

Returns arm, corpus revision, document/chunk counts, retrieval method, index
build duration, and the selected arm-owned local data root.

## `POST /v1/search`

Request fields: `query`, in-scope `scope`, `result_limit`, and
`evidence_token_budget`. The current backend executes BM25 + HNSW + reciprocal-
rank fusion and returns canonical page results with source path, commit, line
span, snippet, score, and retrieval signals.

## `POST /v1/expand`

Neo4j only. Request fields: `query`, `seed_uris`, `result_limit`, and evidence
budget. Every seed URI must have been returned under
`viking://resources/docsqa`. Results preserve `expandedFrom` seed provenance
and expose hop count plus typed path signals. Markdown-link traversal is capped
at two page-to-page hops; shared reusable-content, code-entity, and route
transitions remain capped at one. KGGen entity/claim transitions are also one
hop and degree bounded; every semantic path terminates at a retrieval unit that
stores the source evidence.

## `POST /v1/fetch`

Fetches bounded source text for previously returned in-scope URIs. Out-of-root
and unknown URIs are rejected.

The DSH adapter validates the resource root again and injects only the bounded
evidence text, not the full candidate list, into model context.
