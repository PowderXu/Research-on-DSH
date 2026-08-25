# Local service contract

The DSH plugin calls one local service. All responses are wrapped as
`{"result": ...}` and errors as `{"error":{"message":"..."}}`.

## `GET /health`

Returns arm, corpus revision, document/chunk counts, retrieval method, and index
build duration.

## `POST /v1/search`

Request fields: `query`, in-scope `scope`, `result_limit`, and
`evidence_token_budget`. The current backend executes BM25 + HNSW + reciprocal-
rank fusion and returns canonical page results with source path, commit, line
span, snippet, score, and retrieval signals.

## `POST /v1/expand`

Neo4j only. Request fields: `query`, `seed_uris`, `result_limit`, and evidence
budget. Every seed URI must have been returned under
`viking://resources/docsqa`. Results preserve `expandedFrom` seed provenance
and typed path signals.

## `POST /v1/fetch`

Fetches bounded source text for previously returned in-scope URIs. Out-of-root
and unknown URIs are rejected.

The DSH adapter validates the resource root again and injects only the bounded
evidence text, not the full candidate list, into model context.
