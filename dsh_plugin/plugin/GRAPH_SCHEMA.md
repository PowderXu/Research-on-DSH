# Neo4j schema v2

This is the graph treatment schema owned by the DocsQA plugin. The evaluation
dataset contains corpus facts and qrels, not this schema. All labels are
namespaced so ingestion cannot delete unrelated Neo4j data.

The flat hybrid arm stores the same neutral retrieval units in its local index;
it does **not** need Neo4j. Only the graph arm projects those records into the
model below. The machine-readable contract is
[`graph_schema.json`](graph_schema.json).

## Document and image-text structure

```text
Project -HAS_DOCUMENT-> Document -HAS_SECTION-> Section -HAS_UNIT-> RetrievalUnit
RetrievalUnit -NEXT-> RetrievalUnit

ImageOccurrence:RetrievalUnit -USES_ASSET-> ImageAsset
ImageOccurrence -NEAR-> RetrievalUnit
Document -LINKS_TO-> Document
```

`RetrievalUnit` is extensible to text chunks, code blocks, tables, and image
occurrences. The current parser materializes text chunks and image occurrences;
fenced code and tables remain inside their containing text chunk. An image
occurrence stores its page-specific alt text and location. The shared
asset stores OCR, a normalized image description, dimensions, and content
hash. Keeping occurrence and asset separate prevents one reused image from
losing its local meaning. Image vectors are intentionally excluded.

Images, exact code identifiers, KGGen entities, predicates, and claims are
project-scoped. The same relative image URL or entity name in two independent
repositories therefore cannot create an accidental cross-project traversal;
cross-project discovery requires an explicit authored link.

`KB_LINKS_TO` stores anchor text, source section, target anchor, and local
context. `KB_NEXT` and `KB_NEAR` recover local reading order; they are not
semantic claims.

## KGGen semantic layer

[KGGen 0.4.0](https://github.com/stair-lab/kg-gen) performs open entity and
relation extraction, aggregation, and clustering. The adapter does not invent a
fixed global predicate vocabulary. It stores clustered predicates as data:

```text
Entity -SUBJECT_OF-> Claim -HAS_OBJECT-> Entity
Claim -USES_PREDICATE-> Predicate
Claim -SUPPORTED_BY-> RetrievalUnit
```

Every claim retains its raw subject/predicate/object and a source excerpt. The
canonical entity and predicate IDs come from KGGen's clustering; the plugin
adapter restores unit-level evidence provenance after clustering. Polarity,
condition, and confidence are explicit claim fields so later extractors can
populate them without changing the graph shape.

KGGen is installed in a separate Python environment because its current
`sentence-transformers` requirement conflicts with the benchmark environment.
Its artifact is built offline and stored under `data/neo4j/artifacts/`; Neo4j
ingestion never reads benchmark questions or qrels.

## Indexes and expansion

- Full-text index: `KBDocsQARetrievalUnit.search_text`
- Text HNSW: `KBDocsQARetrievalUnit.text_embedding`

The current query foundation uses BM25 plus text HNSW. Image alt text, OCR, and
normalized descriptions participate only as text.

Query-time expansion begins from hybrid-returned documents. It allows at most
two authored-link hops and one semantic transition through a bounded KGGen
entity. Route, reusable-content, and exact code-identifier transitions are also
degree bounded. Candidates are rescored against the current query before
fusion. A graph path is only a discovery signal; the linked retrieval unit and
its source excerpt remain the answer evidence.
