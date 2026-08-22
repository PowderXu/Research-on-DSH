# Repository graph schema

The graph represents evidence in one pinned Git repository revision. Node identity includes repository identity and commit SHA so a link never silently changes meaning after the source repository moves.

## Nodes

| Label | Stable identity | Purpose |
| --- | --- | --- |
| `Repository` | canonical clone URL | repository boundary |
| `Commit` | full SHA | immutable evaluation/index revision |
| `Directory` | commit + normalized path | hierarchy and soft routing |
| `Document` | commit + normalized Markdown path | retrievable document |
| `Section` | document + normalized heading anchor | citation and section retrieval |
| `CodeFile` | commit + normalized code path | linked implementation/configuration file |
| `Symbol` | code file + language-aware symbol ID | definition/reference target |
| `Snippet` | document + source line range | fenced example embedded in Markdown |
| `ExternalResource` | normalized external URL | recorded boundary; not automatically ingested |

## Evidence-backed edges

| Relationship | From → To | Required evidence |
| --- | --- | --- |
| `AT_COMMIT` | repository content → commit | Git tree entry |
| `CONTAINS` | repository/directory → path node | Git tree path |
| `HAS_SECTION` | document → section | Markdown AST heading |
| `LINKS_TO` | document/section → document | resolved Markdown link |
| `LINKS_TO_SECTION` | document/section → section | resolved path plus heading fragment |
| `LINKS_TO_CODE` | document/section → code file | resolved repository path or pinned GitHub blob URL |
| `REFERENCES_SYMBOL` | document/section/snippet → symbol | explicit symbol link or exact unambiguous static-index match |
| `DEFINES` | code file → symbol | language-aware static index |
| `HAS_SNIPPET` | document/section → snippet | Markdown AST code fence |
| `LINKS_EXTERNAL` | document/section → external resource | external Markdown link |

Every relationship stores `source_path`, `source_line`, `source_anchor`, `parser`, and `commit`. Do not create a generic `RELATED_TO` relationship.

## Retrieval policy

Hybrid lexical+dense retrieval supplies the seeds. Graph expansion is one hop by default, uses only allowlisted relationships, and keeps a small per-seed neighbor limit. Repository hierarchy is a scoring prior, not a filter. Common navigation links and ambiguous identifiers are down-weighted or excluded before expansion.
