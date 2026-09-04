# Dataset design

## Dataset boundary

`evaluation/dataset/` contains exactly four directories:

```text
docs/             generated public Markdown/MDX workspace
templates/        source configuration and JSON data contracts
scripts/          construction, combination, and verification code
evaluation_data/  generated per-project and combined QA packages
```

The documentation and evaluation data are local generated artifacts. Public
source configuration and data contracts live in `templates/`; executable
construction logic lives in `scripts/`. Source repository and downloaded
Discussion caches live under `results/cache/`, outside the dataset.

## Unified public source corpus

`evaluation/dataset/scripts/prepare_raw.sh` clones the four public repositories
at their exact commits and materializes one ignored corpus at
`evaluation/dataset/docs/`. `dataset.scripts.build_all` then reads the frozen
798-row public Discussion manifest, downloads or reuses those pages, builds the
four QA packages, validates them, and combines the accepted subset. A private
user GitHub repository is intentionally not part of this real dataset.

The unified layout preserves each original repository-relative path inside a
project directory rather than flattening files:

```text
docs/
├── index.md
├── manifest.{json,jsonl}
├── github-docs/{index.md,content/...,data/...}
├── tailwind-css/{index.md,src/docs/...}
├── prisma/{index.md,apps/docs/content/docs/...}
└── supabase/{index.md,apps/docs/content/...}
```

This layout gives the agent one realistic multi-project workspace without
destroying authored relative links. Directly linked local code/assets outside a
declared content root are harvested into the same repository-relative location.
Document identities are project-qualified, so identical paths from two projects
cannot collide.

The physical tree retains 59 additional Supabase Markdown/MDX source files that
normalize to duplicate or non-page routes. They are explicitly labeled
`noncanonical_documentation` and excluded from default search. This leaves
5,392 source-document records in the structurally accepted, pre-normalization
package. Answer normalization later removes empty navigation/source rows and
produces the 4,860-page evaluated corpus shared by all three retrieval arms.

The generated root and project `index.md` files provide hierarchical routing
metadata. They are excluded from default search and all relevance judgments.
This supports an explicit routing-index ablation without silently changing the
main BM25/HNSW/graph corpus. The physical source corpus and QA benchmark remain
separate: accepted answers, qrels, and judge references are never copied into
the agent's documentation workspace.

`evaluation/dataset/scripts/combine_datasets.py` constructs the corresponding unified
evaluation package only after the deterministic source-package validator has
accepted a QA case. It namespaces canonical document IDs, graph edges, question
IDs, and qrels by project; preserves the original frozen split membership; and
replaces the raw accepted answer with its recursively expanded validated
reference package. The pre-normalization result is 5,392 source-document
records, 556 questions, and 627 qrels under `evaluation_data/combined/`. The stricter
normalized evaluation package contains 4,860 searchable pages and 467
questions. The routing indexes are not inserted into either canonical corpus.

Graph construction is a retrieval-treatment concern. The evaluation dataset
contains only page facts such as routes and authored links. Neo4j node types,
relationships, indexes, and traversal policy belong to the plugin contract at
`dsh_plugin/plugin/graph_schema.json`.

## Candidate discovery and provenance boundary

A candidate begins as an answered public GitHub Discussion whose
`mainEntity.acceptedAnswer` contains a direct link to the project's exact
documentation host and documentation path. A host match is only a screen; it
does not become a qrel until the URL resolves to a canonical page in the pinned
local corpus. The construction stages are therefore:

1. enumerate `is:answered` Discussions in explicit `created:` ranges;
2. read the public QAPage JSON-LD and require `acceptedAnswer`;
3. retain only links occurring directly in that accepted answer;
4. require the exact configured host and documentation path prefix;
5. resolve at least one link to the pinned corpus, preserving unresolved links
   for the stricter source-package gate;
6. freeze only public identifiers, permalinks, direct documentation links where
   retained, and split membership in `templates/discussion_sources.jsonl`.

`dataset.scripts.candidate_discovery discover` implements stages 1--4 for new
collections. It fails closed when any date partition returns 950 or more unique
IDs because GitHub's searchable web listing has historically stopped around
1,000 results. The range must then be subdivided. This guard prevents a capped
listing from being mistaken for an exhaustive population.

The existing 798-row freeze predates that guard. Its retained lineage is:

| Dataset | Answered listing | Accepted-answer pages | Base-host screen | Exact host + docs path | Frozen rows | Historical limitation |
|---|---:|---:|---:|---:|---:|---|
| GitHub Docs | unknown | unknown | not retained | not retained | 328 | Enumeration and intermediate screen cannot be reconstructed. |
| Prisma | 1,002 | 999 | 239 | 227 | 213 | Unpartitioned screen reached/exceeded the cap; historical post-screen selection and route/redirect state are missing. |
| Supabase | 1,002 | 998 | 166 | 132 | 90 | Unpartitioned screen reached/exceeded the cap; historical post-screen selection and route/redirect state are missing. |
| Tailwind CSS | 999 | 994 | 370 | 202 | 167 | Unpartitioned screen was near the cap; historical post-screen selection and route/redirect state are missing. |

For Prisma, Supabase, and Tailwind CSS, every frozen row is present in the
archived 2026-08-24 host screen, and its source URL, accepted-answer permalink,
and exact-filtered documentation links agree. The archived screens and their
hashes are summarized in `templates/candidate_discovery.json`; the large raw
screens remain outside this clean repository. The exact historical post-screen
selection procedure, including the resolver and redirect snapshot, that reduced
227/132/202 screened rows to 213/90/167 was not retained. One Tailwind exclusion
is separately recorded as a 91-qrel high-link case, but the complete per-row
exclusion lineage is irrecoverable and must not be inferred after the fact.

The reproducible boundary is consequently precise but limited: the frozen 798
identifiers can be rebuilt against the four pinned corpora, and all 798 current
project-package rows have at least one non-dangling local qrel; the later
deterministic source gate accepts 556. The original listing population and all
candidate-to-freeze decisions are not fully reproducible. Current benchmark
claims must not describe the collection as exhaustive or unbiased.

Run the offline lineage and pinned-qrel audit with:

```bash
PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m dataset.scripts.candidate_discovery audit
```

## Questions and relevance judgments

The combined benchmark has 556 structurally accepted questions from public
GitHub Discussions: 232 GitHub Docs, 109 Tailwind CSS, 148 Prisma, and 67
Supabase cases. Queries retain the real question text, while 627 relevance
judgments point to project-namespaced canonical documentation pages resolved
from accepted-answer links.

Accepted-answer citations are sparse judgments. A retrieved page without a qrel
has zero measured gain but is **unjudged**, not proven irrelevant.

## QA source-package validity

The raw accepted-answer citation is only a candidate label. Before semantic
correctness judging, each case must pass a deterministic source-package gate.
The complete evaluation unit is:

```text
question text
+ normalized accepted-answer text
+ documentation pages resolved inside the pinned corpus
+ reproducible image-derived text attached to its local source
```

The gate applies these rules:

- every documentation URL must use the configured documentation host and
  resolve to a document ID in the pinned local corpus;
- an answer containing even one unresolved internal documentation URL is
  rejected rather than partially retained;
- non-document external links reject the case rather than being imported as
  evidence;
- profiles, self-permalinks, and media-hosting URLs are metadata, not qrels;
- when an accepted answer links another GitHub issue or discussion, the frozen
  accepted answer of that target is recursively appended; if no accepted
  answer is available, the original case is rejected;
- image references from the question, answer, and linked docs are resolved
  during normalization. Reproducible pixels are converted once into a local,
  provenance-bearing text description; required unreproducible images reject
  the case. Retrieval and answer judging do not consume image vectors.

The deterministic gate does not decide whether the remaining evidence is
factually correct or complete. Those checks happen after reproducible images
have been converted to provenance-bearing local text.

## Physical split labels and evaluation roles

| Physical source label | Questions | Current role |
|---|---:|---|
| Train | 39 | construction examples |
| Validation | 85 | legacy dataset validation/development |
| Test | 432 | legacy source label; not currently a sealed test cohort |

These source-package labels sum to 556. Normalization retains 467 cases and
preserves their inherited filenames, yielding 33 construction examples, 73
legacy physical-validation records, and 361 records physically named `test`.
The latter records have been inspected while retrievers and evaluation code were
developed, so current reports call them the **development/evaluation
partition**, not held-out or final evidence.

The intended formal benchmark remains zero-shot: a submitted system receives
the pinned corpus and each newly sealed question, but no answer, qrel, aspect,
supervised update, or in-context benchmark example. A formal result therefore
requires a new temporal or source-disjoint cohort collected after the system and
judge are frozen. `graph_opportunity` is intentionally excluded because it was
a model/heuristic label rather than factual evaluation evidence.

## Factual slices

Every result is grouped by:

- intent category;
- evidence category, including direct, section, anchor, linked multi-page, and
  dispersed multi-page questions;
- qrel connectivity (`single`, `linked`, `dispersed`), derived from authored
  corpus links and qrels;
- qrel count (`1` or `2+`).
- whether the question itself contains reproducible image-derived text.

Answer-aspect reports additionally distinguish aspects supported by pinned
local documentation from aspects supported only by the historical accepted
answer or question context. This prevents a docs-only system from being blamed
for evidence absent from its allowed corpus.

Construction, provenance, hashes, and verification are documented in
`evaluation/dataset/scripts/README.md`. Evaluation code and the standalone
prediction contract are documented separately in `evaluation/kbbench/README.md`.

## Public-source, privacy, licensing, and ethics notes

- Collection is limited to unauthenticated public GitHub Discussions and public
  documentation repositories. Private repositories, private discussions, and
  authenticated-only material are out of scope.
- The committed candidate manifest stores public identifiers and permalinks,
  not copied question/answer text. Downloaded HTML, question text, answer text,
  and image bytes stay in ignored local caches or generated evaluation data.
- Public support posts can still contain usernames, email addresses, access
  tokens, or other personal/sensitive text. Raw caches must not be published;
  release examples should be redacted or paraphrased, and documented deletion
  or takedown requests should be honored.
- A selected answer is provenance for a reference package, not proof that the
  answer is correct, complete, current, or posted with benchmark reuse in mind.
  Deterministic source validation and model-assisted semantic review remain
  necessary, and reported qrels are explicitly sparse.
- The four upstream documentation repositories do not share one assumed
  license. Reproduction clones pinned public sources and preserves their
  attribution and repository identity; any redistributed text must follow each
  upstream repository's license. This benchmark's metadata does not relicense
  upstream documentation or Discussion content.
- New live discovery should use date partitions, modest concurrency, a clear
  user agent, and GitHub's applicable access and rate-limit rules. Aggregate
  statistics should be preferred over exposing contributor-level behavior.
