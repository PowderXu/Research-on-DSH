# Dataset pipeline

`evaluation/dataset/` has exactly four responsibilities:

```text
dataset/
├── docs/             generated public documentation workspace
├── templates/        source configuration and data-shape contracts
├── scripts/          dataset construction and verification code
└── evaluation_data/  generated QA packages ready for evaluation
```

The generated directories contain their own `.gitignore`; they stay local.
Source Git checkouts and downloaded Discussion HTML are caches under
`results/cache/`, not dataset contents.

## 1. Prepare documentation

From the repository root:

```bash
evaluation/dataset/scripts/prepare_raw.sh
```

The command reads `templates/public_sources.json`, clones the four public
repositories into `results/cache/docsqa-source-repos/`, verifies their pinned
commits, and materializes:

```text
docs/
├── index.md
├── manifest.json
├── manifest.jsonl
├── github-docs/{index.md,content/...,data/...}
├── tailwind-css/{index.md,src/docs/...}
├── prisma/{index.md,apps/docs/content/docs/...}
└── supabase/{index.md,apps/docs/content/...}
```

Use `--offline` to reuse existing checkouts and `--force` to intentionally
replace an existing generated `docs/` tree:

```bash
evaluation/dataset/scripts/prepare_raw.sh --offline --force
```

Original repository-relative paths are preserved. Directly linked local code
or assets outside a declared content root are copied as support files. Routing
indexes, support files, and noncanonical duplicate pages are marked
non-searchable in `manifest.jsonl`. This creates 5,392 source-document records
in the pre-normalization package. The stricter normalized evaluation
package removes empty navigation/source rows and supplies the same 4,860
searchable documents to all retrieval arms.

## 2. Audit or collect Discussion candidates

The benchmark input is the 798-row
`templates/discussion_sources.jsonl`. It contains only public Discussion
identifiers, source/accepted-answer permalinks, exact documentation links where
retained, and frozen split membership; it does not contain copied question or
answer text. Its collection lineage and historical artifact hashes live in the
small `templates/candidate_discovery.json` record.

Audit the frozen row counts, duplicate keys, retained historical host screens
when locally available, and qrel resolution against the generated pinned
project packages with:

```bash
PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m dataset.scripts.candidate_discovery audit
```

The audit remains useful in a clean checkout without the legacy host-screen
files: it reports them as unavailable instead of pretending their selection
history can be recreated. With this study's older sibling `results/` directory
present, it also verifies the recorded SHA-256 hashes and proves that every
Prisma, Supabase, and Tailwind frozen row came from the retained host screen.

For a new collection, enumerate answered Discussions in explicit date
partitions. For example:

```bash
PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m dataset.scripts.candidate_discovery discover \
  --dataset prisma \
  --repository prisma/prisma \
  --docs-host prisma.io \
  --docs-host www.prisma.io \
  --docs-path-prefix /docs \
  --created-range 2020-01-01..2020-12-31 \
  --created-range 2021-01-01..2021-12-31 \
  --output results/runs/dataset-discovery/prisma
```

Repeat or narrow ranges until each is safely below the listing guard. The tool
raises an error at 950 unique IDs because GitHub's searchable listing has
historically stopped around 1,000 results. It writes a direct-link candidate
screen, not qrels and not a replacement benchmark manifest. Resolve the links
to the pinned corpus, run the deterministic source-package gate, review the
collection provenance, and only then intentionally freeze a new manifest.

The current 798-row manifest is only partially collection-reproducible. The
GitHub Community enumeration/host screen was not retained. The other three
2026-08-24 screens were unpartitioned and at or near the listing ceiling, and
their exact historical post-screen selection procedure and redirect/resolver
snapshot were not retained. These limits are recorded explicitly in
`docs/DATASET_DESIGN.md`; the current pool must not be described as exhaustive.

## 3. Build, validate, and combine the QA packages

Build all four per-project packages, download/cache the public pages, resolve
accepted-answer documentation links to the pinned corpora, apply deterministic
source validation, and combine accepted cases with:

```bash
PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m dataset.scripts.build_all --force
```

Without `--refresh`, cached pages and redirect resolutions are reused. The
validator rejects unresolved internal documentation URLs, non-document
external dependencies, and linked issues/discussions whose accepted answer
cannot be completely recovered. Reference answers and images remain
evaluation-only.

The source-construction stage writes the pre-normalization package at
`evaluation_data/combined/` with:

```text
combined/
├── corpus.jsonl
├── questions.jsonl
├── manifest.json
└── splits/{train,validation,test}.json
```

Every document ID, question ID, graph edge, and qrel is project-namespaced. This
pre-normalization build contains 5,392 source-document records, 556
structurally accepted questions, and 627 qrels. The evaluated normalized corpus
is the separate 4,860-document, 467-question package described above. A clean
cached replay reproduces the combined source files byte-for-byte:

```text
corpus.jsonl    33177f14dbfdea54793ebdc0c41bf0a6a3a945268f72799009adb2c148e9a6fd
questions.jsonl 981a1a3cf403d8c252d2b2c4ff8073445d94811138029792afacc39dbf076785
manifest.json   f224a0f904ecb9027d2238513e8079ba403b9928772734561a93b68f7636d99a
```

`build.py` and `build_discussion_corpus.py` are the project-specific parsers;
`build_all.py` is the public entrypoint. `combine_datasets.py` remains usable
as a lower-level command when validated project packages already exist.

## 4. Verify

```bash
PYTHONPATH=evaluation evaluation/.venv/bin/python \
  -m dataset.scripts.verify
```

Verification checks counts, ID uniqueness, split isolation, qrel resolution,
oracle-field exclusion, and reference-answer leakage. It also rejects a
`graph_schema.json` inside evaluation data: the graph treatment contract is
owned by `dsh_plugin/plugin/graph_schema.json`.

## Handling public-source data

The committed discovery inputs are identifiers and permalinks. Raw public page
HTML, answer/question text, and images stay in ignored caches or generated
evaluation data and should not be published without a separate privacy,
licensing, and sensitive-data review. Public visibility does not guarantee
correctness, consent for benchmark reuse, or absence of personal data. Preserve
source attribution, honor removal requests, redact release examples, and follow
each upstream repository's license rather than assuming one license covers all
four corpora.
