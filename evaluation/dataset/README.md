# Dataset construction

This directory contains only the inputs, builders, frozen output, and integrity
checks for the GitHub Docs benchmark dataset. Retrieval implementations and
metric computation live in `../kbbench/`.

## Contents

```text
dataset/
├── build.py              build normalized corpus and questions from sources
├── prepare_raw.sh        fetch the pinned github/docs source revision
├── verify.py             verify frozen hashes and counts
└── data/                 committed, evaluation-ready dataset
    ├── corpus.jsonl
    ├── questions.jsonl
    ├── dataset_manifest.json
    ├── source_manifest.json
    ├── graph_schema.json
    └── splits/
        ├── train.json
        ├── validation.json
        ├── test.json
        └── split_manifest.json
```

The frozen corpus contains 3,740 canonical pages from `github/docs/content` at
commit `c34e3dccad00f61133c799d20e7d1208a0e6cc92`. Each row preserves searchable
rendered text, source path, route and frontmatter metadata, Markdown-link edges,
reusable references, and code-bearing page content.

The 328 real questions come from public GitHub Community discussions. A qrel is
a distinct canonical `docs.github.com` page linked in the accepted answer.
Visible URLs are removed from question text. The dataset contains 421 qrels and
the following fixed splits: 55 train, 27 validation, and 246 held-out test.
Qrels are sparse accepted-answer citations, not exhaustive relevance judgments.

## Prepare or rebuild

Fetch the pinned documentation repository:

```bash
evaluation/dataset/prepare_raw.sh
```

`build.py` accepts the raw documentation checkout, a directory of captured
GitHub Community discussion pages, an output directory, and the pinned revision:

```bash
evaluation/.venv/bin/python evaluation/dataset/build.py \
  --repo-root evaluation/dataset/raw/github-docs \
  --discussions-dir /path/to/github-community-discussions \
  --output-dir /tmp/github-docs-built \
  --revision c34e3dccad00f61133c799d20e7d1208a0e6cc92 \
  --schema-version v2
```

The committed `data/` is already built; rebuilding is not required to run an
evaluation. Verify its immutable file hashes, counts, and excluded oracle fields
with:

```bash
evaluation/.venv/bin/python evaluation/dataset/verify.py
```

The dataset deliberately excludes the earlier heuristic `graph_opportunity`
field. Graph use must be selected by the tested system rather than leaked by the
evaluation data.

## Provenance

GitHub Docs content is available under CC BY 4.0. Question and accepted-answer
metadata are public user-generated GitHub Community content and should be
redistributed only after reviewing GitHub's applicable terms. This benchmark is
not an official GitHub product.
