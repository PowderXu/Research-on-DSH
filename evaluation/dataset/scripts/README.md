# Dataset download and construction tools

All released data and source manifests belong to
[PowderXu/docsqa-data](https://github.com/PowderXu/docsqa-data).
This benchmark keeps the code, schemas and `../templates/dataset_source.json`
download pin. Generated documentation, data packages and source caches are ignored.

## Normal evaluation

```sh
PYTHONPATH=evaluation:. evaluation/.venv/bin/python -m dataset.scripts.download_dataset
PYTHONPATH=evaluation:. evaluation/.venv/bin/python -m dataset.scripts.verify
```

The downloader checks the exact commit, manifest and file hashes, including the
uncompressed corpus. Private downloads use GitHub CLI authentication. Use
`--repository-dir /path/to/docsqa-data` to read the pinned local Git commit offline.
A replaced local package is preserved in a sibling backup directory.

## Constructing a new release

These optional authoring commands require an external `docsqa-data` checkout.
They are not part of benchmark setup and do not regenerate the downloaded release.
From the benchmark root, set the path to that checkout:

```sh
DOCSQA_DATA_DIR=/absolute/path/to/docsqa-data
```

Prepare documentation from the pinned source configuration:

```sh
evaluation/dataset/scripts/prepare_raw.sh --config "$DOCSQA_DATA_DIR/sources.json"
```

This clones the four upstream repositories into `results/cache/docsqa-source-repos/`
and materializes the ignored `evaluation/dataset/docs/` tree. Add `--offline` to
reuse clean pinned checkouts; add `--force` to replace an existing generated tree.
Original repository paths and declared support files are preserved. Routing and
noncanonical pages are excluded from default search.

Audit the 798 frozen source identifiers and their collection lineage:

```sh
PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m dataset.scripts.candidate_discovery audit \
  --frozen-manifest "$DOCSQA_DATA_DIR/provenance/discussion_sources.jsonl" \
  --lineage "$DOCSQA_DATA_DIR/provenance/candidate_discovery.json"
```

Missing historical host-screen caches are reported as unavailable. The retained
manifest does not make the original collection exhaustive or fully reproducible;
see [dataset design](../../../docs/DATASET_DESIGN.md).

Build, validate and combine the four project packages:

```sh
PYTHONPATH=evaluation:. evaluation/.venv/bin/python \
  -m dataset.scripts.build_all \
  --source-file "$DOCSQA_DATA_DIR/provenance/discussion_sources.jsonl" \
  --config "$DOCSQA_DATA_DIR/sources.json" --force
```

Cached public pages and redirect resolutions are reused unless `--refresh` is
set. The source gate produces 556 eligible questions and 5,392 source documents
under `evaluation_data/combined/`. The separately published normalized release
contains 467 questions and 4,860 searchable pages. Questions and answers are
separate JSONL files joined by `question_id`; neither stage creates partitions.

The lower-level `combine_datasets` command also requires `--config`; image
materialization requires `--source-config`, both pointing to the external
`sources.json`. `build.py` and `build_discussion_corpus.py` implement project parsers.
New candidate discovery uses explicit date ranges and an explicit output directory;
run `python -m dataset.scripts.candidate_discovery discover --help` with the same
Python environment for its options.

## Verification and publication

`dataset.scripts.verify` checks counts, unique IDs, question/answer separation,
qrel resolution and absence of dataset partitions. Graph treatment configuration
belongs to the plugin, never to the data package.

Use `export_dataset --source ... --destination ... --sources "$DOCSQA_DATA_DIR/sources.json"`
for a new empty export directory. Publish dataset changes in `docsqa-data`, then
update the benchmark's pinned commit and manifest hash. Source attribution and
historical collection limits stay with the data. API work logs, source HTML
caches and local research experiments are not release files.
