# Dataset interface contracts

This directory contains the pinned download reference and JSON shape contracts.
It contains no dataset records or source manifests.

- `dataset_source.json` pins the `PowderXu/docsqa-data` commit and manifest hash.
- `corpus.schema.json` describes searchable documentation records.
- `question.schema.json` and `answer.schema.json` describe separate question
  inputs and answer/scoring records, joined by `question_id`.
- `manifest.schema.json` describes a dataset package with no partitions.
- `source_config.schema.json` describes source configuration supplied externally.

Documentation source pins are in `docsqa-data/sources.json`. The 798 public
Discussion identifiers and collection lineage are in that repository's
`provenance/` directory. Construction tools require these paths explicitly;
normal evaluation only downloads the pinned release.

Graph treatment configuration belongs to `dsh_plugin/plugin/graph_schema.json`.
