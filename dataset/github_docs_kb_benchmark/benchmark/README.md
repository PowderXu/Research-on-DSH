# Retrieval evaluation contract

The evaluator consumes JSON Lines predictions and scores canonical GitHub Docs
pages against accepted-answer link qrels.

## Prediction format

Required fields:

- `question_id`: string matching one dataset question;
- `ranked_ids`: ordered list of retrieved sources, best first.

Each source may be a canonical `doc_id`, a `docs.github.com` URL, a repository
`content/.../*.md` path, or a `viking://.../techdocs/...` URI. Unknown sources
are ignored and reported. Duplicate sources are collapsed after normalization.

Optional fields:

- `latency_ms`, or `latency_seconds` when milliseconds are absent;
- `usage.input_tokens`, `usage.output_tokens`, and `usage.total_tokens`;
- `sources` as an alternative to `ranked_ids`; source objects may use `doc_id`,
  `source`, `path`, `uri`, or `url`.

The formal shape is in `prediction_schema.json`.

## Metrics

For question \(q\), let \(R_q\) be the set of canonical pages linked in its
accepted answer and \(L_q@k\) the first \(k\) unique resolved predictions.

- `Hit@k`: 1 when `L@k` contains at least one qrel, otherwise 0.
- `Recall@k`: `|L@k intersection R| / |R|`.
- `nDCG@10`: binary DCG of qrel pages at their retrieved ranks divided by the
  ideal DCG for `min(10, |R|)` pages.
- `AllSupport@10`: 1 when every qrel appears in the first 10 results.
- `p50_latency_ms` / `p95_latency_ms`: percentile across predictions with a
  valid non-negative latency.

Retrieval metrics are macro-averaged across questions. The report includes
overall values and factual slices by:

- `intent_category`;
- `evidence_category`;
- `evidence_structure` (`single`, `linked`, or `dispersed`);
- `qrel_count_group` (`1` or `2+`).

Because the qrels contain only pages cited by accepted answers, pages without a
qrel have zero measured gain but are more accurately described as unjudged than
as definitively irrelevant.

## Completeness and validity

By default, predictions must cover every question exactly once. Duplicate,
unknown, or missing question IDs fail evaluation. `--allow-partial` exists only
for development diagnostics; do not use it for final benchmark results.

The evaluator records unresolved source strings and the number of evaluated
questions. Keep these diagnostics in published reports.
