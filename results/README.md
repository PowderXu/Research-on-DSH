# Generated benchmark results

This directory is the stable output root for local benchmark and optimization
runs. Generated payloads are intentionally not committed.

Use `results/runs/` for outputs, for example:

```text
results/runs/
  retrieval/<run-id>/
  agents/<arm>/<run-id>/
  optimization/<arm>/<run-id>/
```

Evaluators create their requested output directory automatically. Raw events,
predictions, reports, checkpoints, and diagnostics remain local because
`results/runs/` and `results/tmp/` are ignored by Git. The consolidated
historical reference tables are in the repository root `README.md`.
