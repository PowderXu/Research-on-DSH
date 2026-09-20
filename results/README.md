# Local generated outputs

Everything below this directory is local and ignored by Git:

```text
results/
  runs/
    dataset-analysis/  source validation, normalization, aspects, weak supervision
    agents/            actual DSH-harness rollouts and aspect judgments
  cache/               retained source discussion snapshots and image bytes
```

The local workspace keeps only artifacts that support a claim in the paper;
stale trials, direct-retriever reports, copied runtime stores, and temporary
checks are excluded. Generated reports are evidence for a specific run
contract, not maintained documentation. The only current consolidated result
tables are in
[`../docs/RESULTS.md`](../docs/RESULTS.md).
