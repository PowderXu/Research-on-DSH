# KB evaluation

This directory evaluates the already-built package under
`../dataset/evaluation_data/normalized/`. It does not construct or mutate the
dataset. Final arm comparisons use the plugin-owned filesystem workspace
materialized from that same corpus, so every arm sees identical image-derived
text and canonical pages.

## Evaluation paths

- `plugin_eval.py`: matched filesystem, BM25+HNSW, and Neo4j retrieval arms;
- `retrieval.py`: retrieval-component and graph-expansion ablations;
- `scoring.py`: canonical source resolution and shared IR metrics;
- `indexes.py`: the BM25 index used by the local retrieval implementation.

The primary metrics are Recall@5/10, Hit@1/5/10, nDCG@10, AllSupport@10, and
warm p50 retrieval latency. p95 latency and offline construction costs are
optional diagnostics when a run records them. Reports are sliced by question
intent, evidence category, evidence structure, and qrel count. An unjudged page
receives no gain because the accepted-answer qrels are sparse; this does not
prove that page is irrelevant.

When `retrieval.py` or `plugin_eval.py` receives `--aspects`, it evaluates only
questions in that frozen silver annotation file and additionally reports
Weighted Aspect Recall@5/10 and alpha-nDCG@10. Answer-only aspects are not
retrieval targets. Aspect construction and the freeze boundary are documented
in [`../../docs/RULE_OPTIMIZATION.md`](../../docs/RULE_OPTIMIZATION.md).

For a fair plugin comparison, keep the corpus revision, question IDs, top-k,
filesystem binary, embedding model, cache state, and Neo4j schema fixed. Change
only the retrieval arm. Integrated DSH-agent runs are intentionally located in
`../../dsh_plugin/agent_eval/` because they measure orchestration, model tokens,
tool use, and end-to-end latency in addition to retrieval.

`protocol.json` freezes the machine-readable benchmark contract used by these
retrieval and integrated-agent evaluations.
