# KB evaluation

This directory evaluates an already-built dataset from `../dataset/data/`. It
does not construct or mutate the dataset.

## Evaluation paths

- `plugin_eval.py`: matched filesystem, BM25+HNSW, and Neo4j retrieval arms;
- `retrieval.py`: retrieval-component and graph-expansion ablations;
- `evaluate_predictions.py`: backend-neutral scoring for ranked prediction files;
- `scoring.py`: canonical source resolution and shared IR metrics;
- `indexes.py`: the BM25 index used by the local retrieval implementation.

The primary metrics are Recall@k, Hit@k, nDCG@10, AllSupport@10, and warm p50/
p95 retrieval latency. Reports are sliced by question intent, evidence category,
evidence structure, and qrel count. An unjudged page receives no gain because the
accepted-answer qrels are sparse; this does not prove that page is irrelevant.

For a fair plugin comparison, keep the corpus revision, question IDs, top-k,
filesystem binary, embedding model, cache state, and Neo4j schema fixed. Change
only the retrieval arm. Integrated DSH-agent runs are intentionally located in
`../../dsh_plugin/agent_eval/` because they measure orchestration, model tokens,
tool use, and end-to-end latency in addition to retrieval.

`protocol.json` and `prediction_schema.json` define the backend-neutral ranked
prediction contract. `evaluate_predictions.py` rejects duplicate, unknown, or
missing question IDs by default.
