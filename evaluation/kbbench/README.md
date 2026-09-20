# KB evaluation

This directory evaluates the already-built package under
`../dataset/evaluation_data/normalized/`. It does not construct or mutate the
dataset. Question inputs are in `questions.jsonl`; reference answers and qrels
are in `answers.jsonl`, joined by `question_id`. All questions form one pool.
Final arm comparisons use the plugin-owned filesystem workspace
materialized from that same corpus, so every arm sees identical image-derived
text and canonical pages.

## Evaluation paths

- `plugin_eval.py`: run and report the matched three-arm retrieval comparison;
- `backends.py`: filesystem and Neo4j engines, also used by the live service;
- `retrieval.py`: retrieval-component and graph-expansion ablations;
- `scoring.py`: source resolution, binary IR scores, and aspect coverage;
- `provenance.py`: file fingerprints and redacted execution metadata;
- `indexes.py`: the BM25 index used by the local retrieval implementation.

The agent runner and report validator share their runtime manifest through
`../../dsh_plugin/agent_eval/runtime.py`. Moved implementations remain covered
by the run fingerprint. Older imports of retrieval metrics and plugin engines
remain available, and report fields retain their existing names.

The primary metrics are Recall@5/10, Hit@1/5/10, nDCG@10, AllSupport@10, and
warm p50 retrieval latency. p95 latency and offline construction costs are
optional diagnostics when a run records them. Reports are sliced by question
intent, evidence category, evidence structure, and qrel count. An unjudged page
receives no gain because the accepted-answer qrels are sparse; this does not
prove that page is irrelevant.

The `retrieval.py` component evaluator defaults to `--search-scope project`:
warm-up and measured searches receive all corpus document IDs matching the
question's input `project`. Qrels only validate that scope; they do not select
candidates. Missing project metadata or qrels outside the project fail before
indexing. Scope filters apply before candidate limits, route selection, and
link traversal, so an outside page cannot consume graph slots or bridge a path.
Reports record `search_scope`, question `project`, and `scope_documents`.
Use `--search-scope corpus` explicitly for the former unrestricted search.

This entry point keeps a shared index: BM25 statistics come from the full
corpus, and scoped dense search uses exact cosine over permitted chunks.
It is therefore not identical to building a separate BM25/HNSW index per
product. Historical results must retain their original scope and index policy;
the Step 2 five-baseline results used separate product indexes and are unchanged
by this fix. This option does not change `plugin_eval.py` or agent-run scope.

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
