# GitHub Docs real-question retrieval benchmark

## Decision summary

For this corpus, use **BM25 + HNSW with reciprocal-rank fusion (RRF), followed by a small residual expansion over explicit Markdown links**. Do not make directory routing the primary retriever, and do not add the tested generic MS MARCO cross-encoder reranker.

On the frozen 246-question test set, always-on edge-context graph expansion improved BM25+HNSW RRF from **Recall@10 0.493 / Hit@10 0.537 / nDCG@10 0.337 / p50 17.3 ms** to **0.518 / 0.561 / 0.355 / 19.5 ms**. The gain is modest overall and its 95% paired-bootstrap interval crosses zero. On the predeclared graph-opportunity subset, however, the nDCG@10 gain is **+0.051, 95% CI [0.011, 0.090]**.

Graph expansion is therefore useful as a low-cost residual, not as a replacement for indexed retrieval. Directory routing made both standalone BM25 and graph combinations worse.

## Frozen benchmark

- Corpus: `github/docs/content` at commit `c34e3dccad00f61133c799d20e7d1208a0e6cc92` (2026-08-20).
- Corpus size: 3,740 Markdown pages and 19,829 retrieval chunks.
- Graph V2: 13,841 resolved Markdown-link edges with anchor text, source section, local context, and target anchor.
- Questions: 328 public GitHub Community questions with a marked accepted answer that links to GitHub Docs.
- Qrels: 421 accepted-answer page labels covering 270 distinct documentation pages.
- Split: 82 development questions and 246 held-out test questions.
- Leakage control: all visible HTTP(S) URLs are removed from the user question before retrieval.
- Qrel resolution: canonical content paths and repository-authored `redirect_from` mappings are primary; two unambiguous unique-slug resolutions are retained and flagged.
- Evaluation unit: canonical page. URL anchors are retained for evidence-shape diagnostics.
- Cost: local open-source embedding and reranking models only; OpenAI/API spend was **$0**.

The questions file has the same SHA-256 in V1 and V2:

`3dcd8f57c1156d2ee88fe20d7ebc2c2122b37420921a1eaa3247e79743d98317`

## Categories derived from the real data

The taxonomy was constructed after inspecting the corpus and accepted-answer evidence, then frozen before the V2 test.

| Evidence category | Meaning | Dev | Test | Expected graph value |
|---|---|---:|---:|---|
| `single_page_direct` | One gold page, no cited anchor | 41 | 95 | Low |
| `single_page_section` | One gold page with a cited section | 13 | 72 | Low to medium |
| `single_page_composed_anchor` | The cited section composes reusable Markdown content | 8 | 29 | Medium |
| `single_page_variant_anchor` | The cited section is version/plan conditioned and the question contains a variant cue | 0 | 5 | Medium, but underpowered |
| `multi_page_linked` | Multiple accepted-answer pages connected by an explicit docs link | 12 | 21 | High |
| `multi_page_dispersed` | Multiple accepted-answer pages without a direct link | 8 | 24 | Low for one-hop link expansion |

The separate intent facet contains `troubleshooting`, `how_to_configuration`, `capability_limit`, `explanation_comparison`, `policy_billing_account`, and `other_product_question`. Full intent tables are in the machine-readable report.

## Iteration 1: untyped graph plus generic reranking

All methods used the same V1 graph schema and snapshot.

| Architecture | Recall@10 | Hit@10 | nDCG@10 | p50 ms |
|---|---:|---:|---:|---:|
| BM25 | 0.381 | 0.415 | 0.243 | 4.0 |
| HNSW dense | 0.470 | 0.524 | 0.306 | 18.2 |
| BM25 + HNSW RRF | **0.493** | **0.537** | **0.337** | 17.6 |
| RRF + generic reranker | 0.405 | 0.443 | 0.261 | 643.7 |
| Pure routing + BM25 | 0.288 | 0.321 | 0.196 | 3.9 |
| RRF + soft hierarchy + reranker | 0.380 | 0.419 | 0.251 | 627.9 |
| RRF + conditional links + reranker | 0.400 | 0.439 | 0.260 | 783.3 |
| RRF + hierarchy + conditional links + reranker | 0.366 | 0.402 | 0.246 | 679.4 |

Development-only oracle analysis showed that expanding explicit neighbors from RRF's top eight seeds could increase gold coverage on linked multi-page questions from 0.604 to 0.889. The graph contained useful edges; the generic reranker destroyed the stronger RRF order and failed to promote those edges effectively.

## Iteration 2: contextual link edges and residual graph scoring

V2 changed the schema between iterations, then froze it for every V2 architecture:

- `LINKS_TO` gained `anchor_text`, `source_section`, `local_context`, and `target_anchor` properties.
- RRF became a non-destructive base. Link-neighbor scores are a residual derived from seed rank and query-to-edge-context BM25 relevance.
- The generic reranker was removed.

Development calibration selected graph residual weight `0.25`. A strict conditional trigger (`best incident edge-context rank == 1`) and always-expansion were both frozen for test. Routing+graph was retained as an ablation.

### Overall held-out test

| Architecture | R@5 | R@10 | R@20 | Hit@1 | Hit@5 | Hit@10 | nDCG@10 | p50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.298 | 0.381 | 0.476 | 0.134 | 0.329 | 0.415 | 0.243 | 3.8 |
| HNSW dense | 0.385 | 0.470 | 0.603 | 0.167 | 0.435 | 0.524 | 0.306 | 17.3 |
| BM25 + HNSW RRF | 0.419 | 0.493 | 0.569 | **0.215** | 0.463 | 0.537 | 0.337 | 17.3 |
| Pure routing + BM25 | 0.239 | 0.288 | 0.369 | 0.122 | 0.276 | 0.321 | 0.196 | **3.8** |
| RRF + conditional contextual graph | 0.420 | 0.503 | 0.613 | **0.215** | 0.472 | 0.545 | 0.349 | 19.4 |
| RRF + always contextual graph | **0.422** | **0.518** | **0.626** | 0.211 | **0.476** | **0.561** | **0.355** | 19.5 |
| RRF + routing + conditional contextual graph | 0.392 | 0.514 | 0.609 | 0.211 | 0.443 | 0.557 | 0.347 | 19.8 |

### Evidence-category results

This compact table compares the strongest non-graph foundation with the two graph policies. The JSON report contains every architecture and every metric.

| Test category | n | Architecture | Recall@10 | Hit@10 | nDCG@10 | p50 ms |
|---|---:|---|---:|---:|---:|---:|
| Multi-page linked | 21 | RRF | 0.387 | 0.667 | 0.366 | 17.5 |
|  |  | Conditional graph | 0.450 | 0.667 | 0.399 | 19.4 |
|  |  | Always graph | **0.450** | **0.667** | **0.411** | 19.6 |
| Multi-page dispersed | 24 | RRF | **0.336** | 0.542 | **0.256** | 17.6 |
|  |  | Conditional graph | 0.308 | 0.542 | 0.230 | 19.6 |
|  |  | Always graph | 0.329 | **0.583** | 0.250 | 19.5 |
| Reusable-composed anchor | 29 | RRF | 0.724 | 0.724 | 0.485 | 17.8 |
|  |  | Conditional graph | **0.793** | **0.793** | **0.557** | 19.7 |
|  |  | Always graph | **0.793** | **0.793** | 0.550 | 19.6 |
| Variant-conditioned anchor | 5 | RRF | **1.000** | **1.000** | **0.590** | 20.0 |
|  |  | Conditional graph | 0.800 | 0.800 | 0.586 | 22.4 |
|  |  | Always graph | 0.800 | 0.800 | 0.586 | 22.4 |
| Single-page section | 72 | RRF | 0.417 | 0.417 | 0.291 | 16.7 |
|  |  | Conditional graph | 0.444 | 0.444 | 0.297 | 19.1 |
|  |  | Always graph | **0.458** | **0.458** | **0.303** | 18.8 |
| Single-page direct | 95 | RRF | 0.516 | 0.516 | 0.327 | 17.4 |
|  |  | Conditional graph | 0.505 | 0.505 | 0.332 | 19.4 |
|  |  | Always graph | **0.526** | **0.526** | **0.336** | 19.5 |

On the 55 predeclared graph-opportunity test questions, RRF scored Recall@10 0.620 / Hit@10 0.727 / nDCG@10 0.449. Always graph scored **0.663 / 0.745 / 0.500** at p50 19.9 ms.

## Paired uncertainty analysis

The table compares always graph with RRF using 10,000 paired bootstrap resamples, seed 17.

| Slice | n | Metric | Mean delta | 95% CI |
|---|---:|---|---:|---:|
| All test | 246 | Recall@10 | +0.025 | [-0.007, +0.058] |
| All test | 246 | Hit@10 | +0.024 | [-0.008, +0.057] |
| All test | 246 | nDCG@10 | +0.018 | [-0.004, +0.038] |
| Graph opportunity | 55 | nDCG@10 | **+0.051** | **[+0.011, +0.090]** |
| Multi-page linked | 21 | Recall@10 | +0.063 | [0.000, +0.159] |
| Multi-page linked | 21 | nDCG@10 | **+0.045** | **[+0.006, +0.093]** |
| Reusable-composed anchor | 29 | Recall@10 | +0.069 | [0.000, +0.172] |
| Reusable-composed anchor | 29 | nDCG@10 | **+0.064** | **[+0.009, +0.121]** |

The overall gain is not statistically conclusive at this sample size. The ranking gain is supported on the graph-opportunity, linked multi-page, and reusable-composed slices.

## Interpretation

1. **The indexed composite foundation is essential.** RRF is substantially stronger than BM25, HNSW alone overall, and pure routing.
2. **Graph value depends on evidence topology.** Contextual links help linked and reusable-composed evidence, but do not solve dispersed multi-page evidence.
3. **Graph should not replace retrieval.** V2 succeeds because graph is a residual on RRF. The V1 graph-on-reranker design was slower and less accurate.
4. **Routing is not useful as a mandatory first layer here.** Pure routing loses 0.141 nDCG@10 versus RRF. Adding routing to graph also reduces nDCG and Recall@5 versus graph alone.
5. **The off-the-shelf reranker is the wrong domain fit.** It adds roughly 626 ms p50 over RRF while reducing nDCG@10 by 0.076.
6. **Always expansion beats the current conditional trigger by a small margin.** Because expansion adds only about 2.1 ms p50, the production default should currently be always-on with a small residual weight. A learned query planner is not justified by these data yet.

## Limitations and next valid iteration

- A marked Community answer is real user evidence, but it is not a guarantee that the answer or every cited page is correct. The benchmark measures retrieval of accepted-answer citations, not factual answer correctness.
- Graph traversal ran in-process over the frozen backend-neutral schema to isolate retrieval policy. These numbers do not include Neo4j client/network latency; loading the same V2 nodes and edges into Neo4j is an integration step, not a new retrieval algorithm.
- Only five version-conditioned questions are in test and none are in development. The observed regression must not be used to tune and retest on this same holdout.
- Link expansion is one hop. Dispersed multi-page questions need entity/dependency extraction, not more aggressive link walking.
- A V3 schema should add typed product/configuration/prerequisite entities and version-specific rendered views. It requires a new untouched question holdout or nested cross-validation before making a confirmatory claim.

## Reproducibility artifacts

- V1 dataset and schema: `evaluation/github_docs_v1/`
- V2 dataset and schema: `evaluation/github_docs_v2/`
- V1 per-question results: `results/github_docs_v1_iteration1/per_query.jsonl`
- V1 report: `results/github_docs_v1_iteration1/report.json`
- V2 held-out per-question results: `results/github_docs_v2_test/per_query.jsonl`
- V2 report with evidence and intent tables: `results/github_docs_v2_test/report.json`
- Benchmark preparation: `kbbench/github_docs_benchmark.py`
- Retrieval evaluation: `kbbench/github_docs_eval.py`

The embedding cache is keyed by the full chunk text and model name. Index construction is excluded from latency; p50 includes warm query embedding, BM25, HNSW, fusion, routing/edge scoring where applicable, and reranking where applicable.
