# GitHub Docs matched real-agent performance report

This report contains **5 identical real-model questions per arm**. It is a paired engineering pilot, not the final held-out benchmark.

## Overall

| Arm | Recall@10 | Hit@10 | nDCG@10 | Visible Hit@10 (diagnostic) | Visible nDCG@10 (diagnostic) | p50 latency | Tokens / QA | Tool calls / QA | Failures | Graph use |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fs | 0.600 | 0.600 | 0.377 | 0.200 | 0.126 | 35.01 s | 193,230 | 11.2 | 0.000 | 0.000 |
| hybrid | 0.600 | 0.600 | 0.312 | 0.600 | 0.367 | 29.82 s | 36,793 | 5.6 | 0.000 | 0.000 |
| neo4j | 0.200 | 0.200 | 0.126 | 0.600 | 0.463 | 18.41 s | 15,367 | 3.2 | 0.000 | 0.000 |
| codex-fastctx | 0.200 | 0.200 | 0.200 | 0.000 | 0.000 | 34.95 s | 146,444 | 13.0 | 0.000 | 0.000 |

## By evidence category

| Arm | Category | N | Recall@10 | Hit@10 | nDCG@10 | p50 latency | Tokens / QA |
|---|---|---:|---:|---:|---:|---:|---:|
| fs | single_page_composed_anchor | 3 | 0.667 | 0.667 | 0.500 | 21.47 s | 109,259 |
| fs | single_page_section | 2 | 0.500 | 0.500 | 0.193 | 49.25 s | 319,188 |
| hybrid | single_page_composed_anchor | 3 | 1.000 | 1.000 | 0.521 | 28.59 s | 31,190 |
| hybrid | single_page_section | 2 | 0.000 | 0.000 | 0.000 | 32.73 s | 45,198 |
| neo4j | single_page_composed_anchor | 3 | 0.333 | 0.333 | 0.210 | 18.41 s | 18,887 |
| neo4j | single_page_section | 2 | 0.000 | 0.000 | 0.000 | 20.78 s | 10,086 |
| codex-fastctx | single_page_composed_anchor | 3 | 0.333 | 0.333 | 0.333 | 33.19 s | 125,620 |
| codex-fastctx | single_page_section | 2 | 0.000 | 0.000 | 0.000 | 43.40 s | 177,680 |

## By intent category

| Arm | Category | N | Recall@10 | Hit@10 | nDCG@10 | p50 latency | Tokens / QA |
|---|---|---:|---:|---:|---:|---:|---:|
| fs | policy_billing_account | 3 | 0.333 | 0.333 | 0.167 | 35.01 s | 188,508 |
| fs | troubleshooting | 2 | 1.000 | 1.000 | 0.693 | 42.48 s | 200,314 |
| hybrid | policy_billing_account | 3 | 0.667 | 0.667 | 0.310 | 35.65 s | 35,101 |
| hybrid | troubleshooting | 2 | 0.500 | 0.500 | 0.315 | 27.07 s | 39,332 |
| neo4j | policy_billing_account | 3 | 0.333 | 0.333 | 0.210 | 15.68 s | 15,324 |
| neo4j | troubleshooting | 2 | 0.000 | 0.000 | 0.000 | 27.02 s | 15,431 |
| codex-fastctx | policy_billing_account | 3 | 0.000 | 0.000 | 0.000 | 34.95 s | 137,455 |
| codex-fastctx | troubleshooting | 2 | 0.500 | 0.500 | 0.500 | 42.52 s | 159,927 |

## By evidence structure

| Arm | Category | N | Recall@10 | Hit@10 | nDCG@10 | p50 latency | Tokens / QA |
|---|---|---:|---:|---:|---:|---:|---:|
| fs | single | 5 | 0.600 | 0.600 | 0.377 | 35.01 s | 193,230 |
| hybrid | single | 5 | 0.600 | 0.600 | 0.312 | 29.82 s | 36,793 |
| neo4j | single | 5 | 0.200 | 0.200 | 0.126 | 18.41 s | 15,367 |
| codex-fastctx | single | 5 | 0.200 | 0.200 | 0.200 | 34.95 s | 146,444 |

## By qrel count

| Arm | Category | N | Recall@10 | Hit@10 | nDCG@10 | p50 latency | Tokens / QA |
|---|---|---:|---:|---:|---:|---:|---:|
| fs | 1 | 5 | 0.600 | 0.600 | 0.377 | 35.01 s | 193,230 |
| hybrid | 1 | 5 | 0.600 | 0.600 | 0.312 | 29.82 s | 36,793 |
| neo4j | 1 | 5 | 0.200 | 0.200 | 0.126 | 18.41 s | 15,367 |
| codex-fastctx | 1 | 5 | 0.200 | 0.200 | 0.200 | 34.95 s | 146,444 |

## Interpretation boundary

- Only five paired real-model questions are included; confidence intervals and final ranking claims are not warranted.
- The sample contains no multi_page_linked questions.
- The Neo4j arm made no techdocs_expand calls, so differences cannot be attributed to graph traversal.
- Accepted-answer citation qrels are incomplete and may drift relative to the pinned current documentation corpus.
- Latency is end-to-end agent latency; provider-only latency was not recorded in these trajectories.
- Primary retrieval metrics rank the agent's final ordered sources. First-visible tool documents are retained only as diagnostic fields.
- Requested/actual models differ by arm; accuracy and efficiency differences are descriptive system results, not a harness-only causal estimate.
- Codex receives the matched policy as both a project skill and invocation-scoped developer instructions; DSH loads its translation through the skill plugin.
