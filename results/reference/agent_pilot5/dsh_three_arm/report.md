# GitHub Docs matched real-DSH performance report

This report contains **5 identical real-model questions per arm**. It is a paired engineering pilot, not the final held-out benchmark.

## Overall

| Arm | Recall@10 | Hit@10 | nDCG@10 | p50 latency | Tokens / QA | Tool calls / QA | Failures | Graph use |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fs | 0.200 | 0.200 | 0.126 | 35.01 s | 193,230 | 11.2 | 0.000 | 0.000 |
| hybrid | 0.600 | 0.600 | 0.367 | 29.82 s | 36,793 | 5.6 | 0.000 | 0.000 |
| neo4j | 0.600 | 0.600 | 0.463 | 18.41 s | 15,367 | 3.2 | 0.000 | 0.000 |

## By evidence category

| Arm | Category | N | Recall@10 | Hit@10 | nDCG@10 | p50 latency | Tokens / QA |
|---|---|---:|---:|---:|---:|---:|---:|
| fs | single_page_composed_anchor | 3 | 0.333 | 0.333 | 0.210 | 21.47 s | 109,259 |
| fs | single_page_section | 2 | 0.000 | 0.000 | 0.000 | 49.25 s | 319,188 |
| hybrid | single_page_composed_anchor | 3 | 0.667 | 0.667 | 0.500 | 28.59 s | 31,190 |
| hybrid | single_page_section | 2 | 0.500 | 0.500 | 0.167 | 32.73 s | 45,198 |
| neo4j | single_page_composed_anchor | 3 | 0.667 | 0.667 | 0.667 | 18.41 s | 18,887 |
| neo4j | single_page_section | 2 | 0.500 | 0.500 | 0.158 | 20.78 s | 10,086 |

## By intent category

| Arm | Category | N | Recall@10 | Hit@10 | nDCG@10 | p50 latency | Tokens / QA |
|---|---|---:|---:|---:|---:|---:|---:|
| fs | policy_billing_account | 3 | 0.000 | 0.000 | 0.000 | 35.01 s | 188,508 |
| fs | troubleshooting | 2 | 0.500 | 0.500 | 0.315 | 42.48 s | 200,314 |
| hybrid | policy_billing_account | 3 | 0.667 | 0.667 | 0.278 | 35.65 s | 35,101 |
| hybrid | troubleshooting | 2 | 0.500 | 0.500 | 0.500 | 27.07 s | 39,332 |
| neo4j | policy_billing_account | 3 | 0.667 | 0.667 | 0.438 | 15.68 s | 15,324 |
| neo4j | troubleshooting | 2 | 0.500 | 0.500 | 0.500 | 27.02 s | 15,431 |

## By graph opportunity

| Arm | Category | N | Recall@10 | Hit@10 | nDCG@10 | p50 latency | Tokens / QA |
|---|---|---:|---:|---:|---:|---:|---:|
| fs | false | 2 | 0.000 | 0.000 | 0.000 | 49.25 s | 319,188 |
| fs | true | 3 | 0.333 | 0.333 | 0.210 | 21.47 s | 109,259 |
| hybrid | false | 2 | 0.500 | 0.500 | 0.167 | 32.73 s | 45,198 |
| hybrid | true | 3 | 0.667 | 0.667 | 0.500 | 28.59 s | 31,190 |
| neo4j | false | 2 | 0.500 | 0.500 | 0.158 | 20.78 s | 10,086 |
| neo4j | true | 3 | 0.667 | 0.667 | 0.667 | 18.41 s | 18,887 |

## Interpretation boundary

- Only five paired real-model questions are included; confidence intervals and final ranking claims are not warranted.
- The sample contains no multi_page_linked questions.
- The Neo4j arm made no techdocs_expand calls, so differences cannot be attributed to graph traversal.
- Accepted-answer citation qrels are incomplete and may drift relative to the pinned current documentation corpus.
- Latency is end-to-end DSH latency; provider-only latency was not recorded in these trajectories.
