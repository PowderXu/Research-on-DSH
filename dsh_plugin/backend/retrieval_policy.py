"""Frozen production-style candidate policy for DocsQA retrieval.

The numbers are stage-specific rather than a claim that one universal
"production top-k" exists.  The rationale and primary references are recorded
in ``docs/EVALUATION_PROTOCOL.md``.
"""

from __future__ import annotations

from typing import Any


POLICY_ID = "production-window-v1"

# BM25 and HNSW each contribute this many documents to RRF.  The union can be
# at most twice this size before deduplication.
RRF_CANDIDATES_PER_RETRIEVER = 50

# The plugin and benchmark expose the same citation-sized final list.
FINAL_RESULT_LIMIT = 10

# Neo4j traversal starts from a small verified seed set and has a true global
# per-seed cap across every relationship branch, not just Markdown links.
GRAPH_SEED_LIMIT = 5
GRAPH_CANDIDATES_PER_SEED = 10
GRAPH_MAX_HOPS = 2

# Hub suppression is independent of the candidate window.  These bounds stop
# generic entities/routes from becoming traversal shortcuts.
GRAPH_ENTITY_DEGREE_CAP = 20
GRAPH_ROUTE_PAGE_CAP = 20

# Neo4j GraphRAG documents 1 as the default.  With a first-stage top-k of 50,
# using 2 would silently request 100 vector candidates.
NEO4J_EFFECTIVE_SEARCH_RATIO = 1


def retrieval_contract(arm: str) -> dict[str, Any]:
    """Return the serializable contract used in evaluation fingerprints."""

    contract: dict[str, Any] = {
        "policy_id": POLICY_ID,
        "final_result_limit": FINAL_RESULT_LIMIT,
    }
    if arm in {"hybrid", "neo4j"}:
        contract.update(
            {
                "rrf_candidates_per_retriever": RRF_CANDIDATES_PER_RETRIEVER,
                "rrf_union_upper_bound": 2 * RRF_CANDIDATES_PER_RETRIEVER,
            }
        )
    if arm == "neo4j":
        contract.update(
            {
                "graph_seed_limit": GRAPH_SEED_LIMIT,
                "graph_candidates_per_seed": GRAPH_CANDIDATES_PER_SEED,
                "graph_candidate_upper_bound": (
                    GRAPH_SEED_LIMIT * GRAPH_CANDIDATES_PER_SEED
                ),
                "graph_max_hops": GRAPH_MAX_HOPS,
                "graph_entity_degree_cap": GRAPH_ENTITY_DEGREE_CAP,
                "graph_route_page_cap": GRAPH_ROUTE_PAGE_CAP,
                "neo4j_effective_search_ratio": NEO4J_EFFECTIVE_SEARCH_RATIO,
            }
        )
    return contract
