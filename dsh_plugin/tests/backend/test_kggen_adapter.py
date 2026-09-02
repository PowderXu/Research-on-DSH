from dataclasses import dataclass, field

from dsh_plugin.backend.kggen_adapter import build_semantic_artifact
from dsh_plugin.backend.kggen_scalable import (
    align_relation,
    build_batches,
    build_scalable_semantic_artifact,
    filter_weak_evidence_alignments,
)
from dsh_plugin.backend.semantic_store import validate_semantic_artifact


@dataclass
class Graph:
    entities: set[str] = field(default_factory=set)
    edges: set[str] = field(default_factory=set)
    relations: set[tuple[str, str, str]] = field(default_factory=set)
    entity_clusters: dict[str, list[str]] = field(default_factory=dict)
    edge_clusters: dict[str, list[str]] = field(default_factory=dict)


class FakeKGGen:
    def __init__(self, **_: object) -> None:
        pass

    def generate(self, input_data: str, **_: object) -> Graph:
        if "CLI" in input_data:
            return Graph(
                entities={"CLI", "configuration file"},
                edges={"overrides"},
                relations={("CLI", "overrides", "configuration file")},
            )
        return Graph(
            entities={"command line", "config file"},
            edges={"takes precedence over"},
            relations={("command line", "takes precedence over", "config file")},
        )

    def aggregate(self, graphs: list[Graph]) -> Graph:
        return Graph(
            entities=set().union(*(graph.entities for graph in graphs)),
            edges=set().union(*(graph.edges for graph in graphs)),
            relations=set().union(*(graph.relations for graph in graphs)),
        )

    def cluster(self, graph: Graph, **_: object) -> Graph:
        graph.entity_clusters = {
            "command line": ["CLI", "command line"],
            "configuration file": ["config file", "configuration file"],
        }
        graph.edge_clusters = {
            "overrides": ["overrides", "takes precedence over"]
        }
        return graph


def test_kggen_clustering_preserves_claim_evidence_units() -> None:
    artifact = build_semantic_artifact(
        [
            {
                "unit_id": "doc::c0",
                "search_text": "The CLI overrides the configuration file for this option.",
            },
            {
                "unit_id": "doc::c1",
                "search_text": "The command line takes precedence over the config file.",
            },
        ],
        model="fake",
        context="technical docs",
        kggen_factory=FakeKGGen,
    )

    validate_semantic_artifact(artifact)
    assert artifact["units_processed"] == 2
    assert artifact["projects_processed"] == ["default"]
    assert {row["canonical_name"] for row in artifact["predicates"]} == {"overrides"}
    assert {row["evidence_unit_id"] for row in artifact["claims"]} == {
        "doc::c0",
        "doc::c1",
    }
    assert all(row["evidence_excerpt"] for row in artifact["claims"])
    assert all("question" not in row for row in artifact["claims"])


def test_kggen_entities_are_scoped_to_independent_projects() -> None:
    artifact = build_semantic_artifact(
        [
            {
                "unit_id": "prisma::c0",
                "doc_id": "prisma::/config",
                "search_text": "The CLI overrides the configuration file for this option.",
            },
            {
                "unit_id": "supabase::c0",
                "doc_id": "supabase::/config",
                "search_text": "The CLI overrides the configuration file for this option.",
            },
        ],
        model="fake",
        context="technical docs",
        kggen_factory=FakeKGGen,
    )

    cli_entities = [
        row for row in artifact["entities"] if row["canonical_name"] == "command line"
    ]
    assert {row["project_id"] for row in cli_entities} == {"prisma", "supabase"}
    assert len({row["entity_id"] for row in cli_entities}) == 2


def test_scalable_kggen_batches_and_aligns_claims(monkeypatch, tmp_path) -> None:
    units = [
        {
            "unit_id": "docs::c0",
            "doc_id": "docs::/config",
            "search_text": "The CLI overrides the configuration file.",
        },
        {
            "unit_id": "docs::c1",
            "doc_id": "docs::/other",
            "search_text": "A dashboard shows deployment status.",
        },
    ]
    batches = build_batches(
        [{**unit, "project_id": "docs"} for unit in units], 2_000
    )
    assert len(batches) == 1
    selected, method = align_relation(
        ("CLI", "overrides", "configuration file"), batches[0].units
    )
    assert selected["unit_id"] == "docs::c0"
    assert method == "both_entities_exact"

    monkeypatch.setattr(
        "dsh_plugin.backend.kggen_scalable._union_find_aliases",
        lambda names, _threshold: {name: {name} for name in names},
    )
    artifact = build_scalable_semantic_artifact(
        units,
        model="fake",
        context="technical docs",
        kggen_factory=FakeKGGen,
        cache_dir=tmp_path / "raw",
        batch_chars=2_000,
        workers=1,
    )

    validate_semantic_artifact(artifact)
    assert artifact["units_processed"] == 2
    assert artifact["batches"] == 1
    assert artifact["claims"][0]["evidence_unit_id"] == "docs::c0"
    assert artifact["claims"][0]["evidence_alignment"] == "both_entities_exact"


def test_filter_weak_evidence_alignments_prunes_orphans() -> None:
    payload = {
        "entities": [
            {"entity_id": "e1", "claim_count": 2},
            {"entity_id": "e2", "claim_count": 1},
            {"entity_id": "e3", "claim_count": 1},
        ],
        "predicates": [{"predicate_id": "p1"}, {"predicate_id": "p2"}],
        "claims": [
            {
                "claim_id": "c1",
                "subject_id": "e1",
                "object_id": "e2",
                "predicate_id": "p1",
                "evidence_alignment": "both_entities_exact",
            },
            {
                "claim_id": "c2",
                "subject_id": "e1",
                "object_id": "e3",
                "predicate_id": "p2",
                "evidence_alignment": "token_overlap",
            },
        ],
    }

    filtered = filter_weak_evidence_alignments(payload)

    assert [row["claim_id"] for row in filtered["claims"]] == ["c1"]
    assert {row["entity_id"] for row in filtered["entities"]} == {"e1", "e2"}
    assert [row["predicate_id"] for row in filtered["predicates"]] == ["p1"]
    assert all(row["claim_count"] == 1 for row in filtered["entities"])
    assert filtered["evidence_alignment"]["rejected"] == 1
