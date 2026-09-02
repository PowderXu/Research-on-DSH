import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dsh_plugin.backend.graph_records import build_graph_records, build_graph_snapshot
from dsh_plugin.backend.retrieval_policy import (
    GRAPH_CANDIDATES_PER_SEED,
    GRAPH_MAX_HOPS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str


def test_public_graph_schema_matches_runtime_candidate_policy() -> None:
    schema = json.loads(
        (PROJECT_ROOT / "dsh_plugin/plugin/graph_schema.json").read_text(
            encoding="utf-8"
        )
    )
    policy = schema["retrieval_policy"]

    assert policy["link_hops"] == GRAPH_MAX_HOPS
    assert policy["max_graph_candidates_per_seed"] == GRAPH_CANDIDATES_PER_SEED


def test_graph_records_keep_images_in_reading_order_and_near_text() -> None:
    text = """# Configure widgets

Enable widgets first.

![Widget settings](assets/widget.png "Settings")

Then save the project.
"""
    corpus = [
        {
            "doc_id": "product::/widgets",
            "title": "Widgets",
            "source_path": "widgets.md",
            "route": "/widgets",
            "content_type": "article",
            "variant_conditioned": False,
            "rendered_text": text,
        }
    ]
    chunks = [
        Chunk(
            "product::/widgets::c0000",
            "product::/widgets",
            "Widgets",
            "Widgets\nproduct::/widgets\n# Configure widgets\n\nEnable widgets first.",
        ),
        Chunk(
            "product::/widgets::c0001",
            "product::/widgets",
            "Widgets",
            "Widgets\nproduct::/widgets\nThen save the project.",
        ),
    ]

    records = build_graph_records(corpus, chunks)

    assert records["projects"] == [{"project_id": "product"}]
    assert len(records["image_assets"]) == 1
    assert len(records["image_occurrences"]) == 1
    image = records["image_occurrences"][0]
    assert image["alt_text"] == "Widget settings"
    assert "Widget settings" in image["search_text"]
    ordered = [records["next_edges"][0]["source"]]
    ordered.extend(edge["target"] for edge in records["next_edges"])
    assert ordered[1] == image["unit_id"]
    assert records["near_edges"][0]["occurrence_id"] == image["unit_id"]


def test_graph_snapshot_covers_corpus_embeddings_and_semantic_artifact() -> None:
    chunks = [Chunk("c1", "project::/page", "Page", "evidence")]
    corpus = [{"doc_id": "project::/page", "rendered_text": "evidence"}]
    embeddings = np.asarray([[1.0, 0.0]], dtype=np.float32)
    semantic = {
        "schema_version": 1,
        "entities": [],
        "predicates": [],
        "claims": [],
    }

    first = build_graph_snapshot(
        corpus,
        chunks,
        embeddings,
        embedding_model_name="model-a",
        semantic_artifact=semantic,
    )
    same = build_graph_snapshot(
        corpus,
        chunks,
        embeddings.copy(),
        embedding_model_name="model-a",
        semantic_artifact=dict(semantic),
    )
    changed = build_graph_snapshot(
        corpus,
        chunks,
        np.asarray([[0.0, 1.0]], dtype=np.float32),
        embedding_model_name="model-a",
        semantic_artifact=semantic,
    )

    assert first == same
    assert first["snapshot_sha256"] != changed["snapshot_sha256"]
