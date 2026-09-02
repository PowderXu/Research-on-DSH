"""Build the Neo4j arm from its plugin-owned local data store."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .data_paths import arm_data_layout
from .semantic_store import load_semantic_artifact


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument(
        "--neo4j-uri",
        default=os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7688"),
    )
    parser.add_argument(
        "--neo4j-username", default=os.environ.get("NEO4J_USERNAME", "neo4j")
    )
    parser.add_argument(
        "--neo4j-password",
        default=os.environ.get("NEO4J_PASSWORD", "secretgraph"),
    )
    parser.add_argument(
        "--neo4j-database", default=os.environ.get("NEO4J_DATABASE", "neo4j")
    )
    parser.add_argument("--kggen-artifact", type=Path)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    from kbbench.plugin_eval import Neo4jGitHubDocsGraphRAG
    from kbbench.retrieval import build_chunks, build_or_load_embeddings

    layout = arm_data_layout("neo4j", args.data_root).ensure()
    corpus_rows = _jsonl(layout.corpus / "corpus.jsonl")
    chunks = build_chunks(corpus_rows)
    model = SentenceTransformer(
        args.embedding_model,
        device=args.device,
        local_files_only=not args.allow_model_download,
    )
    embeddings = build_or_load_embeddings(
        chunks,
        model,
        args.embedding_model,
        layout.indexes,
    )
    kggen_artifact_path = args.kggen_artifact
    if kggen_artifact_path is None:
        candidate = layout.artifacts / "kggen_graph.json"
        kggen_artifact_path = candidate if candidate.exists() else None

    graph = Neo4jGitHubDocsGraphRAG(
        args.neo4j_uri,
        args.neo4j_username,
        args.neo4j_password,
        args.neo4j_database,
    )
    try:
        stats = graph.ingest(
            corpus_rows,
            chunks,
            embeddings,
            embedding_model_name=args.embedding_model,
            semantic_artifact=load_semantic_artifact(kggen_artifact_path),
        )
    finally:
        graph.close()

    output = {
        "neo4j_uri": args.neo4j_uri,
        "data_root": str(layout.root),
        "kggen_artifact": str(kggen_artifact_path) if kggen_artifact_path else None,
        **asdict(stats),
    }
    (layout.artifacts / "graph_stats.json").write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
