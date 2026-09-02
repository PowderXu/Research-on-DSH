"""Prepare local-only data owned by one DSH DocsQA plugin arm."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from .data_paths import arm_data_layout
from .graph_records import build_graph_records


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _materialize_searchable_documents(
    corpus: list[dict[str, Any]], destination: Path
) -> int:
    """Write the exact corpus representation used by every retrieval arm."""

    written = 0
    for row in corpus:
        relative = PurePosixPath(str(row.get("source_path") or ""))
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe corpus source_path: {relative}")
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(row.get("rendered_text") or ""), encoding="utf-8")
        written += 1
    return written


def prepare(
    *,
    arm: str,
    source_dataset: Path,
    source_documents: Path | None,
    data_root: Path | None,
    copy_documents: bool,
) -> dict[str, Any]:
    layout = arm_data_layout(arm, data_root).ensure()
    source_dataset = source_dataset.resolve()
    shutil.copytree(source_dataset, layout.corpus, dirs_exist_ok=True)
    corpus = _jsonl(layout.corpus / "corpus.jsonl")

    if source_documents is not None:
        source_documents = source_documents.resolve()
        if copy_documents:
            shutil.copytree(source_documents, layout.documents, dirs_exist_ok=True)
        else:
            (layout.root / "documents.path").write_text(
                str(source_documents) + "\n", encoding="utf-8"
            )

    materialized_documents = _materialize_searchable_documents(
        corpus, layout.documents
    )

    artifact_counts: dict[str, int] = {}
    if arm in {"hybrid", "neo4j"}:
        from kbbench.retrieval import build_chunks

        chunks = build_chunks(corpus)
        records = build_graph_records(corpus, chunks)
        _write_jsonl(layout.artifacts / "retrieval_units.jsonl", records["units"])
        _write_jsonl(layout.artifacts / "image_assets.jsonl", records["image_assets"])
        _write_jsonl(
            layout.artifacts / "image_occurrences.jsonl",
            records["image_occurrences"],
        )
        artifact_counts = {
            "retrieval_units": len(records["units"]),
            "image_assets": len(records["image_assets"]),
            "image_occurrences": len(records["image_occurrences"]),
        }

    return {
        "arm": arm,
        "root": str(layout.root),
        "corpus": str(layout.corpus),
        "documents": str(layout.documents),
        "indexes": str(layout.indexes),
        "assets": str(layout.assets),
        "artifacts": str(layout.artifacts),
        "materialized_documents": materialized_documents,
        **artifact_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("fs", "hybrid", "neo4j"), required=True)
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--source-documents", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--copy-documents", action="store_true")
    parser.add_argument("--run-kggen", action="store_true")
    parser.add_argument("--kggen-python", type=Path)
    parser.add_argument("--kggen-model", default="openai/gpt-5.6-luna")
    parser.add_argument("--kggen-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--kggen-base-url")
    parser.add_argument("--kggen-max-units", type=int)
    parser.add_argument("--kggen-batch-chars", type=int, default=12_000)
    parser.add_argument("--kggen-workers", type=int, default=8)
    parser.add_argument("--kggen-max-cost-usd", type=float)
    args = parser.parse_args()

    result = prepare(
        arm=args.arm,
        source_dataset=args.source_dataset,
        source_documents=args.source_documents,
        data_root=args.data_root,
        copy_documents=args.copy_documents,
    )
    if args.run_kggen:
        if args.arm != "neo4j":
            raise SystemExit("KGGen artifacts belong to the neo4j plugin arm")
        if args.kggen_python is None:
            raise SystemExit("--kggen-python is required with --run-kggen")
        if args.kggen_api_key_env == "OPENAI_API_KEY":
            from dsh_plugin.agent_eval.credentials import (
                load_openai_key_from_configured_env,
            )

            if not load_openai_key_from_configured_env():
                raise SystemExit(
                    "OPENAI_API_KEY is unset; set it directly or point "
                    "KBBENCH_OPENAI_ENV_FILE to a private env file"
                )
        artifacts = Path(result["artifacts"])
        indexes = Path(result["indexes"])
        command = [
            str(args.kggen_python),
            "-m",
            "dsh_plugin.backend.kggen_scalable",
            "--input",
            str(artifacts / "retrieval_units.jsonl"),
            "--output",
            str(artifacts / "kggen_graph.json"),
            "--cache-dir",
            str(indexes / "kggen_raw"),
            "--model",
            args.kggen_model,
            "--api-key-env",
            args.kggen_api_key_env,
            "--batch-chars",
            str(args.kggen_batch_chars),
            "--workers",
            str(args.kggen_workers),
        ]
        if args.kggen_max_cost_usd is not None:
            command.extend(("--max-cost-usd", str(args.kggen_max_cost_usd)))
        if args.kggen_base_url:
            command.extend(("--base-url", args.kggen_base_url))
        if args.kggen_max_units is not None:
            command.extend(("--max-units", str(args.kggen_max_units)))
        subprocess.run(command, check=True)
        result["kggen_artifact"] = str(artifacts / "kggen_graph.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
