#!/usr/bin/env python3
"""Build all public DocsQA question packages from one frozen source manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from dataset.scripts import build as github_build
from dataset.scripts.build_discussion_corpus import (
    CONFIG_DEFAULTS,
    CorpusConfig,
    _is_exact_docs_url,
    build_corpus,
    build_questions,
    fetch_discussions,
    resolve_live_routes,
    write_dataset,
)
from dataset.scripts.combine_datasets import combine_datasets
from dataset.scripts.prepare_unified_corpus import load_source_specs
from dataset_analysis.validate_sources import validate_datasets


PROJECT_TO_DISCUSSION_DATASET = {
    "github-docs": "github_docs",
    "tailwind-css": "tailwind",
    "prisma": "prisma",
    "supabase": "supabase",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _revision(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()


def _materialize_frozen_splits(
    dataset_dir: Path, candidates: list[dict[str, Any]]
) -> dict[str, int]:
    membership = {
        str(row["source_url"]).rstrip("/"): str(row["benchmark_split"])
        for row in candidates
    }
    questions = _load_jsonl(dataset_dir / "questions.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for question in questions:
        source_url = str(question.get("source_url") or "").rstrip("/")
        split = membership.get(source_url)
        if split not in grouped:
            raise ValueError(
                f"missing or invalid frozen split for {source_url}: {split!r}"
            )
        grouped[split].append(question)
    split_dir = dataset_dir / "splits"
    split_dir.mkdir(exist_ok=True)
    for split, rows in grouped.items():
        (split_dir / f"{split}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return {split: len(rows) for split, rows in grouped.items()}


def _discussion_config(project_root: Path, dataset: str) -> CorpusConfig:
    default = CONFIG_DEFAULTS[dataset]
    repo_root = project_root / str(default["repo"])
    content_root = repo_root / str(default["content"])
    if not content_root.is_dir():
        raise FileNotFoundError(
            f"missing pinned documentation checkout: {content_root}; "
            "run evaluation/dataset/scripts/prepare_raw.sh first"
        )
    return CorpusConfig(
        dataset,
        repo_root,
        content_root,
        tuple(default["hosts"]),
        str(default["route_kind"]),
        _revision(repo_root),
    )


def _build_standard_project(
    *,
    project_root: Path,
    project: str,
    dataset: str,
    candidates: list[dict[str, Any]],
    output_dir: Path,
    workers: int,
    refresh: bool,
) -> dict[str, Any]:
    config = _discussion_config(project_root, dataset)
    corpus, corpus_stats = build_corpus(config)
    cache_dir = project_root / "results/cache/discussions" / dataset
    discussions, failures = fetch_discussions(
        candidates, cache_dir, workers, refresh
    )
    hosts = {host.casefold() for host in config.docs_hosts}
    for row in discussions:
        if not row.get("docs_host_links"):
            row["docs_host_links"] = [
                str(url)
                for url in row.get("accepted_answer_links") or []
                if _is_exact_docs_url(str(url), hosts)
            ]
    docs_urls = [
        str(url)
        for row in discussions
        for url in row.get("docs_host_links") or []
    ]
    live_aliases, redirect_failures = resolve_live_routes(
        docs_urls,
        local_routes={str(row["doc_id"]) for row in corpus},
        cache_path=cache_dir / "redirects.json",
        workers=workers,
    )
    questions, question_stats = build_questions(
        config, corpus, discussions, live_aliases
    )
    question_stats["live_redirect_aliases"] = len(live_aliases)
    question_stats["redirect_failures"] = redirect_failures
    manifest = write_dataset(
        output_dir,
        config,
        corpus,
        questions,
        corpus_stats,
        question_stats,
        failures,
    )
    manifest["project"] = project
    return manifest


def build_all(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[3]
    source_rows = _load_jsonl(args.source_file)
    by_dataset = {
        dataset: [
            row
            for row in source_rows
            if str(row.get("dataset") or "") == dataset
        ]
        for dataset in PROJECT_TO_DISCUSSION_DATASET.values()
    }
    missing = [name for name, rows in by_dataset.items() if not rows]
    if missing:
        raise ValueError(f"source manifest has no rows for: {', '.join(missing)}")

    projects_root = args.output_root / "projects"
    projects_root.mkdir(parents=True, exist_ok=True)
    discussion_dirs: dict[str, Path] = {}
    project_manifests: dict[str, dict[str, Any]] = {}

    github_repo = project_root / "results/cache/docsqa-source-repos/github-docs"
    if not github_repo.is_dir():
        raise FileNotFoundError(
            f"missing pinned documentation checkout: {github_repo}; "
            "run evaluation/dataset/scripts/prepare_raw.sh first"
        )
    github_cache = project_root / "results/cache/discussions/github_docs"
    _, github_failures = fetch_discussions(
        by_dataset["github_docs"], github_cache, args.workers, args.refresh
    )
    github_output = projects_root / "github-docs"
    project_manifests["github-docs"] = github_build.prepare(
        github_repo, github_cache, github_output, _revision(github_repo)
    )
    project_manifests["github-docs"]["frozen_splits"] = _materialize_frozen_splits(
        github_output, by_dataset["github_docs"]
    )
    project_manifests["github-docs"]["fetch_failures"] = github_failures
    discussion_dirs["github_docs"] = github_cache

    for project, dataset in PROJECT_TO_DISCUSSION_DATASET.items():
        if project == "github-docs":
            continue
        project_manifests[project] = _build_standard_project(
            project_root=project_root,
            project=project,
            dataset=dataset,
            candidates=by_dataset[dataset],
            output_dir=projects_root / project,
            workers=args.workers,
            refresh=args.refresh,
        )
        project_manifests[project]["frozen_splits"] = _materialize_frozen_splits(
            projects_root / project, by_dataset[dataset]
        )
        discussion_dirs[dataset] = project_root / "results/cache/discussions" / dataset

    validation_inputs = {
        dataset: projects_root / project
        for project, dataset in PROJECT_TO_DISCUSSION_DATASET.items()
    }
    validation_rows, validation_report = validate_datasets(
        validation_inputs, discussion_dirs
    )
    args.validation_output.mkdir(parents=True, exist_ok=True)
    for name, predicate in {
        "per_question": lambda row: True,
        "structurally_eligible": lambda row: row["status"] == "accepted_structurally",
        "rejected": lambda row: row["status"] == "rejected",
    }.items():
        with (args.validation_output / f"{name}.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in validation_rows:
                if predicate(row):
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.validation_output / "report.json").write_text(
        json.dumps(validation_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    combined = combine_datasets(
        {project: projects_root / project for project in PROJECT_TO_DISCUSSION_DATASET},
        load_source_specs(args.config),
        args.validation_output / "structurally_eligible.jsonl",
        args.output_root / "combined",
        force=args.force,
    )
    report = {
        "source_records": len(source_rows),
        "projects": project_manifests,
        "source_validation": validation_report,
        "combined": combined,
    }
    (args.output_root / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-file",
        type=Path,
        default=project_root
        / "evaluation/dataset/templates/discussion_sources.jsonl",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "evaluation/dataset/templates/public_sources.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "evaluation/dataset/evaluation_data",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=project_root / "results/runs/dataset-analysis/source-validation",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    print(json.dumps(build_all(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
