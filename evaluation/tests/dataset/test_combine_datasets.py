from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from dataset.scripts.combine_datasets import combine_datasets
from dataset.scripts.prepare_unified_corpus import SourceSpec


PROJECTS = ("github-docs", "tailwind-css", "prisma", "supabase")
VALIDATION_NAMES = {
    "github-docs": "github_docs",
    "tailwind-css": "tailwind",
    "prisma": "prisma",
    "supabase": "supabase",
}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _source_package(root: Path, project: str) -> Path:
    package = root / project
    (package / "splits").mkdir(parents=True)
    _write_jsonl(
        package / "corpus.jsonl",
        [
            {
                "doc_id": "/guide",
                "source_path": "guide.md",
                "title": "Guide",
                "route": "/",
                "rendered_text": "Answer evidence.",
                "outgoing_ids": ["/other"],
                "outgoing_paths": ["/other"],
                "link_edges": [{"target_id": "/other", "anchor_text": "other"}],
            },
            {
                "doc_id": "/other",
                "source_path": "other.md",
                "title": "Other",
                "route": "/",
                "rendered_text": "Other evidence.",
                "outgoing_ids": [],
                "outgoing_paths": [],
                "link_edges": [],
            },
        ],
    )
    question_id = "1" if project == "github-docs" else f"{VALIDATION_NAMES[project]}-1"
    question = {
        "question_id": question_id,
        "query": "How?",
        "reference_answer": "old answer",
        "qrel_ids": ["/guide"],
        "qrel_anchors": {"/guide": ["answer"]},
        "qrel_count": 1,
        "split": "test",
    }
    _write_jsonl(package / "questions.jsonl", [question])
    for split in ("train", "validation"):
        (package / "splits" / f"{split}.json").write_text("[]\n", encoding="utf-8")
    (package / "splits" / "test.json").write_text(
        json.dumps([question]) + "\n", encoding="utf-8"
    )
    return package


def _spec(project: str) -> SourceSpec:
    return SourceSpec(
        source_id=project,
        display_name=project,
        description=project,
        repository=f"https://github.com/example/{project}.git",
        revision="a" * 40,
        checkout_dir=project,
        normalizer="fixture",
        docs_hosts=("docs.example.com",),
        content_roots=(PurePosixPath("docs"),),
        support_roots=(),
    )


def test_combiner_keeps_only_validated_cases_and_namespaces_every_relation(tmp_path: Path) -> None:
    datasets = {project: _source_package(tmp_path / "datasets", project) for project in PROJECTS}
    eligible_rows = []
    for project in PROJECTS:
        question_id = "1" if project == "github-docs" else f"{VALIDATION_NAMES[project]}-1"
        eligible_rows.append(
            {
                "dataset": VALIDATION_NAMES[project],
                "question_id": question_id,
                "status": "accepted_structurally",
                "rejection_reasons": [],
                "expanded_reference_answer": f"validated {project}",
                "resolved_internal_docs_urls": ["https://docs.example/guide"],
                "qrel_ids": ["/guide"],
                "question_images": [],
                "answer_images": [],
                "document_images": {"/guide": []},
                "linked_answer_sources": [],
                "requires_multimodal_judgment": False,
            }
        )
    eligible = tmp_path / "eligible.jsonl"
    _write_jsonl(eligible, eligible_rows)
    output = tmp_path / "combined"

    manifest = combine_datasets(
        datasets, [_spec(project) for project in PROJECTS], eligible, output, force=False
    )

    assert manifest["documents"] == 8
    assert manifest["questions"] == 4
    questions = [json.loads(line) for line in (output / "questions.jsonl").read_text().splitlines()]
    assert {row["project"] for row in questions} == set(PROJECTS)
    assert all(row["source_validation"] == "accepted_structurally" for row in questions)
    assert all(row["reference_answer"].startswith("validated ") for row in questions)
    assert all(row["qrel_ids"] == [f"{row['project']}::/guide"] for row in questions)
    corpus = [json.loads(line) for line in (output / "corpus.jsonl").read_text().splitlines()]
    github_guide = next(row for row in corpus if row["doc_id"] == "github-docs::/guide")
    assert github_guide["source_path"] == "github-docs/docs/guide.md"
    assert github_guide["outgoing_ids"] == ["github-docs::/other"]
    assert github_guide["link_edges"][0]["target_id"] == "github-docs::/other"
