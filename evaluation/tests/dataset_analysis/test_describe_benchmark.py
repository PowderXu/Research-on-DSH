from __future__ import annotations

import json
from pathlib import Path

from dataset_analysis.describe_benchmark import describe


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_describe_separates_question_images_and_corpus_support(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_jsonl(
        dataset / "corpus.jsonl",
        [
            {
                "doc_id": "d1",
                "rendered_text": "```sh\nrun\n```",
                "link_edges": [{"target_id": "d2"}],
                "image_texts": [{"text": "dialog"}],
            }
        ],
    )
    _write_jsonl(
        dataset / "questions.jsonl",
        [
            {
                "project": "docs",
                "split": "test",
                "qrel_count": 1,
                "evidence_structure": "single",
                "evidence_category": "single_page",
                "intent_category": "how_to",
                "question_images": [{"source_url": "q.png"}],
            }
        ],
    )
    aspects = tmp_path / "aspects.jsonl"
    _write_jsonl(
        aspects,
        [
            {
                "aspects": [
                    {"critical": True, "retrieval_doc_ids": ["d1"]},
                    {"critical": False, "retrieval_doc_ids": []},
                ]
            }
        ],
    )

    report = describe(dataset, aspects)

    assert report["question_images"]["questions_with_images"] == 1
    assert report["corpus_structure"]["markdown_link_edges"] == 1
    assert report["aspect_support"]["document_supported_aspects"] == 1
    assert report["aspect_support"]["questions_with_document_supported_critical_aspect"] == 1
