from __future__ import annotations

import json
from pathlib import Path

from dataset_analysis.validate_sources import extract_qa_html, validate_datasets


def _page(question: str, answer: str) -> str:
    payload = {
        "@type": "QAPage",
        "mainEntity": {
            "@type": "Question",
            "name": "How?",
            "text": question,
            "acceptedAnswer": {"@type": "Answer", "text": answer},
        },
    }
    return '<script type="application/ld+json">' + json.dumps(payload) + "</script>"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _dataset(
    tmp_path: Path,
    questions: list[dict[str, object]],
    pages: dict[str, str],
) -> tuple[Path, Path]:
    dataset_dir = tmp_path / "dataset"
    html_dir = tmp_path / "html"
    dataset_dir.mkdir()
    html_dir.mkdir()
    _write_jsonl(dataset_dir / "questions.jsonl", questions)
    _write_jsonl(
        dataset_dir / "corpus.jsonl",
        [{"doc_id": "/docs/a", "rendered_text": "Answer text."}],
    )
    (dataset_dir / "manifest.json").write_text(
        json.dumps({"docs_hosts": ["docs.example.com"]}), encoding="utf-8"
    )
    for name, value in pages.items():
        (html_dir / f"{name}.html").write_text(value, encoding="utf-8")
    return dataset_dir, html_dir


def _question(number: str, source_url: str) -> dict[str, object]:
    docs_url = "https://docs.example.com/docs/a"
    return {
        "question_id": f"x-{number}",
        "source_url": source_url,
        "docs_urls": [docs_url],
        "qrel_ids": ["/docs/a"],
        "resolution_kinds": {docs_url: "canonical_path"},
    }


def test_extract_qa_html_preserves_pixels_as_references_and_links() -> None:
    parsed = extract_qa_html(
        _page(
            '<p>Error</p><img src="https://cdn.example/q.png" alt="error">',
            '<p>Fix it.</p><a href="https://github.com/acme/repo/issues/2">details</a>'
            '<img src="/answer.png" alt="result">',
        ),
        "https://github.com/acme/repo/discussions/1",
    )
    assert parsed["question_images"] == [
        {"url": "https://cdn.example/q.png", "alt": "error", "title": ""}
    ]
    assert parsed["answer_images"][0]["url"] == "https://github.com/answer.png"
    assert parsed["answer_links"][0]["url"] == "https://github.com/acme/repo/issues/2"


def test_unresolved_internal_documentation_rejects_case(tmp_path: Path) -> None:
    question = _question("1", "https://github.com/acme/repo/discussions/1")
    question["resolution_kinds"] = {}
    dataset_dir, html_dir = _dataset(
        tmp_path, [question], {"1": _page("Question", "Direct answer")}
    )
    rows, _ = validate_datasets({"x": dataset_dir}, {"x": html_dir})
    assert rows[0]["status"] == "rejected"
    assert "unresolved_internal_documentation" in rows[0]["rejection_reasons"]


def test_linked_discussion_answer_is_appended_when_frozen(tmp_path: Path) -> None:
    first = _question("1", "https://github.com/acme/repo/discussions/1")
    second = _question("2", "https://github.com/acme/repo/discussions/2")
    pages = {
        "1": _page(
            "Question one",
            '<p>Start here.</p><a href="https://github.com/acme/repo/discussions/2">follow-up</a>',
        ),
        "2": _page("Question two", "The linked accepted answer."),
    }
    dataset_dir, html_dir = _dataset(tmp_path, [first, second], pages)
    rows, _ = validate_datasets({"x": dataset_dir}, {"x": html_dir})
    row = next(value for value in rows if value["question_id"] == "x-1")
    assert row["status"] == "accepted_structurally"
    assert "The linked accepted answer." in row["expanded_reference_answer"]
    assert row["linked_answer_sources"] == [
        "https://github.com/acme/repo/discussions/2"
    ]


def test_unfrozen_linked_issue_rejects_case(tmp_path: Path) -> None:
    question = _question("1", "https://github.com/acme/repo/discussions/1")
    dataset_dir, html_dir = _dataset(
        tmp_path,
        [question],
        {
            "1": _page(
                "Question",
                '<p>See this.</p><a href="https://github.com/acme/repo/issues/99">issue</a>',
            )
        },
    )
    rows, _ = validate_datasets({"x": dataset_dir}, {"x": html_dir})
    assert rows[0]["status"] == "rejected"
    assert any(
        reason.startswith("unresolved_linked_qa:")
        for reason in rows[0]["rejection_reasons"]
    )


def test_non_document_external_link_rejects_but_image_link_does_not(
    tmp_path: Path,
) -> None:
    first = _question("1", "https://github.com/acme/repo/discussions/1")
    second = _question("2", "https://github.com/acme/repo/discussions/2")
    pages = {
        "1": _page(
            "Question",
            '<p>See <a href="https://external.example/tutorial">tutorial</a>.</p>',
        ),
        "2": _page(
            "Question",
            '<p>Result</p><a href="https://cdn.example/result.png">'
            '<img src="https://cdn.example/result.png" alt="result"></a>',
        ),
    }
    dataset_dir, html_dir = _dataset(tmp_path, [first, second], pages)
    rows, _ = validate_datasets({"x": dataset_dir}, {"x": html_dir})
    by_id = {row["question_id"]: row for row in rows}
    assert "external_non_document_link" in by_id["x-1"]["rejection_reasons"]
    assert by_id["x-2"]["status"] == "accepted_structurally"
