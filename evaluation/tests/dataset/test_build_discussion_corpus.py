from pathlib import Path

from dataset.scripts.build_discussion_corpus import (
    CorpusConfig,
    _candidate_link_paths,
    _path_route,
    _is_exact_docs_url,
    build_questions,
    parse_discussion_html,
)


def test_routes_are_dataset_specific() -> None:
    assert _path_route(Path("flex.mdx"), {}, "tailwind") == "/docs/flex"
    assert _path_route(Path("orm/index.mdx"), {"url": "/orm"}, "prisma") == "/docs/orm"
    assert _path_route(Path("guides/database/overview.mdx"), {}, "supabase") == "/docs/guides/database"


def test_exact_hosts_do_not_accept_subdomains() -> None:
    assert _candidate_link_paths(
        "https://tailwindcss.com/docs/flex", "/docs/grid", {"tailwindcss.com"}
    ) == ["/docs/flex"]
    assert _candidate_link_paths(
        "https://play.tailwindcss.com/example", "/docs/grid", {"tailwindcss.com"}
    ) == []
    assert _is_exact_docs_url("https://tailwindcss.com/docs/flex", {"tailwindcss.com"})
    assert not _is_exact_docs_url("https://tailwindcss.com/blog/release", {"tailwindcss.com"})
    assert _candidate_link_paths("../other", "/docs/orm/current", set()) == [
        "/docs/other"
    ]


def test_discussion_jsonld_keeps_question_and_reference_answer() -> None:
    source = """
    <script type="application/ld+json">{
      "@type": "QAPage",
      "mainEntity": {
        "@type": "Question", "name": "How?", "text": "Question body",
        "acceptedAnswer": {"@type": "Answer", "url": "https://github.com/acme/repo/discussions/1#discussioncomment-2", "text": "<p>Use the docs.</p><img src='https://cdn.example/result.png' alt='result'>"}
      }
    }</script>
    """
    parsed = parse_discussion_html(source)
    assert parsed["title"] == "How?"
    assert parsed["question"] == "Question body"
    assert parsed["accepted_answer"] == "Use the docs."
    assert parsed["accepted_answer_url"] == (
        "https://github.com/acme/repo/discussions/1#discussioncomment-2"
    )
    assert parsed["accepted_answer_images"] == [
        {"url": "https://cdn.example/result.png", "alt": "result", "title": ""}
    ]


def test_unresolved_internal_doc_link_is_preserved_for_validation(tmp_path: Path) -> None:
    config = CorpusConfig(
        name="tailwind",
        repo_root=tmp_path,
        content_root=tmp_path,
        docs_hosts=("tailwindcss.com",),
        route_kind="tailwind",
        revision="abc",
    )
    corpus = [
        {
            "doc_id": "/docs/flex",
            "outgoing_ids": [],
        }
    ]
    discussion = {
        "discussion_number": "1",
        "source_url": "https://github.com/acme/repo/discussions/1",
        "accepted_answer_url": "https://github.com/acme/repo/discussions/1#answer-1",
        "title": "How?",
        "question": "Question",
        "accepted_answer": "Answer",
        "docs_host_links": [
            "https://tailwindcss.com/docs/flex",
            "https://tailwindcss.com/docs/removed-page",
        ],
    }
    questions, stats = build_questions(config, corpus, [discussion])
    assert len(questions) == 1
    assert questions[0]["qrel_ids"] == ["/docs/flex"]
    assert questions[0]["docs_urls"] == [
        "https://tailwindcss.com/docs/flex",
        "https://tailwindcss.com/docs/removed-page",
    ]
    assert questions[0]["resolution_kinds"] == {
        "https://tailwindcss.com/docs/flex": "canonical_path"
    }
    assert stats["excluded_unresolved_internal_answers"] == [
        {
            "discussion": "1",
            "urls": ["https://tailwindcss.com/docs/removed-page"],
        }
    ]
