from __future__ import annotations

import json
from pathlib import Path

from kbbench.github_docs_benchmark import (
    GitHubDocsCorpus,
    html_to_text,
    normalize_docs_path,
    parse_discussion,
)


def _write_fixture_repo(root: Path) -> None:
    content = root / "content" / "actions" / "guides"
    content.mkdir(parents=True)
    reusables = root / "data" / "reusables" / "actions"
    reusables.mkdir(parents=True)
    variables = root / "data" / "variables"
    variables.mkdir(parents=True)
    (reusables / "limits.md").write_text("The timeout is 10 minutes.\n", encoding="utf-8")
    (variables / "product.yml").write_text("github: GitHub\n", encoding="utf-8")
    (content / "runner.md").write_text(
        """---
title: Runner limits
redirect_from:
  - /actions/old-runner
versions:
  fpt: '*'
---
## Limits
{% data reusables.actions.limits %}

See [tokens](/authentication/tokens).
""",
        encoding="utf-8",
    )
    auth = root / "content" / "authentication"
    auth.mkdir(parents=True)
    (auth / "tokens.md").write_text("---\ntitle: Tokens\n---\n## Use tokens\n", encoding="utf-8")


def test_normalize_docs_path_strips_locale_and_version() -> None:
    assert normalize_docs_path(
        "https://docs.github.com/en/enterprise-cloud@latest/actions/guides/runner#Limits"
    ) == ("/actions/guides/runner", "limits")


def test_corpus_resolves_redirects_and_renders_reusables(tmp_path: Path) -> None:
    _write_fixture_repo(tmp_path)
    corpus = GitHubDocsCorpus(tmp_path)
    doc_id, anchor, kind = corpus.resolve("https://docs.github.com/en/actions/old-runner#limits")
    assert (doc_id, anchor, kind) == (
        "/actions/guides/runner",
        "limits",
        "frontmatter_redirect",
    )
    page = corpus.by_id["/actions/guides/runner"]
    assert "timeout is 10 minutes" in page.rendered_text
    assert page.outgoing_ids == ["/authentication/tokens"]
    assert page.link_edges == [
        {
            "target_id": "/authentication/tokens",
            "target_anchor": "",
            "anchor_text": "tokens",
            "source_section": "limits",
            "context": "See [tokens](/authentication/tokens).",
        }
    ]


def test_parse_discussion_uses_accepted_answer_and_removes_url_leakage(tmp_path: Path) -> None:
    _write_fixture_repo(tmp_path / "repo")
    corpus = GitHubDocsCorpus(tmp_path / "repo")
    question = {
        "@context": "https://schema.org",
        "@type": "QAPage",
        "mainEntity": {
            "@type": "Question",
            "name": "Why does my runner stop?",
            "text": '<p>I read <a href="https://docs.github.com/en/actions/old-runner#limits">https://docs.github.com/en/actions/old-runner#limits</a>.</p>',
            "acceptedAnswer": {
                "@type": "Answer",
                "text": '<p>See <a href="https://docs.github.com/en/actions/old-runner#limits">the limits</a>.</p>',
                "url": "https://github.com/orgs/community/discussions/7#discussioncomment-1",
            },
        },
    }
    discussion = tmp_path / "7.html"
    discussion.write_text(
        '<a href="/orgs/community/discussions/categories/actions">Actions</a>'
        f'<script type="application/ld+json">{json.dumps(question)}</script>',
        encoding="utf-8",
    )
    parsed = parse_discussion(discussion, corpus)
    assert parsed is not None
    assert parsed.qrel_ids == ["/actions/guides/runner"]
    assert parsed.qrel_anchors == {"/actions/guides/runner": ["limits"]}
    assert "https://" not in parsed.query
    assert parsed.evidence_category == "single_page_composed_anchor"


def test_html_to_text_keeps_anchor_label_but_removes_visible_url() -> None:
    assert html_to_text('<p>Read <a href="https://example.com">the guide</a> at https://x.test/a</p>') == "Read the guide at"
