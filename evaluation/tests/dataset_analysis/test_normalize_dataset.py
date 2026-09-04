from __future__ import annotations

import json
from pathlib import Path

from dataset_analysis.normalize_dataset import (
    AnswerGrade,
    ClaimGrade,
    DraftCitation,
    DraftClaim,
    NormalizedAnswerDraft,
    RequirementGrade,
    RUBRIC_CONFIG,
    RUBRIC_SHA256,
    _contains_live_prose_url,
    _enrich_corpus,
    _load_rubric_config,
    _sanitize_normalized_answer,
    _upgrade_normalized_question,
    build_local_resolution_index,
    canonicalize_draft_evidence_ids,
    compute_grade,
    deduplicate_exact_grade_requirements,
    expand_empty_routing_pages,
    map_urls_to_qrels,
    parse_sections,
    passes,
    resolve_urls_to_qrels,
    recover_historical_targets,
    select_sections,
)


def test_production_rubric_is_domain_neutral_and_has_no_case_overrides() -> None:
    serialized = json.dumps(RUBRIC_CONFIG, ensure_ascii=False).casefold()
    for forbidden in (
        "prisma",
        "supabase",
        "tailwind",
        "codeql",
        "dependabot",
        "cargo",
        "one query",
        "environment variable",
        "question_overrides",
        "project_overrides",
        "dataset_overrides",
        "case_overrides",
    ):
        assert forbidden not in serialized
    assert len(RUBRIC_SHA256) == 64


def test_rubric_loader_rejects_per_case_override_field(tmp_path: Path) -> None:
    payload = dict(RUBRIC_CONFIG)
    payload["question_overrides"] = {"example": {"outcome": "solves"}}
    path = tmp_path / "rubric.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        _load_rubric_config(path)
    except ValueError as error:
        assert "forbidden per-case override" in str(error)
    else:
        raise AssertionError("per-case overrides must be rejected")


def test_nested_section_extraction_preserves_subsections() -> None:
    document = {
        "doc_id": "prisma::relations",
        "title": "Relations",
        "rendered_text": (
            "Intro\n\n## Filter parent records by relation data\n"
            "Use relation filters.\n\n### None\n"
            "Use `none` to return parent records with no matching related records.\n\n"
            "## Other section\nOther material."
        ),
    }
    sections = parse_sections(document)
    parent = next(row for row in sections if row.anchor == "filter-parent-records-by-relation-data")
    assert "Use `none`" in parent.text
    assert "Other material" not in parent.text


def test_stale_anchor_maps_to_current_semantic_heading() -> None:
    document = {
        "doc_id": "prisma::relations",
        "title": "Relations",
        "rendered_text": (
            "## Nested writes\nCreate records together.\n\n"
            "## Filter parent records by relation data\n"
            "Use `some`, `every`, and `none`. `none` returns parents with zero matches.\n\n"
            "## Sorting\nSort related rows."
        ),
    }
    selected, mappings = select_sections(
        document,
        ["filter-on-presence-of-related-records"],
        "How do I find parents that have no related records?",
        "Use the relation filter.",
    )
    assert mappings[0]["canonical_anchor"] == "filter-parent-records-by-relation-data"
    assert any("zero matches" in row["text"] for row in selected)


def test_redirect_cache_preserves_fragment_to_qrel(tmp_path: Path) -> None:
    redirects = tmp_path / "discussions/prisma/redirects.json"
    redirects.parent.mkdir(parents=True)
    redirects.write_text(
        '{"https://www.prisma.io/docs/concepts/components/prisma-client/relation-queries#filter-on-presence-of-related-records": "/docs/orm/fundamentals/relations-and-joins"}',
        encoding="utf-8",
    )
    question = {
        "project": "prisma",
        "docs_urls": [
            "https://www.prisma.io/docs/concepts/components/prisma-client/relation-queries#filter-on-presence-of-related-records"
        ],
        "qrel_ids": ["prisma::/docs/orm/fundamentals/relations-and-joins"],
        "qrel_anchors": {},
    }
    corpus = {
        "prisma::/docs/orm/fundamentals/relations-and-joins": {
            "source_doc_id": "/docs/orm/fundamentals/relations-and-joins"
        }
    }
    mapped = map_urls_to_qrels(question, corpus, tmp_path)
    assert mapped["prisma::/docs/orm/fundamentals/relations-and-joins"] == [
        "filter-on-presence-of-related-records"
    ]


def test_single_qrel_is_not_proof_of_url_resolution(tmp_path: Path) -> None:
    question = {
        "project": "github-docs",
        "docs_urls": ["https://docs.github.com/unrelated/historical-page"],
        "qrel_ids": ["github-docs::/expected"],
        "qrel_anchors": {},
        "resolution_kinds": {},
    }
    corpus = {
        "github-docs::/expected": {"source_doc_id": "/expected", "title": "Expected"}
    }
    mapped, records = resolve_urls_to_qrels(question, corpus, tmp_path)
    assert mapped == {"github-docs::/expected": []}
    assert records[0]["local_doc_id"] is None
    assert records[0]["verified"] is False


def test_verified_frontmatter_redirect_preserves_route_intent(tmp_path: Path) -> None:
    url = (
        "https://docs.github.com/pt/code-security/code-scanning/enabling-code-scanning/"
        "configuring-default-setup-for-code-scanning"
    )
    doc_id = "github-docs::/code-security/configure-code-scanning"
    question = {
        "project": "github-docs",
        "docs_urls": [url],
        "qrel_ids": [doc_id],
        "qrel_anchors": {},
        "resolution_kinds": {url: "frontmatter_redirect"},
    }
    corpus = {
        doc_id: {
            "source_doc_id": "/code-security/configure-code-scanning",
            "title": "Configuring default setup for code scanning",
        }
    }
    mapped, records = resolve_urls_to_qrels(question, corpus, tmp_path)
    assert mapped[doc_id] == ["configuring-default-setup-for-code-scanning"]
    assert records[0]["mapping_method"] == "verified_builder_single_qrel"
    assert records[0]["verified"] is True


def test_accepted_answer_cannot_change_selected_section() -> None:
    document = {
        "doc_id": "github-docs::codeql",
        "title": "Code scanning",
        "rendered_text": (
            "## Default setup\nEnable default scanning.\n\n"
            "## Advanced setup\nConfigure a YAML workflow for API uploads."
        ),
    }
    selected_default, _ = select_sections(
        document,
        [],
        "How do I enable default code scanning?",
        "Use Advanced setup and YAML uploads.",
        max_sections=1,
    )
    selected_other, _ = select_sections(
        document,
        [],
        "How do I enable default code scanning?",
        "Completely unrelated accepted answer.",
        max_sections=1,
    )
    assert selected_default[0]["evidence_id"] == selected_other[0]["evidence_id"]
    assert selected_default[0]["heading"] == "Default setup"


def test_historical_topic_recovery_finds_moved_local_page() -> None:
    old_id = "supabase::/docs/guides/auth"
    new_id = "supabase::/docs/guides/auth/redirect-urls"
    corpus = {
        old_id: {
            "doc_id": old_id,
            "project": "supabase",
            "source_doc_id": "/docs/guides/auth",
            "title": "Auth",
            "rendered_text": "## About authentication\nGeneral authentication material.",
        },
        new_id: {
            "doc_id": new_id,
            "project": "supabase",
            "source_doc_id": "/docs/guides/auth/redirect-urls",
            "title": "Redirect URLs",
            "rendered_text": "## Use wildcards in redirect URLs\nAdd wildcard patterns to the allow list.",
        },
    }
    question = {"project": "supabase", "qrel_ids": [old_id]}
    fragments = {old_id: ["redirect-urls-and-wildcards"]}
    records = [
        {
            "local_doc_id": old_id,
            "original_fragment": "redirect-urls-and-wildcards",
            "route_intent": "overview",
            "normalized_route": "/docs/guides/auth/overview",
        }
    ]
    recovered = recover_historical_targets(
        question,
        corpus,
        fragments,
        records,
        build_local_resolution_index(corpus),
    )
    assert recovered == [new_id]
    assert records[0]["semantic_recovery_method"] == "local_heading_topic_recovery"


def test_generic_exact_anchor_does_not_override_relevant_current_page() -> None:
    current_id = "tailwind-css::/docs/colors"
    unrelated_id = "tailwind-css::/docs/font-feature-settings"
    corpus = {
        current_id: {
            "doc_id": current_id,
            "project": "tailwind-css",
            "source_doc_id": "/docs/colors",
            "title": "Colors",
            "rendered_text": "## Referencing in CSS\nUse color CSS variables.",
        },
        unrelated_id: {
            "doc_id": unrelated_id,
            "project": "tailwind-css",
            "source_doc_id": "/docs/font-feature-settings",
            "title": "Font feature settings",
            "rendered_text": "## Using CSS variables\nConfigure a font feature variable.",
        },
    }
    question = {"project": "tailwind-css", "qrel_ids": [current_id]}
    fragments = {current_id: ["using-css-variables"]}
    records = [
        {
            "local_doc_id": current_id,
            "original_fragment": "using-css-variables",
            "route_intent": "customizing-colors",
            "normalized_route": "/docs/customizing-colors",
        }
    ]
    recovered = recover_historical_targets(
        question,
        corpus,
        fragments,
        records,
        build_local_resolution_index(corpus),
    )
    assert recovered == []


def test_contentless_index_expands_to_question_relevant_pinned_child() -> None:
    parent = "github-docs::/get-started"
    create = "github-docs::/get-started/creating-a-repository"
    format_doc = "github-docs::/get-started/writing-and-formatting"
    corpus = {
        parent: {
            "doc_id": parent,
            "project": "github-docs",
            "source_doc_id": "/get-started",
            "title": "Get started",
            "rendered_text": "",
        },
        create: {
            "doc_id": create,
            "project": "github-docs",
            "source_doc_id": "/get-started/creating-a-repository",
            "title": "Creating a repository",
            "rendered_text": "# Creating a repository\nChoose New repository.",
        },
        format_doc: {
            "doc_id": format_doc,
            "project": "github-docs",
            "source_doc_id": "/get-started/writing-and-formatting",
            "title": "Writing and formatting",
            "rendered_text": "# Writing and formatting\nFormat Markdown text.",
        },
    }
    fragments = {parent: []}
    expanded, records = expand_empty_routing_pages(
        {"project": "github-docs", "query": "How do I create a repository?"},
        [parent],
        fragments,
        build_local_resolution_index(corpus),
        max_children=1,
    )
    assert expanded == [create]
    assert records[0]["routing_doc_id"] == parent
    assert records[0]["method"] == "question_only_pinned_descendant_ranking"


def test_contentful_qrel_is_preserved_beside_empty_routing_page() -> None:
    content = "github-docs::/webhooks/about-webhooks"
    empty = "github-docs::/webhooks"
    corpus = {
        empty: {
            "doc_id": empty,
            "project": "github-docs",
            "source_doc_id": "/webhooks",
            "title": "Webhooks",
            "rendered_text": "",
        },
        content: {
            "doc_id": content,
            "project": "github-docs",
            "source_doc_id": "/webhooks/about-webhooks",
            "title": "About webhooks",
            "rendered_text": "# About webhooks\nWebhooks deliver events.",
        },
    }
    expanded, _ = expand_empty_routing_pages(
        {"project": "github-docs", "query": "Which events do webhooks deliver?"},
        [content, empty],
        {content: [], empty: []},
        build_local_resolution_index(corpus),
        max_children=1,
    )
    assert expanded == [content]


def test_conservative_score_requires_full_verifiable_coverage() -> None:
    draft = NormalizedAnswerDraft(
        answer_text="Use the `none` relation filter.",
        evidence_mode="compositional",
        claims=[DraftClaim(claim_id="C1", claim="Use none.", evidence_ids=["doc#filter"])],
        citations=[DraftCitation(evidence_id="doc#filter", reason="Documents none semantics")],
        image_evidence_used=[],
        insufficient_evidence=False,
        missing_evidence=[],
    ).model_dump()
    grade = AnswerGrade(
        outcome="solves",
        requirements=[
            RequirementGrade(
                requirement_id="R1",
                requirement="Find parents with zero related records",
                critical=True,
                coverage="full",
                evidence_ids=["doc#filter"],
                explanation="The answer identifies the required operator.",
            )
        ],
        claims=[
            ClaimGrade(
                claim_id="C1",
                support="full",
                evidence_ids=["doc#filter"],
                explanation="Directly supported.",
            )
        ],
        correctness=4,
        actionability=4,
        directness=4,
        citation_integrity=True,
        unsupported_or_invented_information=[],
        repair_instructions=[],
        explanation="Complete.",
        confidence="high",
    )
    result = compute_grade(grade, draft, {"doc#filter"})
    assert result["score"] == 1.0
    assert passes({**grade.model_dump(), **result}, 0.9)


def test_grader_cannot_replace_question_derived_requirements() -> None:
    draft = {
        "claims": [{"claim_id": "C1", "claim": "x", "evidence_ids": ["doc"]}],
        "citations": [{"evidence_id": "doc", "reason": "x"}],
    }
    grade = AnswerGrade(
        outcome="solves",
        requirements=[
            RequirementGrade(
                requirement_id="R1",
                requirement="Configure Advanced setup",
                critical=True,
                coverage="full",
                evidence_ids=["doc"],
                explanation="x",
            )
        ],
        claims=[
            ClaimGrade(
                claim_id="C1",
                support="full",
                evidence_ids=["doc"],
                explanation="x",
            )
        ],
        correctness=4,
        actionability=4,
        directness=4,
        citation_integrity=True,
        unsupported_or_invented_information=[],
        repair_instructions=[],
        explanation="x",
        confidence="high",
    )
    result = compute_grade(
        grade,
        draft,
        {"doc"},
        [
            {
                "requirement_id": "R1",
                "requirement": "Enable code scanning while uploading results",
                "critical": True,
            }
        ],
    )
    assert result["requirements_match"] is False
    assert not passes({**grade.model_dump(), **result}, 0.9)


def test_high_score_can_tolerate_bounded_partial_claim_support() -> None:
    draft = {
        "claims": [
            {"claim_id": f"C{index}", "claim": "x", "evidence_ids": ["doc"]}
            for index in range(1, 5)
        ],
        "citations": [{"evidence_id": "doc", "reason": "x"}],
    }
    grade = AnswerGrade(
        outcome="solves",
        requirements=[
            RequirementGrade(
                requirement_id="R1",
                requirement="Solve the request",
                critical=True,
                coverage="full",
                evidence_ids=["doc"],
                explanation="full",
            )
        ],
        claims=[
            ClaimGrade(
                claim_id=f"C{index}",
                support="partial" if index == 4 else "full",
                evidence_ids=["doc"],
                explanation="bounded qualification" if index == 4 else "full",
            )
            for index in range(1, 5)
        ],
        correctness=4,
        actionability=4,
        directness=4,
        citation_integrity=True,
        unsupported_or_invented_information=[],
        repair_instructions=[],
        explanation="complete",
        confidence="high",
    )
    result = compute_grade(grade, draft, {"doc"})
    assert result["score"] == 0.975
    assert result["claims_fully_supported"] is False
    assert result["claims_safely_supported"] is True
    assert passes({**grade.model_dump(), **result}, 0.9)


def test_weighted_score_includes_directness_dimension() -> None:
    draft = {
        "claims": [{"claim_id": "C1", "claim": "x", "evidence_ids": ["doc"]}],
        "citations": [{"evidence_id": "doc", "reason": "x"}],
    }
    grade = AnswerGrade(
        outcome="solves",
        requirements=[
            RequirementGrade(
                requirement_id="R1",
                requirement="Solve the request",
                critical=True,
                coverage="full",
                evidence_ids=["doc"],
                explanation="full",
            )
        ],
        claims=[
            ClaimGrade(
                claim_id="C1",
                support="partial",
                evidence_ids=["doc"],
                explanation="bounded partial support",
            )
        ],
        correctness=3,
        actionability=4,
        directness=4,
        citation_integrity=True,
        unsupported_or_invented_information=[],
        repair_instructions=[],
        explanation="direct and actionable",
        confidence="high",
    )
    result = compute_grade(grade, draft, {"doc"})
    assert result["score"] == 0.8375


def test_unsupported_claim_remains_a_hard_veto() -> None:
    draft = {
        "claims": [{"claim_id": "C1", "claim": "x", "evidence_ids": ["doc"]}],
        "citations": [{"evidence_id": "doc", "reason": "x"}],
    }
    grade = AnswerGrade(
        outcome="solves",
        requirements=[
            RequirementGrade(
                requirement_id="R1",
                requirement="Solve the request",
                critical=True,
                coverage="full",
                evidence_ids=["doc"],
                explanation="full",
            )
        ],
        claims=[
            ClaimGrade(
                claim_id="C1",
                support="unsupported",
                evidence_ids=["doc"],
                explanation="unsupported",
            )
        ],
        correctness=4,
        actionability=4,
        directness=4,
        citation_integrity=True,
        unsupported_or_invented_information=["x"],
        repair_instructions=[],
        explanation="bad claim",
        confidence="high",
    )
    result = compute_grade(grade, draft, {"doc"})
    assert result["claims_safely_supported"] is False
    assert not passes({**grade.model_dump(), **result}, 0.9)


def test_invalid_citation_cannot_pass_even_with_high_model_scores() -> None:
    draft = {
        "claims": [{"claim_id": "C1", "claim": "x", "evidence_ids": ["invented"]}],
        "citations": [{"evidence_id": "invented", "reason": "x"}],
    }
    grade = AnswerGrade(
        outcome="solves",
        requirements=[
            RequirementGrade(
                requirement_id="R1",
                requirement="x",
                critical=True,
                coverage="full",
                evidence_ids=["invented"],
                explanation="x",
            )
        ],
        claims=[
            ClaimGrade(
                claim_id="C1",
                support="full",
                evidence_ids=["invented"],
                explanation="x",
            )
        ],
        correctness=4,
        actionability=4,
        directness=4,
        citation_integrity=True,
        unsupported_or_invented_information=[],
        repair_instructions=[],
        explanation="x",
        confidence="high",
    )
    result = compute_grade(grade, draft, {"real"})
    assert result["provenance_valid"] is False
    assert not passes({**grade.model_dump(), **result}, 0.9)


def test_unique_near_route_citation_is_canonicalized_but_unknown_anchor_is_not() -> None:
    supplied = {
        "github-docs::/codespaces/using-github-codespaces-in-visual-studio-code#opening-a-codespace",
        "github-docs::/codespaces/using-github-codespaces-in-visual-studio-code#deleting-a-codespace",
        "accepted_answer",
    }
    draft = {
        "claims": [
            {
                "claim_id": "C1",
                "claim": "Open the codespace.",
                "evidence_ids": [
                    "github-docs::/codespaces/using-github-codespaces-in-vs-code#opening-a-codespace"
                ],
            },
            {
                "claim_id": "C2",
                "claim": "Invented.",
                "evidence_ids": ["github-docs::/codespaces/unknown#invented-heading"],
            },
        ],
        "citations": [
            {
                "evidence_id": "github-docs::/codespaces/using-github-codespaces-in-vs-code#opening-a-codespace",
                "reason": "same heading and near route",
            }
        ],
    }
    corrections = canonicalize_draft_evidence_ids(draft, supplied)
    assert corrections == [
        {
            "original_evidence_id": "github-docs::/codespaces/using-github-codespaces-in-vs-code#opening-a-codespace",
            "canonical_evidence_id": "github-docs::/codespaces/using-github-codespaces-in-visual-studio-code#opening-a-codespace",
            "method": "exact_anchor_unique_near_route",
        }
    ]
    assert draft["claims"][0]["evidence_ids"][0] in supplied
    assert draft["claims"][1]["evidence_ids"] == [
        "github-docs::/codespaces/unknown#invented-heading"
    ]


def test_unique_near_anchor_on_exact_document_is_canonicalized() -> None:
    valid = {
        "github-docs::/ssh/adding-a-new-ssh-key#adding-a-new-ssh-key-to-your-account",
        "github-docs::/ssh/adding-a-new-ssh-key#prerequisites",
    }
    draft = {
        "claims": [
            {
                "claim_id": "C1",
                "claim": "Add the key.",
                "evidence_ids": [
                    "github-docs::/ssh/adding-a-new-ssh-key#adding-a-new-ssh-key-to-your-github-account"
                ],
            }
        ],
        "citations": [],
    }
    corrections = canonicalize_draft_evidence_ids(draft, valid)
    assert corrections[0]["canonical_evidence_id"].endswith(
        "#adding-a-new-ssh-key-to-your-account"
    )
    assert corrections[0]["method"] == "exact_document_unique_near_anchor"


def test_exact_duplicate_grader_requirement_is_removed_but_changed_one_is_kept() -> None:
    common = dict(
        requirement_id="R1",
        requirement="Explain the timeout and fix it.",
        critical=True,
        coverage="full",
        evidence_ids=["accepted_answer"],
        explanation="Supported.",
    )
    grade = AnswerGrade(
        outcome="solves",
        requirements=[
            RequirementGrade(**common),
            RequirementGrade(**common),
            RequirementGrade(**{**common, "requirement": "A changed requirement."}),
        ],
        claims=[
            ClaimGrade(
                claim_id="C1",
                support="full",
                evidence_ids=["accepted_answer"],
                explanation="Supported.",
            )
        ],
        correctness=4,
        actionability=4,
        directness=4,
        citation_integrity=True,
        unsupported_or_invented_information=[],
        repair_instructions=[],
        explanation="Supported.",
        confidence="high",
    )
    assert deduplicate_exact_grade_requirements(grade) == 1
    assert [row.requirement for row in grade.requirements] == [
        "Explain the timeout and fix it.",
        "A changed requirement.",
    ]


def test_output_upgrade_appends_used_image_text_and_keeps_all_artifacts() -> None:
    row = {
        "normalized_question": {
            "normalized_answer": "Click Revoke.",
            "local_document_citations": [
                {"local_path": "docs/project/page.md", "headings": ["Revoke access"]}
            ],
            "image_text_evidence": [
                {
                    "evidence_id": "img-1",
                    "text": "A Revoke button is visible at https://example.test/revoke.",
                }
            ],
            "normalized_claims": [
                {
                    "claim_id": "claim-1",
                    "claim": "A Revoke button is visible.",
                    "evidence_ids": ["img-1"],
                }
            ],
        },
        "evidence_package": {
            "images": [
                {
                    "evidence_id": "img-1",
                    "text": "A Revoke button is visible at https://example.test/revoke.",
                },
                {"evidence_id": "img-2", "text": "A settings tab is visible."},
            ]
        },
    }
    _upgrade_normalized_question(row)
    question = row["normalized_question"]
    assert len(question["image_text_evidence"]) == 2
    assert question["image_text_evidence_used"] == ["img-1"]
    assert "A Revoke button is visible" in question["normalized_answer_with_local_sources"]
    assert "`https://example.test/revoke`" in question["normalized_answer_with_local_sources"]
    assert not _contains_live_prose_url(question["normalized_answer_with_local_sources"])
    assert "docs/project/page.md" in question["normalized_answer_with_local_sources"]
    assert question["retrieval_modalities"] == ["text", "image_derived_text"]
    _upgrade_normalized_question(row)
    assert row["normalized_question"]["image_text_evidence_used"] == ["img-1"]


def test_corpus_enrichment_accepts_pinned_alt_text_without_pixel_hash() -> None:
    doc_id = "docs::/page"
    corpus = [
        {
            "doc_id": doc_id,
            "source_path": "project/page.md",
            "rendered_text": "# Page\nBody.",
        }
    ]
    accepted = [
        {
            "normalized_question": {
                "qrel_ids": [doc_id],
                "image_text_evidence": [
                    {
                        "evidence_id": "img-alt-1",
                        "kind": "pinned_document_alt_text",
                        "doc_id": doc_id,
                        "text": "Diagram showing the supported flow.",
                        "pixel_verified": False,
                    }
                ],
            }
        }
    ]
    enriched = _enrich_corpus(corpus, accepted)
    assert "[Image img-alt-1]" in enriched[0]["rendered_text"]
    assert enriched[0]["image_texts"][0]["pixel_verified"] is False


def test_url_cleanup_preserves_technical_literals_but_removes_live_links() -> None:
    answer = """Use [the local guide](https://docs.example.test/guide).
Open https://support.example.test/ if needed.
Donate at www.example.test/donate.
Keep `https://user.example.test/` as a literal.

```yaml
url: https://registry.example.test/simple
```
"""
    sanitized = _sanitize_normalized_answer(answer)
    assert "[the local guide]" not in sanitized
    assert "the local guide" in sanitized
    assert "`https://support.example.test/`" in sanitized
    assert "`www.example.test/donate`" in sanitized
    assert "`https://user.example.test/`" in sanitized
    assert "url: https://registry.example.test/simple" in sanitized
    assert not _contains_live_prose_url(sanitized)


def test_live_url_detector_ignores_code_but_rejects_prose() -> None:
    assert _contains_live_prose_url("Open https://example.test now.")
    assert _contains_live_prose_url("Read [this](https://example.test).")
    assert _contains_live_prose_url("Open www.example.test now.")
    assert not _contains_live_prose_url("Use `https://example.test` as the value.")
    assert not _contains_live_prose_url("Use `www.example.test` as the value.")
    assert not _contains_live_prose_url("```text\nGET https://api.example.test/v1\n```")
