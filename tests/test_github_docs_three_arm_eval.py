from __future__ import annotations

from kbbench.github_docs_three_arm_eval import (
    _parse_rg_matches,
    extract_code_entities,
    lucene_query_text,
    select_fs_terms,
)


def test_fs_planner_prefers_exact_identifiers_and_is_bounded() -> None:
    terms = select_fs_terms(
        "How can I use `--ignore-revs-file` when automated formatting changes blame?",
        max_terms=3,
    )
    assert terms[0] == "--ignore-revs-file"
    assert len(terms) == 3


def test_parse_rg_matches_keeps_dsh_visible_cap() -> None:
    rows = "\n".join(
        [
            '{"type":"begin","data":{}}',
            '{"type":"match","data":{"path":{"text":"content/a.md"},"lines":{"text":"first\\n"},"line_number":2}}',
            '{"type":"match","data":{"path":{"text":"content/b.md"},"lines":{"text":"second\\n"},"line_number":4}}',
        ]
    )
    matches = _parse_rg_matches(rows, limit=1)
    assert matches == [{"path": "content/a.md", "line": "first", "line_number": 2}]


def test_code_entity_extraction_is_exact_and_code_shaped() -> None:
    entities = extract_code_entities(
        "Set `ACTIONS_STEP_DEBUG`, call `git blame`, and pass --ignore-revs-file. "
        "Do not turn `ordinary` into an entity."
    )
    assert "actions_step_debug" in entities
    assert "--ignore-revs-file" in entities
    assert "ordinary" not in entities


def test_lucene_query_removes_query_syntax() -> None:
    assert lucene_query_text("[Actions] secrets + permissions?") == "actions secrets permissions"


def test_lucene_query_is_bounded_for_long_issue_bodies() -> None:
    query = " ".join(f"term{index}" for index in range(2_000))
    assert len(lucene_query_text(query).split()) == 64
