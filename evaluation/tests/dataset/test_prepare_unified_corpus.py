from __future__ import annotations

import json
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from dataset.scripts.prepare_unified_corpus import (
    SourceSpec,
    load_source_specs,
    materialize_unified_corpus,
    prepare_checkout,
)


def _spec(source_id: str, revision: str = "a" * 40) -> SourceSpec:
    return SourceSpec(
        source_id=source_id,
        display_name=source_id.replace("-", " ").title(),
        description=f"Documentation for {source_id}.",
        repository=f"https://github.com/example/{source_id}.git",
        revision=revision,
        checkout_dir=source_id,
        normalizer="fixture",
        docs_hosts=("docs.example.com",),
        content_roots=(PurePosixPath("docs"),),
        support_roots=(PurePosixPath("data/reusables"),),
    )


def _fixture_checkout(root: Path, title: str) -> None:
    (root / "docs" / "guides").mkdir(parents=True)
    (root / "data" / "reusables").mkdir(parents=True)
    (root / "examples").mkdir(parents=True)
    (root / "docs" / "guide.md").write_text(
        f"# {title}\n\nSee [details](guides/details.md) and [the example](../examples/demo.ts).\n",
        encoding="utf-8",
    )
    (root / "docs" / "guides" / "details.md").write_text(
        "# Details\n\nExternal references such as [MDN](https://developer.mozilla.org/) stay links.\n",
        encoding="utf-8",
    )
    (root / "data" / "reusables" / "note.md").write_text("Reusable note.\n", encoding="utf-8")
    (root / "examples" / "demo.ts").write_text("export const demo = true;\n", encoding="utf-8")


def test_materialization_namespaces_projects_and_preserves_linked_files(tmp_path: Path) -> None:
    first = tmp_path / "sources" / "alpha-docs"
    second = tmp_path / "sources" / "beta-docs"
    _fixture_checkout(first, "Alpha guide")
    _fixture_checkout(second, "Beta guide")
    specs = [_spec("alpha-docs"), _spec("beta-docs")]
    output = tmp_path / "unified"

    summary = materialize_unified_corpus(
        specs,
        {"alpha-docs": first, "beta-docs": second},
        output,
        force=False,
    )

    assert summary["canonical_documentation_files"] == 4
    assert (output / "alpha-docs" / "docs" / "guide.md").is_file()
    assert (output / "beta-docs" / "docs" / "guide.md").is_file()
    assert (output / "alpha-docs" / "examples" / "demo.ts").is_file()
    assert "(alpha-docs/index.md)" in (output / "index.md").read_text(encoding="utf-8")

    records = [
        json.loads(line)
        for line in (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_path = {row["dataset_path"]: row for row in records}
    assert by_path["alpha-docs/docs/guide.md"]["file_id"] == "alpha-docs::docs/guide.md"
    assert by_path["beta-docs/docs/guide.md"]["file_id"] == "beta-docs::docs/guide.md"
    assert by_path["alpha-docs/examples/demo.ts"]["kind"] == "linked_support"
    assert by_path["index.md"]["searchable_by_default"] is False
    assert by_path["alpha-docs/index.md"]["kind"] == "routing"
    assert by_path["alpha-docs/data/reusables/note.md"]["kind"] == "support"


def test_noncanonical_markdown_is_copied_but_not_searchable(tmp_path: Path) -> None:
    checkout = tmp_path / "source"
    _fixture_checkout(checkout, "Guide")
    output = tmp_path / "unified"
    materialize_unified_corpus(
        [_spec("alpha-docs")],
        {"alpha-docs": checkout},
        output,
        force=False,
        canonical_paths={
            "alpha-docs": {"docs/guide.md"},
        },
    )
    records = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    details = next(row for row in records if row["dataset_path"] == "alpha-docs/docs/guides/details.md")
    assert details["kind"] == "noncanonical_documentation"
    assert details["searchable_by_default"] is False


def test_existing_output_requires_explicit_force(tmp_path: Path) -> None:
    checkout = tmp_path / "source"
    _fixture_checkout(checkout, "Guide")
    output = tmp_path / "unified"
    output.mkdir()
    with pytest.raises(RuntimeError, match="--force"):
        materialize_unified_corpus(
            [_spec("alpha-docs")], {"alpha-docs": checkout}, output, force=False
        )


def test_public_source_config_is_pinned_and_unique() -> None:
    config = Path(__file__).resolve().parents[2] / "dataset" / "templates" / "public_sources.json"
    specs = load_source_specs(config)
    assert [spec.source_id for spec in specs] == [
        "github-docs",
        "tailwind-css",
        "prisma",
        "supabase",
    ]
    assert all(len(spec.revision) == 40 for spec in specs)


def test_offline_checkout_verifies_exact_clean_revision(tmp_path: Path) -> None:
    checkout = tmp_path / "sources" / "alpha-docs"
    checkout.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=checkout, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=checkout, check=True)
    (checkout / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=checkout, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=checkout, check=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()

    assert prepare_checkout(_spec("alpha-docs", revision), tmp_path / "sources", offline=True) == checkout
    (checkout / "README.md").write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="local changes"):
        prepare_checkout(_spec("alpha-docs", revision), tmp_path / "sources", offline=True)
