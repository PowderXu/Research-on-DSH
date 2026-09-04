from pathlib import Path
import json
import os
import shutil
import subprocess

import pytest

from dsh_plugin.backend.corpus_workspace import (
    materialize_corpus_workspace,
    validate_corpus_workspace,
)
from dsh_plugin.backend.data_paths import arm_data_layout
from dsh_plugin.backend.prepare_plugin_data import prepare


def test_arm_data_layout_is_owned_by_the_selected_plugin(tmp_path: Path) -> None:
    layout = arm_data_layout("neo4j", tmp_path).ensure()

    assert layout.root == (tmp_path / "neo4j").resolve()
    assert layout.corpus.is_dir()
    assert layout.indexes.is_dir()
    assert layout.artifacts.is_dir()
    assert layout.neo4j.is_dir()


def test_non_graph_arm_does_not_create_a_neo4j_database(tmp_path: Path) -> None:
    layout = arm_data_layout("hybrid", tmp_path).ensure()

    assert not layout.neo4j.exists()


def test_prepare_materializes_the_exact_corpus_text_for_filesystem_search(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    row = {
        "doc_id": "project::/guide",
        "source_path": "project/docs/guide.md",
        "rendered_text": "Guide text\n\n[Image evidence] Settings is highlighted.",
    }
    (dataset / "corpus.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (dataset / "manifest.json").write_text("{}\n", encoding="utf-8")

    result = prepare(
        arm="fs",
        source_dataset=dataset,
        source_documents=None,
        data_root=tmp_path / "plugin-data",
        copy_documents=False,
    )

    materialized = Path(result["documents"]) / "project/docs/guide.md"
    assert materialized.read_text(encoding="utf-8") == row["rendered_text"]
    assert result["materialized_documents"] == 1
    assert result["corpus_workspace"] == validate_corpus_workspace(
        [row], Path(result["documents"])
    )


def _corpus(*paths: str) -> list[dict[str, str]]:
    return [
        {"source_path": path, "rendered_text": f"Unique corpus text {index}.\n"}
        for index, path in enumerate(paths or ("project/docs/guide.md",))
    ]


def test_prepare_does_not_overlay_raw_source_documents(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    corpus = _corpus()
    (dataset / "corpus.jsonl").write_text(json.dumps(corpus[0]) + "\n")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "unselected.md").write_text("This must never be searchable.")

    result = prepare(
        arm="fs", source_dataset=dataset, source_documents=raw,
        data_root=tmp_path / "plugin-data", copy_documents=True,
    )

    documents = Path(result["documents"])
    assert not (documents / "unselected.md").exists()
    assert result["copy_documents"] is False
    assert result["copy_documents_requested"] is True
    assert (Path(result["root"]) / "documents.path").read_text().strip() == str(raw)
    validate_corpus_workspace(corpus, documents)


@pytest.mark.parametrize("source_path", [
    "", ".", "../outside.md", "docs/../outside.md", "/absolute.md",
    "docs//guide.md", "docs/./guide.md", "docs/", "docs/line\nbreak.md",
    ".ignore", ".IGNORE", "docs/.ignore", ".git/config.md",
])
def test_unsafe_manifest_paths_are_rejected_before_writes(
    tmp_path: Path, source_path: str,
) -> None:
    destination = tmp_path / "documents"
    valid = _corpus()
    materialize_corpus_workspace(valid, destination)
    target = destination / valid[0]["source_path"]
    corpus = [
        {**valid[0], "rendered_text": "Must not overwrite valid content."},
        {"source_path": source_path, "rendered_text": "bad"},
    ]

    with pytest.raises(ValueError, match="source_path"):
        materialize_corpus_workspace(corpus, destination)

    assert target.read_text() == valid[0]["rendered_text"]


@pytest.mark.parametrize("corpus", [
    [],
    _corpus("same.md", "same.md"),
    _corpus("Same.md", "same.md"),
    _corpus("Docs/first.md", "docs/second.md"),
    _corpus("docs", "docs/file.md"),
    [{"source_path": "guide.md", "rendered_text": None}],
])
def test_invalid_corpus_is_rejected_without_creating_workspace(
    tmp_path: Path, corpus: list[dict],
) -> None:
    destination = tmp_path / "documents"
    with pytest.raises(ValueError):
        materialize_corpus_workspace(corpus, destination)
    assert not destination.exists()


@pytest.mark.parametrize("extra", ["old.md", "project/docs/extra.mdx", "empty-directory/"])
def test_prepare_refuses_stale_extras_without_deleting_or_overwriting(
    tmp_path: Path, extra: str,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    corpus = _corpus()
    (dataset / "corpus.jsonl").write_text(json.dumps(corpus[0]) + "\n")
    layout = arm_data_layout("fs", tmp_path / "plugin-data").ensure()
    target = layout.documents / corpus[0]["source_path"]
    target.parent.mkdir(parents=True)
    target.write_text("Existing target is not overwritten on rejection.")
    stale = layout.documents / extra
    if extra.endswith("/"):
        stale.mkdir()
    else:
        stale.write_text("Preserve me.")

    with pytest.raises(ValueError, match="extra"):
        prepare(
            arm="fs", source_dataset=dataset, source_documents=None,
            data_root=tmp_path / "plugin-data", copy_documents=False,
        )

    assert target.read_text() == "Existing target is not overwritten on rejection."
    assert stale.exists()
    assert not (layout.corpus / "corpus.jsonl").exists()


@pytest.mark.parametrize("change", ["missing", "mismatch", "extra", "ignore", "no_ignore"])
def test_validation_rejects_workspace_drift(tmp_path: Path, change: str) -> None:
    corpus = _corpus()
    materialize_corpus_workspace(corpus, tmp_path)
    target = tmp_path / corpus[0]["source_path"]
    if change == "missing":
        target.unlink()
    elif change == "mismatch":
        target.write_text("Unrecognized text.")
    elif change == "extra":
        (target.parent / "extra.md").write_text("Extra text.")
    elif change == "ignore":
        (tmp_path / ".ignore").write_text("*\n")
    else:
        (tmp_path / ".ignore").unlink()

    with pytest.raises(ValueError):
        validate_corpus_workspace(corpus, tmp_path)


@pytest.mark.parametrize("symlink_kind", ["root", "ancestor", "directory", "file", "dangling"])
def test_symlinks_are_rejected_without_modifying_targets(
    tmp_path: Path, symlink_kind: str,
) -> None:
    corpus = _corpus()
    external = tmp_path / "external"
    external.mkdir()
    external_target = external / "guide.md"
    external_target.write_text("Outside workspace; do not change.")
    destination = tmp_path / "documents"
    if symlink_kind == "root":
        destination.symlink_to(external, target_is_directory=True)
    elif symlink_kind == "ancestor":
        link = tmp_path / "link"
        link.symlink_to(external, target_is_directory=True)
        destination = link / "documents"
    else:
        (destination / "project").mkdir(parents=True)
        if symlink_kind == "directory":
            (destination / "project/docs").symlink_to(external, target_is_directory=True)
        else:
            (destination / "project/docs").mkdir()
            link_target = external_target if symlink_kind == "file" else external / "missing.md"
            (destination / "project/docs/guide.md").symlink_to(link_target)

    for operation in (materialize_corpus_workspace, validate_corpus_workspace):
        with pytest.raises(ValueError, match="symlink"):
            operation(corpus, destination)
    assert external_target.read_text() == "Outside workspace; do not change."


def test_unmanaged_ignore_file_is_not_overwritten(tmp_path: Path) -> None:
    original = "# User-owned ignore rules\nsecret.md\n"
    (tmp_path / ".ignore").write_text(original)
    with pytest.raises(ValueError, match="unmanaged"):
        materialize_corpus_workspace(_corpus(), tmp_path)
    assert (tmp_path / ".ignore").read_text() == original
    assert not (tmp_path / "project").exists()


def test_snapshot_hashes_are_order_independent_and_content_sensitive(tmp_path: Path) -> None:
    corpus = _corpus("project/a.md", "project/b.mdx")
    first = materialize_corpus_workspace(corpus, tmp_path)
    assert validate_corpus_workspace(list(reversed(corpus)), tmp_path) == first
    changed = [{**corpus[0], "rendered_text": "Changed."}, corpus[1]]
    second = materialize_corpus_workspace(changed, tmp_path)
    assert first["corpus_sha256"] != second["corpus_sha256"]
    assert first["workspace_sha256"] != second["workspace_sha256"]
    assert first["ignore_sha256"] == second["ignore_sha256"]


def _ripgrep() -> str:
    candidate = os.environ.get("DSH_TEST_RIPGREP") or shutil.which("rg")
    if not candidate:
        pytest.skip("ripgrep unavailable; set DSH_TEST_RIPGREP to test official binary")
    return candidate


def test_generated_ignore_exposes_exact_nested_corpus_under_git_ignored_data(
    tmp_path: Path,
) -> None:
    rg = _ripgrep()
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    arm_root = repository / "data/fs"
    arm_root.mkdir(parents=True)
    (arm_root / ".gitignore").write_text("*\n")
    destination = arm_root / "documents"
    corpus = _corpus(
        "project/docs/nested/guide.md", "project/docs/nested/page.mdx",
        "project/[draft] * ? ! #/literal [v1] * ? ! # \\ name.md",
        "project/space /trailing .md ", ".hidden/docs.md",
    )
    materialize_corpus_workspace(corpus, destination)

    for flags in (
        [], ["--glob", "*.md*"], ["--hidden", "--no-ignore", "--glob", "*.md*"],
    ):
        result = subprocess.run(
            [rg, "--files", *flags, "--", "."], cwd=destination,
            text=True, capture_output=True, check=True,
        )
        visible = {line.removeprefix("./") for line in result.stdout.splitlines()}
        assert visible - {".ignore"} == {row["source_path"] for row in corpus}
    result = subprocess.run(
        [rg, "--glob", "*.md*", "--", "Unique corpus text", "."],
        cwd=destination, text=True, capture_output=True, check=True,
    )
    assert len(result.stdout.splitlines()) == len(corpus)


def test_ignore_rules_are_not_treated_as_a_search_boundary(tmp_path: Path) -> None:
    corpus = _corpus("project/docs/a*b?.md")
    materialize_corpus_workspace(corpus, tmp_path)
    stale = tmp_path / "project/docs/axby.md"
    stale.write_text("This extra document must fail validation.")
    # The literal wildcard path must not accidentally unignore other files.
    ordinary = subprocess.run(
        [_ripgrep(), "--files", "--", "."],
        cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    assert "axby.md" not in ordinary.stdout
    # Official grep's positive glob can override .ignore's file exclusions.
    result = subprocess.run(
        [_ripgrep(), "--files", "--glob", "*.md*", "--", "."],
        cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    assert "axby.md" in result.stdout
    with pytest.raises(ValueError, match="extra file"):
        validate_corpus_workspace(corpus, tmp_path)
