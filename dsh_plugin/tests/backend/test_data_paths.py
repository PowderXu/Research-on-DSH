from pathlib import Path
import json

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
