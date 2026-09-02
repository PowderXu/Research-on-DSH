"""Local, arm-owned storage paths for the DSH DocsQA plugins."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ARMS = ("fs", "hybrid", "neo4j")
PLUGIN_PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "plugin"
DEFAULT_DATA_ROOT = PLUGIN_PACKAGE_ROOT / "data"


@dataclass(frozen=True)
class ArmDataLayout:
    """Paths owned by one plugin arm; generated contents are never committed."""

    arm: str
    root: Path
    documents: Path
    corpus: Path
    indexes: Path
    assets: Path
    artifacts: Path
    traces: Path
    neo4j: Path

    def ensure(self) -> "ArmDataLayout":
        for path in (
            self.root,
            self.documents,
            self.corpus,
            self.indexes,
            self.assets,
            self.artifacts,
            self.traces,
        ):
            path.mkdir(parents=True, exist_ok=True)
        if self.arm == "neo4j":
            self.neo4j.mkdir(parents=True, exist_ok=True)
        return self


def arm_data_layout(arm: str, data_root: Path | None = None) -> ArmDataLayout:
    normalized = str(arm).strip().casefold()
    if normalized not in ARMS:
        raise ValueError(f"unknown DocsQA plugin arm: {arm}")
    base = (data_root or DEFAULT_DATA_ROOT).expanduser().resolve() / normalized
    return ArmDataLayout(
        arm=normalized,
        root=base,
        documents=base / "documents",
        corpus=base / "corpus",
        indexes=base / "indexes",
        assets=base / "assets",
        artifacts=base / "artifacts",
        traces=base / "traces",
        neo4j=base / "database",
    )
