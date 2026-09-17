"""Shared runtime manifest and byte identities for execution and report validation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

from kbbench.provenance import RETRIEVAL_SOURCE_PATHS


AGENT_SOURCE_PATHS = (
    "dsh_plugin/agent_eval/runner.py",
    "dsh_plugin/agent_eval/runtime.py",
    "dsh_plugin/backend/corpus_workspace.py",
    "dsh_plugin/backend/prepare_plugin_data.py",
    "dsh_plugin/scripts/preflight_fs.ts",
    "dsh_plugin/backend/http_contract.py",
    "dsh_plugin/backend/service.py",
    "dsh_plugin/plugin/package.json",
    *RETRIEVAL_SOURCE_PATHS,
)


def logical_relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"runtime file is outside logical root: {path}") from error


def files_sha256(paths: Iterable[Path], *, logical_root: Path) -> str:
    """Hash file bytes under stable logical paths, never host-absolute paths."""

    digest = hashlib.sha256()
    resolved = sorted(
        (value.resolve() for value in paths),
        key=lambda path: logical_relative_path(path, logical_root),
    )
    for path in resolved:
        digest.update(logical_relative_path(path, logical_root).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def file_identity(path: Path, logical_path: str) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": logical_path,
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def repo_file_bundle_identity(
    paths: Iterable[Path], project_root: Path
) -> dict[str, object]:
    files = sorted(
        (path.resolve() for path in paths),
        key=lambda path: logical_relative_path(path, project_root),
    )
    return {
        "sha256": files_sha256(files, logical_root=project_root),
        "files": [
            file_identity(path, logical_relative_path(path, project_root))
            for path in files
        ],
    }


def current_runtime_identity(
    project_root: Path,
    arm: str,
    *,
    model_patch: Path | None = None,
    common_patch: Path | None = None,
    arm_patch: Path | None = None,
) -> dict[str, Any]:
    """Use the same source list and hashing in the runner and report validator."""

    compiled = sorted((project_root / "dsh_plugin/plugin/lib").rglob("*.js"))
    if not compiled:
        raise FileNotFoundError(
            "compiled DSH plugin runtime is missing; run npm run build in dsh_plugin/plugin"
        )
    harness = project_root / "dsh_plugin/harness"
    paths = [
        *(project_root / relative for relative in AGENT_SOURCE_PATHS),
        model_patch if model_patch is not None else harness / "model-openai.patch.yml",
        common_patch if common_patch is not None else harness / "docsqa_dsh_common.patch.yml",
        arm_patch if arm_patch is not None else harness / f"docsqa_{arm}_system.patch.yml",
        *compiled,
    ]
    return {
        "runtime": repo_file_bundle_identity(paths, project_root),
        "compiled_plugin": repo_file_bundle_identity(compiled, project_root),
    }
