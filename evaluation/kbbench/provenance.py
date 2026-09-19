"""Stable input identities and redacted execution metadata for saved runs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
from pathlib import Path
from typing import Any, Iterable, Sequence


RETRIEVAL_SOURCE_PATHS = (
    "evaluation/kbbench/backends.py",
    "evaluation/kbbench/provenance.py",
    "evaluation/kbbench/retrieval.py",
    "evaluation/dataset/scripts/records.py",
    "evaluation/kbbench/indexes.py",
    "evaluation/kbbench/scoring.py",
    "dsh_plugin/backend/graph_contract.py",
    "dsh_plugin/backend/graph_records.py",
    "dsh_plugin/backend/retrieval_policy.py",
    "dsh_plugin/backend/semantic_store.py",
)


def canonical_sha256(value: Any) -> str:
    """Hash JSON-like data with a stable encoding and key order."""

    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def file_identity(path: Path, logical_path: str) -> dict[str, Any]:
    """Describe an input by logical name and bytes, not a host-absolute path."""

    resolved = path.resolve()
    return {
        "path": logical_path,
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "bytes": resolved.stat().st_size,
    }


def file_bundle_identity(
    paths: Sequence[tuple[Path, str]],
) -> dict[str, Any]:
    """Fingerprint a small source/dependency bundle under stable logical paths."""

    existing: list[dict[str, Any]] = []
    missing: list[str] = []
    for path, logical_path in sorted(paths, key=lambda item: item[1]):
        if path.is_file():
            existing.append(file_identity(path, logical_path))
        else:
            missing.append(logical_path)
    return {
        "sha256": canonical_sha256({"files": existing, "missing": missing}),
        "files": existing,
        "missing": missing,
    }


def markdown_tree_identity(root: Path) -> dict[str, Any]:
    """Fingerprint the exact Markdown bytes visible to the filesystem arm."""

    resolved_root = root.resolve()
    files = sorted(
        (
            path
            for path in resolved_root.rglob("*")
            if path.is_file() and path.suffix.casefold() in {".md", ".mdx"}
        ),
        key=lambda path: path.relative_to(resolved_root).as_posix(),
    )
    digest = hashlib.sha256()
    byte_count = 0
    for path in files:
        logical_path = path.relative_to(resolved_root).as_posix()
        content = path.read_bytes()
        digest.update(logical_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
        byte_count += len(content)
    return {
        "path": "filesystem_documents",
        "sha256": digest.hexdigest(),
        "files": len(files),
        "bytes": byte_count,
        "extensions": [".md", ".mdx"],
    }


def installed_dependency_versions(names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def sanitize_connection_uri(value: str) -> str:
    """Remove URI userinfo while preserving the endpoint needed for latency context."""

    return re.sub(r"(?<=://)[^/@]+@", "<redacted>@", value, count=1)


def sanitize_argv(argv: Sequence[str]) -> list[str]:
    """Retain the executed CLI while ensuring Neo4j credentials never enter reports."""

    sanitized: list[str] = []
    redact_next = False
    sanitize_uri_next = False
    for value in argv:
        if redact_next:
            sanitized.append("<redacted>")
            redact_next = False
            continue
        if sanitize_uri_next:
            sanitized.append(sanitize_connection_uri(value))
            sanitize_uri_next = False
            continue
        if value == "--neo4j-password":
            sanitized.append(value)
            redact_next = True
            continue
        if value.startswith("--neo4j-password="):
            sanitized.append("--neo4j-password=<redacted>")
            continue
        if value == "--neo4j-uri":
            sanitized.append(value)
            sanitize_uri_next = True
            continue
        if value.startswith("--neo4j-uri="):
            sanitized.append(
                "--neo4j-uri="
                + sanitize_connection_uri(value.split("=", 1)[1])
            )
            continue
        sanitized.append(value)
    return sanitized


def _embedding_model_identity(model: Any, configured_name: str) -> dict[str, Any]:
    """Record the configured model and the resolved Hugging Face revision when exposed."""

    resolved_revision = ""
    resolved_architecture = ""
    try:
        transformer = model[0]
        auto_model = getattr(transformer, "auto_model", None)
        config = getattr(auto_model, "config", None)
        resolved_revision = str(getattr(config, "_commit_hash", "") or "")
        architectures = list(getattr(config, "architectures", None) or [])
        resolved_architecture = ",".join(map(str, architectures))
    except (IndexError, KeyError, TypeError):
        pass
    actual_device = str(getattr(model, "device", "") or "")
    return {
        "configured_name": configured_name,
        "resolved_revision": resolved_revision,
        "resolved_architecture": resolved_architecture,
        "actual_device": actual_device,
    }


def _hardware_and_thread_policy() -> dict[str, Any]:
    thread_environment = {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    torch_threads: dict[str, int] | None = None
    try:
        import torch

        torch_threads = {
            "intraop": int(torch.get_num_threads()),
            "interop": int(torch.get_num_interop_threads()),
        }
    except (ImportError, RuntimeError):
        pass
    return {
        "hardware": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "threads": {
            "evaluation_query_execution": "serial",
            "arm_execution": "serial",
            "hnsw_build_threads": 1,
            "hnsw_query_threads": 1,
            "embedding_build_batch_size": 64,
            "torch": torch_threads,
            "environment": thread_environment,
        },
    }
