#!/usr/bin/env python3
"""Clone pinned public documentation repos and build one routed corpus tree.

The output is a filesystem corpus for retrieval and agent evaluation. Accepted
answers and qrels are deliberately not copied into it. Generated ``index.md``
files are routing metadata and are marked non-searchable in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import unquote, urlsplit


MARKDOWN_SUFFIXES = {".md", ".mdx"}
SOURCE_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
MARKDOWN_TARGET_RE = re.compile(
    r"!?\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))", re.MULTILINE
)
HTML_TARGET_RE = re.compile(
    r"(?:href|src|to)\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE
)
MDX_IMPORT_RE = re.compile(
    r"(?:import|export)\s+[^\n;]*?\sfrom\s*[\"']([^\"']+)[\"']",
    re.MULTILINE,
)


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    display_name: str
    description: str
    repository: str
    revision: str
    checkout_dir: str
    normalizer: str
    docs_hosts: tuple[str, ...]
    content_roots: tuple[PurePosixPath, ...]
    support_roots: tuple[PurePosixPath, ...]


def _relative_path(value: str, field: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a non-empty relative path: {value!r}")
    return path


def load_source_specs(config_path: Path) -> list[SourceSpec]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("sources"), list):
        raise ValueError("public source config must use schema_version 1 and contain sources")
    specs: list[SourceSpec] = []
    seen_ids: set[str] = set()
    seen_checkouts: set[str] = set()
    for raw in payload["sources"]:
        source_id = str(raw.get("id") or "")
        checkout_dir = str(raw.get("checkout_dir") or "")
        revision = str(raw.get("revision") or "").casefold()
        if not SOURCE_ID_RE.fullmatch(source_id):
            raise ValueError(f"invalid source id: {source_id!r}")
        if source_id in seen_ids:
            raise ValueError(f"duplicate source id: {source_id}")
        if checkout_dir in seen_checkouts:
            raise ValueError(f"duplicate checkout_dir: {checkout_dir}")
        if not REVISION_RE.fullmatch(revision):
            raise ValueError(f"source {source_id} must use a full 40-character commit")
        repository = str(raw.get("repository") or "")
        if not repository.startswith("https://github.com/") or not repository.endswith(".git"):
            raise ValueError(f"source {source_id} must use a public HTTPS GitHub clone URL")
        normalizer = str(raw.get("normalizer") or "")
        if normalizer not in {"github", "tailwind", "prisma", "supabase"}:
            raise ValueError(f"source {source_id} has unsupported normalizer: {normalizer!r}")
        docs_hosts = tuple(str(value).casefold() for value in raw.get("docs_hosts") or [])
        if not docs_hosts:
            raise ValueError(f"source {source_id} has no documentation hosts")
        content_roots = tuple(
            _relative_path(str(value), f"{source_id}.content_roots")
            for value in raw.get("content_roots") or []
        )
        if not content_roots:
            raise ValueError(f"source {source_id} has no content roots")
        specs.append(
            SourceSpec(
                source_id=source_id,
                display_name=str(raw.get("display_name") or source_id),
                description=str(raw.get("description") or ""),
                repository=repository,
                revision=revision,
                checkout_dir=_relative_path(checkout_dir, f"{source_id}.checkout_dir").as_posix(),
                normalizer=normalizer,
                docs_hosts=docs_hosts,
                content_roots=content_roots,
                support_roots=tuple(
                    _relative_path(str(value), f"{source_id}.support_roots")
                    for value in raw.get("support_roots") or []
                ),
            )
        )
        seen_ids.add(source_id)
        seen_checkouts.add(checkout_dir)
    return specs


def _git(*args: str, cwd: Path | None = None, no_lazy_fetch: bool = False) -> str:
    environment = os.environ.copy()
    if no_lazy_fetch:
        environment["GIT_NO_LAZY_FETCH"] = "1"
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def prepare_checkout(spec: SourceSpec, source_root: Path, *, offline: bool) -> Path:
    checkout = source_root / spec.checkout_dir
    if checkout.exists() and not (checkout / ".git").is_dir():
        raise RuntimeError(f"refusing to replace non-git path: {checkout}")
    if not (checkout / ".git").is_dir():
        if offline:
            raise RuntimeError(f"offline source checkout is missing: {checkout}")
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--filter=blob:none", "--no-checkout", spec.repository, str(checkout))

    dirty = _git("status", "--porcelain", cwd=checkout, no_lazy_fetch=offline)
    if dirty:
        raise RuntimeError(f"source checkout has local changes: {checkout}")
    if not offline:
        _git("fetch", "--depth", "1", "origin", spec.revision, cwd=checkout)
        _git("checkout", "--detach", spec.revision, cwd=checkout)
    actual = _git("rev-parse", "HEAD", cwd=checkout, no_lazy_fetch=offline).casefold()
    if actual != spec.revision:
        mode = "offline checkout" if offline else "checkout"
        raise RuntimeError(
            f"{mode} revision mismatch for {spec.source_id}: expected {spec.revision}, got {actual}"
        )
    return checkout


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination, follow_symlinks=True)


def _link_targets(text: str) -> list[str]:
    targets = [match.group(1) or match.group(2) for match in MARKDOWN_TARGET_RE.finditer(text)]
    targets.extend(match.group(1) for match in HTML_TARGET_RE.finditer(text))
    targets.extend(match.group(1) for match in MDX_IMPORT_RE.finditer(text))
    return [html.unescape(target.strip()) for target in targets if target.strip()]


def _resolve_local_target(raw: str, source_file: Path, repository_root: Path) -> Path | None:
    parsed = urlsplit(raw)
    if parsed.scheme or parsed.netloc or raw.startswith(("#", "/", "mailto:", "javascript:")):
        return None
    target_text = unquote(parsed.path)
    if not target_text or target_text.startswith("{"):
        return None
    base = (source_file.parent / target_text).resolve()
    repository_root = repository_root.resolve()
    if not _inside(base, repository_root):
        return None
    candidates = [base]
    if not base.suffix:
        candidates.extend(
            [
                base.with_suffix(".md"),
                base.with_suffix(".mdx"),
                base / "index.md",
                base / "index.mdx",
                base / "README.md",
            ]
        )
    for candidate in candidates:
        if candidate.is_file() and _inside(candidate.resolve(), repository_root):
            return candidate
    return None


def _copy_declared_roots(
    spec: SourceSpec,
    checkout: Path,
    project_output: Path,
) -> tuple[dict[str, str], list[Path]]:
    roles: dict[str, str] = {}
    documentation_files: list[Path] = []
    declared = [(root, "content") for root in spec.content_roots]
    declared.extend((root, "support") for root in spec.support_roots)
    for relative_root, root_role in declared:
        source_root = checkout / relative_root
        if not source_root.is_dir():
            raise RuntimeError(f"missing {root_role} root for {spec.source_id}: {source_root}")
        for source_file in _iter_files(source_root):
            source_relative = source_file.relative_to(checkout)
            destination = project_output / source_relative
            _copy_file(source_file, destination)
            role = (
                "documentation"
                if root_role == "content" and source_file.suffix.casefold() in MARKDOWN_SUFFIXES
                else "asset"
                if root_role == "content"
                else "support"
            )
            roles[source_relative.as_posix()] = role
            if role == "documentation":
                documentation_files.append(source_file)
    return roles, documentation_files


def canonical_repository_paths(spec: SourceSpec, checkout: Path) -> set[str]:
    """Return the exact repository paths selected by the canonical normalizer."""

    if spec.normalizer == "github":
        from .build import GitHubDocsCorpus

        rows = GitHubDocsCorpus(checkout).pages
        return {
            f"{spec.content_roots[0].as_posix()}/{row.source_path}" for row in rows
        }
    if spec.normalizer in {"tailwind", "prisma", "supabase"}:
        from .build_discussion_corpus import CorpusConfig, build_corpus

        content_root = checkout / spec.content_roots[0]
        config = CorpusConfig(
            name=spec.source_id,
            repo_root=checkout,
            content_root=content_root,
            docs_hosts=spec.docs_hosts,
            route_kind=spec.normalizer,
            revision=spec.revision,
        )
        rows, _ = build_corpus(config)
        return {
            f"{spec.content_roots[0].as_posix()}/{row['source_path']}" for row in rows
        }
    raise ValueError(f"unsupported canonical normalizer for {spec.source_id}: {spec.normalizer!r}")


def _copy_linked_files(
    checkout: Path,
    project_output: Path,
    documentation_files: list[Path],
    roles: dict[str, str],
) -> int:
    queue = list(documentation_files)
    scanned: set[Path] = set()
    copied = 0
    while queue:
        source_file = queue.pop()
        resolved_source = source_file.resolve()
        if resolved_source in scanned:
            continue
        scanned.add(resolved_source)
        text = source_file.read_text(encoding="utf-8", errors="replace")
        for raw_target in _link_targets(text):
            target = _resolve_local_target(raw_target, source_file, checkout)
            if target is None:
                continue
            relative = target.relative_to(checkout).as_posix()
            if relative in roles:
                continue
            _copy_file(target, project_output / relative)
            roles[relative] = "linked_support"
            copied += 1
            if target.suffix.casefold() in MARKDOWN_SUFFIXES:
                queue.append(target)
    return copied


def _project_index(spec: SourceSpec, checkout: Path) -> str:
    lines = [
        f"# {spec.display_name}",
        "",
        spec.description,
        "",
        f"Source: `{spec.repository}`",
        f"Pinned revision: `{spec.revision}`",
        "",
        "This file is routing metadata. It is not an answer-bearing documentation page.",
        "",
        "## Documentation roots",
        "",
    ]
    for root in spec.content_roots:
        count = sum(
            path.suffix.casefold() in MARKDOWN_SUFFIXES
            for path in _iter_files(checkout / root)
        )
        lines.append(f"- [`{root.as_posix()}/`]({root.as_posix()}/) — {count} Markdown/MDX files")
    sections: list[tuple[str, str]] = []
    for root in spec.content_roots:
        for child in sorted((checkout / root).iterdir(), key=lambda path: path.name.casefold()):
            if child.is_dir() and not child.name.startswith((".", "_", "(")):
                relative = child.relative_to(checkout).as_posix() + "/"
                sections.append((child.name.replace("-", " ").title(), relative))
    if sections:
        lines.extend(["", "## Top-level sections", ""])
        for title, relative in sections:
            lines.append(f"- [{title}]({relative})")
    return "\n".join(lines) + "\n"


def _root_index(specs: list[SourceSpec]) -> str:
    lines = [
        "# Unified Public Documentation Corpus",
        "",
        "This corpus combines four independently pinned public documentation repositories.",
        "Use this routing index to choose a project, then consult that project's index.",
        "Routing indexes are metadata and are excluded from answer relevance judgments.",
        "",
        "## Projects",
        "",
    ]
    for spec in specs:
        lines.append(f"- [{spec.display_name}]({spec.source_id}/index.md) — {spec.description}")
    return "\n".join(lines) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_web_url(repository: str, revision: str, source_path: str) -> str:
    base = repository.removesuffix(".git")
    return f"{base}/blob/{revision}/{source_path}"


def materialize_unified_corpus(
    specs: list[SourceSpec],
    checkouts: dict[str, Path],
    output_root: Path,
    *,
    force: bool,
    include_linked_files: bool = True,
    canonical_paths: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    if output_root.exists() and not force:
        raise RuntimeError(f"output already exists; pass --force to replace it: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    completed = False
    try:
        role_maps: dict[str, dict[str, str]] = {}
        source_summaries: list[dict[str, Any]] = []
        for spec in specs:
            checkout = checkouts[spec.source_id]
            project_output = staging / spec.source_id
            project_output.mkdir(parents=True)
            roles, documentation_files = _copy_declared_roots(spec, checkout, project_output)
            selected_paths = (canonical_paths or {}).get(spec.source_id)
            if selected_paths is not None:
                for source_path, role in list(roles.items()):
                    if role == "documentation" and source_path not in selected_paths:
                        roles[source_path] = "noncanonical_documentation"
            linked_count = 0
            if include_linked_files:
                linked_count = _copy_linked_files(
                    checkout, project_output, documentation_files, roles
                )
            (project_output / "index.md").write_text(
                _project_index(spec, checkout), encoding="utf-8"
            )
            role_maps[spec.source_id] = roles
            source_summaries.append(
                {
                    "id": spec.source_id,
                    "display_name": spec.display_name,
                    "description": spec.description,
                    "repository": spec.repository,
                    "revision": spec.revision,
                    "checkout_dir": spec.checkout_dir,
                    "content_roots": [root.as_posix() for root in spec.content_roots],
                    "support_roots": [root.as_posix() for root in spec.support_roots],
                    "documentation_source_files": sum(
                        role in {"documentation", "noncanonical_documentation"}
                        for role in roles.values()
                    ),
                    "canonical_documentation_files": sum(
                        role == "documentation" for role in roles.values()
                    ),
                    "linked_support_files": linked_count,
                }
            )
        (staging / "index.md").write_text(_root_index(specs), encoding="utf-8")
        (staging / ".gitignore").write_text("*\n!.gitignore\n", encoding="utf-8")

        records: list[dict[str, Any]] = [
            {
                "file_id": "routing::root",
                "project": None,
                "kind": "routing",
                "searchable_by_default": False,
                "dataset_path": "index.md",
                "source_path": None,
                "source_url": None,
                "source_repository": None,
                "source_revision": None,
                "sha256": _sha256(staging / "index.md"),
                "bytes": (staging / "index.md").stat().st_size,
            }
        ]
        for spec in specs:
            project_index = staging / spec.source_id / "index.md"
            records.append(
                {
                    "file_id": f"{spec.source_id}::routing",
                    "project": spec.source_id,
                    "kind": "routing",
                    "searchable_by_default": False,
                    "dataset_path": f"{spec.source_id}/index.md",
                    "source_path": None,
                    "source_url": None,
                    "source_repository": spec.repository,
                    "source_revision": spec.revision,
                    "sha256": _sha256(project_index),
                    "bytes": project_index.stat().st_size,
                }
            )
            for source_path, role in sorted(role_maps[spec.source_id].items()):
                dataset_path = f"{spec.source_id}/{source_path}"
                path = staging / dataset_path
                records.append(
                    {
                        "file_id": f"{spec.source_id}::{source_path}",
                        "project": spec.source_id,
                        "kind": role,
                        "searchable_by_default": role == "documentation",
                        "dataset_path": dataset_path,
                        "source_path": source_path,
                        "source_url": _source_web_url(
                            spec.repository, spec.revision, source_path
                        ),
                        "source_repository": spec.repository,
                        "source_revision": spec.revision,
                        "sha256": _sha256(path),
                        "bytes": path.stat().st_size,
                    }
                )
        records.sort(key=lambda row: str(row["dataset_path"]))
        with (staging / "manifest.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        summary = {
            "schema_version": 1,
            "name": "docsqa-unified-public-docs",
            "description": "Pinned public documentation corpus; no private user repository is included.",
            "source_file_id_format": "<project>::<original-repository-relative-path>",
            "routing_indexes_searchable_by_default": False,
            "files": len(records),
            "canonical_documentation_files": sum(
                record["kind"] == "documentation" for record in records
            ),
            "noncanonical_documentation_files": sum(
                record["kind"] == "noncanonical_documentation" for record in records
            ),
            "sources": source_summaries,
        }
        (staging / "manifest.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if output_root.exists():
            shutil.rmtree(output_root)
        staging.replace(output_root)
        completed = True
        return summary
    finally:
        if not completed and staging.exists():
            shutil.rmtree(staging)


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root / "evaluation/dataset/templates/public_sources.json",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=project_root / "results/cache/docsqa-source-repos",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "evaluation/dataset/docs",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use existing clean checkouts and fail instead of fetching or cloning.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing generated output tree after a complete staging build.",
    )
    parser.add_argument(
        "--no-linked-files",
        action="store_true",
        help="Do not copy local code/assets directly referenced from Markdown/MDX files.",
    )
    args = parser.parse_args()

    specs = load_source_specs(args.config)
    checkouts = {
        spec.source_id: prepare_checkout(spec, args.source_root, offline=args.offline)
        for spec in specs
    }
    canonical_paths = {
        spec.source_id: canonical_repository_paths(spec, checkouts[spec.source_id])
        for spec in specs
    }
    summary = materialize_unified_corpus(
        specs,
        checkouts,
        args.output_root,
        force=args.force,
        include_linked_files=not args.no_linked_files,
        canonical_paths=canonical_paths,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
