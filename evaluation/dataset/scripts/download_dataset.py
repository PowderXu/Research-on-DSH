"""Download and verify the dataset pinned by dataset_source.json."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import urllib.request
import uuid
from pathlib import Path, PurePosixPath

from .verify import verify_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = PROJECT_ROOT / "evaluation/dataset/templates/dataset_source.json"
DEFAULT_DESTINATION = PROJECT_ROOT / "evaluation/dataset/evaluation_data/normalized"
ALLOWED_FILES = {"questions.jsonl", "answers.jsonl", "corpus.jsonl", "image_text.jsonl", "aspects.jsonl"}


def _check_file(path: Path, expected: dict) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if path.stat().st_size != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError(f"size or SHA-256 mismatch: {path.name}")


def _fetch(source: dict, relative: str, output: Path, repository_dir: Path | None) -> None:
    if PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
        raise ValueError("unsafe repository path")
    if repository_dir is not None:
        with output.open("wb") as stream:
            subprocess.run(["git", "-C", str(repository_dir), "show", f"{source['revision']}:{relative}"], stdout=stream, check=True)
    elif source.get("visibility") == "public":
        url = f"https://raw.githubusercontent.com/{source['repository']}/{source['revision']}/{relative}"
        request = urllib.request.Request(url, headers={"User-Agent": "DocsQA-benchmark-dataset/1"})
        with urllib.request.urlopen(request, timeout=60) as response, output.open("wb") as stream:
            shutil.copyfileobj(response, stream)
    else:
        if shutil.which("gh") is None:
            raise RuntimeError("Private dataset download requires GitHub CLI: install gh and run gh auth login")
        # Base64 avoids CLI sanitization of raw Unicode/control characters and
        # binary gzip bytes. Contents larger than 1 MB require the blob endpoint.
        payload = json.loads(subprocess.check_output([
            "gh", "api", f"repos/{source['repository']}/contents/{relative}?ref={source['revision']}",
        ]))
        if payload.get("encoding") != "base64":
            blob_sha = payload.get("sha", "")
            if not re.fullmatch(r"[0-9a-f]{40}", blob_sha):
                raise ValueError("invalid GitHub blob SHA")
            payload = json.loads(subprocess.check_output([
                "gh", "api", f"repos/{source['repository']}/git/blobs/{blob_sha}",
            ]))
        if payload.get("encoding") != "base64":
            raise ValueError("GitHub did not return base64 blob content")
        output.write_bytes(base64.b64decode("".join(payload["content"].split()), validate=True))


def download_dataset(source_file: Path = DEFAULT_SOURCE, destination: Path = DEFAULT_DESTINATION,
                     *, repository_dir: Path | None = None) -> dict:
    source = json.loads(source_file.read_text(encoding="utf-8"))
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", source["repository"]):
        raise ValueError("repository must be owner/name")
    if not re.fullmatch(r"[0-9a-f]{40}", source["revision"]):
        raise ValueError("dataset revision must be a full commit SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", source["manifest_sha256"]):
        raise ValueError("invalid manifest SHA-256")
    destination = destination.resolve()
    marker = destination / "download_source.json"
    if marker.exists() and json.loads(marker.read_text()) == source:
        if hashlib.sha256((destination / "manifest.json").read_bytes()).hexdigest() != source["manifest_sha256"]:
            raise ValueError("cached dataset manifest SHA-256 mismatch")
        counts = verify_dataset(destination)
        return {"status": "cached", "destination": str(destination), **counts}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".docsqa-download-", dir=destination.parent) as temporary:
        work = Path(temporary)
        staged = work / "package"
        staged.mkdir()
        manifest_path = staged / "manifest.json"
        _fetch(source, "data/manifest.json", manifest_path, repository_dir)
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != source["manifest_sha256"]:
            raise ValueError("dataset manifest SHA-256 mismatch")
        manifest = json.loads(manifest_path.read_text())
        files, downloads = manifest["files"], manifest["downloads"]
        if not {"questions.jsonl", "answers.jsonl", "corpus.jsonl"} <= files.keys():
            raise ValueError("dataset omits required files")
        if files.keys() != downloads.keys() or set(files) - ALLOWED_FILES:
            raise ValueError("unexpected dataset file inventory")
        if manifest.get("partitioning") != "none":
            raise ValueError("expected one unpartitioned question pool")
        for name, expected in files.items():
            artifact = downloads[name]
            if artifact["compression"] not in {"gzip", "none"}:
                raise ValueError("unsupported compression")
            stored = work / (name + ".download")
            _fetch(source, artifact["path"], stored, repository_dir)
            _check_file(stored, artifact)
            target = staged / name
            if artifact["compression"] == "gzip":
                with gzip.open(stored, "rb") as input_file, target.open("wb") as output:
                    remaining = expected["bytes"]
                    while block := input_file.read(min(1024 * 1024, remaining + 1)):
                        remaining -= len(block)
                        if remaining < 0:
                            raise ValueError("decompressed dataset exceeds declared size")
                        output.write(block)
            else:
                shutil.copyfile(stored, target)
            _check_file(target, expected)
        counts = verify_dataset(staged)
        (staged / "download_source.json").write_text(json.dumps(source, indent=2) + "\n")
        previous = None
        if destination.exists():
            if not destination.is_dir():
                raise ValueError("dataset destination is not a directory")
            previous = destination.with_name(f".{destination.name}.previous-{uuid.uuid4().hex[:8]}")
            destination.rename(previous)
        try:
            staged.rename(destination)
        except Exception:
            if previous is not None:
                previous.rename(destination)
            raise
    return {"status": "downloaded", "destination": str(destination), "previous_dataset": str(previous) if previous else None, **counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--repository-dir", type=Path, help="Read the pinned local Git commit without network access")
    args = parser.parse_args()
    print(json.dumps(download_dataset(args.source, args.output, repository_dir=args.repository_dir), indent=2))


if __name__ == "__main__":
    main()
