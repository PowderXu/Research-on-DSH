"""Migrate existing QA packages locally, without rebuilding or model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from dataset.scripts.records import PARTITION_FIELDS, RECORD_LAYOUT, load_records, write_records


def migrate_package(dataset_dir: Path, backup_dir: Path) -> dict:
    dataset_dir = dataset_dir.resolve()
    records = load_records(dataset_dir)
    expected = [{k: v for k, v in row.items() if k not in PARTITION_FIELDS} for row in records]
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("record_layout") == RECORD_LAYOUT and not (dataset_dir / "splits").exists():
        return {"dataset": str(dataset_dir), "records": len(records), "status": "already_migrated"}
    backup_dir.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha256(str(dataset_dir).encode()).hexdigest()[:12]
    archive = backup_dir / f"{dataset_dir.name}-{name}.zip"
    # An existing backup is never overwritten by a retry.
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as output:
        for name in ("questions.jsonl", "answers.jsonl", "manifest.json", "verification.json"):
            path = dataset_dir / name
            if path.exists():
                output.write(path, name)
        if (dataset_dir / "splits").exists():
            for path in sorted((dataset_dir / "splits").rglob("*")):
                if path.is_file():
                    output.write(path, str(path.relative_to(dataset_dir)))
    with tempfile.TemporaryDirectory(prefix=".separate-records-", dir=dataset_dir.parent) as temp:
        staging = Path(temp)
        write_records(staging, records)
        if load_records(staging) != expected:
            raise ValueError("migration changed record contents")
        manifest.pop("splits", None)
        manifest.pop("split_counts", None)
        manifest.update(schema_version=3, record_layout=RECORD_LAYOUT, partitioning="none")
        manifest["answers"] = len(records)
        files = {k: v for k, v in (manifest.get("files") or {}).items() if not k.startswith("splits/")}
        for name in ("questions.jsonl", "answers.jsonl"):
            raw = (staging / name).read_bytes()
            files[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        manifest["files"] = files
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        for name in ("answers.jsonl", "questions.jsonl", "manifest.json"):
            (staging / name).replace(dataset_dir / name)
    if (dataset_dir / "splits").exists():
        shutil.rmtree(dataset_dir / "splits")
    verification_path = dataset_dir / "verification.json"
    if verification_path.exists():
        verification = json.loads(verification_path.read_text())
        verification.pop("splits", None)
        verification.update(record_layout=RECORD_LAYOUT, answers=len(records), record_roundtrip_verified=True)
        verification_path.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n")
    if load_records(dataset_dir) != expected:
        raise ValueError("on-disk migration did not preserve records")
    return {"dataset": str(dataset_dir), "records": len(records), "status": "migrated", "backup": str(archive)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", action="append", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    args = parser.parse_args()
    for path in args.dataset_dir:
        print(json.dumps(migrate_package(path, args.backup_dir), ensure_ascii=False))


if __name__ == "__main__":
    main()
