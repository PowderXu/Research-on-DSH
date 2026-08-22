from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from datasets import load_dataset


DATASET = "SWE-bench/SWE-bench_Verified"
DJANGO_REPOSITORY = "https://github.com/django/django.git"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Download the public SWE-bench Verified split and Django mirror."
    )
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--force-parquet", action="store_true")
    args = parser.parse_args()

    trial_root = args.root.resolve()
    data_root = trial_root / "data/swebench_verified"
    parquet = data_root / "test.parquet"
    mirror = data_root / "repos/django.git"
    manifest_path = trial_root / "config/swebench_fastctx_pilot5_v1.json"
    data_root.mkdir(parents=True, exist_ok=True)

    if args.force_parquet or not parquet.is_file():
        dataset = load_dataset(DATASET, split="test")
        dataset.to_parquet(str(parquet))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {str(task["instance_id"]) for task in manifest["tasks"]}
    selected = load_dataset("parquet", data_files=str(parquet), split="train")
    records = {str(row["instance_id"]): row for row in selected}
    available = set(records)
    missing = sorted(expected - available)
    if missing:
        raise SystemExit(f"Downloaded split is missing manifest tasks: {', '.join(missing)}")

    if not mirror.is_dir():
        mirror.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--mirror", DJANGO_REPOSITORY, str(mirror)],
            check=True,
        )

    for instance_id in sorted(expected):
        commit = str(records[instance_id]["base_commit"])
        subprocess.run(
            ["git", "-C", str(mirror), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
        )

    print(f"Parquet: {parquet}")
    print(f"Django mirror: {mirror}")
    print(f"Validated manifest tasks and base commits: {len(expected)}")


if __name__ == "__main__":
    main()
