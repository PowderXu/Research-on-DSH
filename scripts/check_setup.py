from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGES = {
    "datasets": "5.0.1",
    "hnswlib": "0.8.0",
    "markdown-it-py": "4.0.0",
    "numpy": "1.26.4",
    "pandas": "2.3.3",
    "pyarrow": "23.0.1",
    "scikit-learn": "1.7.2",
    "swebench": "5.0.0",
}


def command_version(name: str, command: list[str]) -> tuple[bool, str]:
    executable = shutil.which(command[0])
    if not executable:
        return False, f"{name}: missing ({command[0]} is not on PATH)"
    completed = subprocess.run(
        [executable, *command[1:]], capture_output=True, text=True, check=False
    )
    output = (completed.stdout or completed.stderr).strip().splitlines()
    detail = output[0] if output else f"exit {completed.returncode}"
    return completed.returncode == 0, f"{name}: {detail}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the standalone trial prerequisites.")
    parser.add_argument(
        "--dsh-only",
        action="store_true",
        help="Check only prerequisites needed by the standalone D3-S run and scoring.",
    )
    args = parser.parse_args()

    checks: list[tuple[bool, str]] = []
    for distribution, expected in PYTHON_PACKAGES.items():
        try:
            actual = importlib.metadata.version(distribution)
            checks.append(
                (
                    actual == expected,
                    f"python package {distribution}: {actual} (expected {expected})",
                )
            )
        except Exception as error:  # pragma: no cover - diagnostic script
            checks.append((False, f"python package {distribution}: {error}"))

    # Import the names that differ from their distribution names as a smoke test.
    for module in ("datasets", "hnswlib", "markdown_it", "numpy", "pandas", "pyarrow", "sklearn", "swebench"):
        try:
            importlib.import_module(module)
        except Exception as error:  # pragma: no cover - diagnostic script
            checks.append((False, f"python import {module}: {error}"))

    checks.extend(
        command_version(label, command)
        for label, command in (
            ("git", ["git", "--version"]),
            ("node", ["node", "--version"]),
            ("npm", ["npm", "--version"]),
            ("docker", ["docker", "--version"]),
        )
    )
    if not args.dsh_only:
        checks.append(command_version("codex", ["codex", "--version"]))

    dsh_bin = ROOT / "node_modules/.bin/dsh"
    fastctx_bin = ROOT / "node_modules/.bin/fastctx"
    required_paths = [
        ("DSH CLI", dsh_bin),
        (
            "DSH profile plugin",
            ROOT / "dsh_home/profiles/headless/node_modules/@kbbench/dsh-techdocs",
        ),
        ("trial manifest", ROOT / "config/swebench_fastctx_pilot5_v1.json"),
        ("SWE-bench parquet", ROOT / "data/swebench_verified/test.parquet"),
        ("Django mirror", ROOT / "data/swebench_verified/repos/django.git"),
    ]
    if not args.dsh_only:
        required_paths.append(("FastCtx CLI", fastctx_bin))
    for label, path in required_paths:
        checks.append((path.exists(), f"{label}: {path}"))

    if dsh_bin.is_file():
        completed = subprocess.run(
            [str(dsh_bin), "--version"], capture_output=True, text=True, check=False
        )
        actual = completed.stdout.strip()
        checks.append((actual == "0.1.0-rc.6", f"DSH version: {actual}"))
    if not args.dsh_only and fastctx_bin.is_file():
        completed = subprocess.run(
            [str(fastctx_bin), "--version"], capture_output=True, text=True, check=False
        )
        actual = completed.stdout.strip()
        checks.append((actual == "fastctx 0.2.5", f"FastCtx version: {actual}"))

    manifest = json.loads(
        (ROOT / "config/swebench_fastctx_pilot5_v1.json").read_text(encoding="utf-8")
    )
    checks.append((len(manifest.get("tasks") or []) == 5, "manifest contains five tasks"))
    checks.append((bool(os.environ.get("OPENAI_API_KEY")), "OPENAI_API_KEY is set"))

    for passed, message in checks:
        print(f"[{'ok' if passed else 'missing'}] {message}")
    if not all(passed for passed, _ in checks):
        raise SystemExit(1)

    print(f"[ok] Python {sys.version.split()[0]}; setup is ready")


if __name__ == "__main__":
    main()
