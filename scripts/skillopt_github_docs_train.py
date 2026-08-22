#!/usr/bin/env python3
"""Register the local DSH benchmark adapter, then run official SkillOpt train."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SKILLOPT_ROOT = PROJECT_ROOT / "vendor" / "SkillOpt"


def _load_official_train():
    for path in (str(PROJECT_ROOT), str(SKILLOPT_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)
    script = SKILLOPT_ROOT / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("skillopt_official_train", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load official SkillOpt train entry point: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))
    from kbbench.skillopt_github_docs.credentials import load_openai_key_from_configured_env

    load_openai_key_from_configured_env()
    official = _load_official_train()
    from kbbench.skillopt_github_docs.adapter import GitHubDocsDshAdapter

    official._ENV_REGISTRY["github_docs_dsh"] = GitHubDocsDshAdapter
    official.main()


if __name__ == "__main__":
    main()
