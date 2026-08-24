#!/usr/bin/env python3
"""Register the local DSH benchmark adapter, then run installed SkillOpt."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))
    from kbbench.skillopt_github_docs.credentials import load_openai_key_from_configured_env
    from kbbench.skillopt_github_docs.upstream import (
        load_skillopt_cli,
        register_environment,
    )

    load_openai_key_from_configured_env()
    official = load_skillopt_cli("eval_only")
    from kbbench.skillopt_github_docs.adapter import GitHubDocsDshAdapter

    register_environment(official, "github_docs_dsh", GitHubDocsDshAdapter)
    official.main()


if __name__ == "__main__":
    main()
