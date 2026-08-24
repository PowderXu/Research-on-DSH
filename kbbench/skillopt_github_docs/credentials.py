"""Minimal local credential bridge for DSH and SkillOpt OpenAI clients."""

from __future__ import annotations

import os
from pathlib import Path


def load_openai_key_from_configured_env() -> bool:
    """Load only OPENAI_API_KEY from the operator-selected env file.

    The file path comes from ``KBBENCH_OPENAI_ENV_FILE``. The file is parsed as
    data, never sourced as shell code, and the key is not logged or persisted.
    SkillOpt's pinned OpenAI-compatible adapter uses its historical
    ``AZURE_OPENAI_API_KEY`` variable, so both clients receive the same value.
    """

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        configured = os.environ.get("KBBENCH_OPENAI_ENV_FILE", "").strip()
        if configured:
            for raw_line in Path(configured).expanduser().read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if line.startswith("export "):
                    line = line[7:].lstrip()
                if not line.startswith("OPENAI_API_KEY="):
                    continue
                key = line.split("=", 1)[1].strip().strip("'\"")
                break
    if not key:
        return False
    os.environ["OPENAI_API_KEY"] = key
    os.environ.setdefault("AZURE_OPENAI_API_KEY", key)
    return True
