"""Minimal local credential bridge for DSH OpenAI clients."""

from __future__ import annotations

import os
from pathlib import Path


def load_openai_key_from_configured_env() -> bool:
    """Load only OPENAI_API_KEY from the operator-selected env file.

    The file path comes from ``KBBENCH_OPENAI_ENV_FILE``. The file is parsed as
    data, never sourced as shell code, and the key is not logged or persisted.
    No credentials are written to benchmark outputs.
    """

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        configured = os.environ.get("KBBENCH_OPENAI_ENV_FILE", "").strip()
        if configured:
            for raw_line in Path(configured).expanduser().read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if line.startswith("export "):
                    line = line[7:].lstrip()
                if "=" not in line:
                    continue
                name, value = line.split("=", 1)
                if name.strip() != "OPENAI_API_KEY":
                    continue
                key = value.strip().strip("'\"")
                break
    if not key:
        return False
    os.environ["OPENAI_API_KEY"] = key
    return True
