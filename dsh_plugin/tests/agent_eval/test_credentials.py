from __future__ import annotations

import os
from pathlib import Path

from dsh_plugin.agent_eval.credentials import load_openai_key_from_configured_env


def test_loader_accepts_spaces_around_env_assignment(
    tmp_path: Path, monkeypatch
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY = 'test-value'\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("KBBENCH_OPENAI_ENV_FILE", str(env_file))

    assert load_openai_key_from_configured_env()
    assert os.environ["OPENAI_API_KEY"] == "test-value"
