from __future__ import annotations

import json
from pathlib import Path

from kbbench.github_docs_codex_fastctx_eval import (
    _codex_invocation,
    _load_items,
    _trace,
)


ROOT = Path(__file__).resolve().parents[1]


def test_codex_skill_preserves_dsh_filesystem_strategy() -> None:
    dsh = (ROOT / "dsh-techdocs-plugin/skills/fs/initial_skill.md").read_text()
    codex = (ROOT / "codex-techdocs-plugin/skills/github-docs-fastctx/SKILL.md").read_text()
    for number in range(1, 8):
        dsh_line = next(line for line in dsh.splitlines() if line.startswith(f"{number}. "))
        assert dsh_line in codex
    for tool in ("mcp__fastctx__grep", "mcp__fastctx__glob", "mcp__fastctx__read"):
        assert tool in codex


def test_trace_preserves_first_visible_markdown_order_and_flags_shell() -> None:
    events = [
        {
            "type": "item.completed",
            "item": {
                "type": "mcp_tool_call",
                "server": "fastctx",
                "tool": "grep",
                "arguments": {"pattern": "x"},
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": "/repo/content/a/first.md:1:x\n/repo/content/b/second.md:2:x",
                        }
                    ]
                },
            },
        },
        {
            "type": "item.completed",
            "item": {
                "type": "mcp_tool_call",
                "server": "fastctx",
                "tool": "read",
                "arguments": {"path": "content/a/first.md"},
                "result": {"content": [{"type": "text", "text": "content/a/first.md\nbody"}]},
            },
        },
        {
            "type": "item.completed",
            "item": {"type": "command_execution", "command": "rg x content"},
        },
    ]
    trace = _trace("\n".join(json.dumps(event) for event in events))
    assert trace["tool_sequence"] == ["grep", "read"]
    assert trace["visible_sources"] == [
        "repo/content/a/first.md",
        "repo/content/b/second.md",
        "content/a/first.md",
    ]
    assert trace["shell_calls"] == ["rg x content"]


def test_frozen_paired_items_span_train_and_validation() -> None:
    items = _load_items(
        ROOT / "evaluation/skillopt/github_docs_v2_split",
        ["108045", "26749"],
    )
    assert [item["id"] for item in items] == ["108045", "26749"]


def test_codex_invocation_isolated_by_default_and_opt_in_provider_home(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("CODEX_HOME", raising=False)
    common = {
        "codex_bin": tmp_path / "codex",
        "workspace": tmp_path / "workspace",
        "fastctx_bin": tmp_path / "fastctx",
        "schema": tmp_path / "schema.json",
        "last_message": tmp_path / "last-message.json",
        "prompt": "question",
        "skill_content": "instructions",
        "model": "gpt-test",
        "reasoning_effort": "low",
    }

    default_command, default_environment, default_custom = _codex_invocation(
        **common,
        codex_home=tmp_path / "missing-home",
    )
    assert not default_custom
    assert "--ignore-user-config" in default_command
    assert "CODEX_HOME" not in default_environment

    custom_home = tmp_path / "provider-home"
    custom_home.mkdir()
    (custom_home / "config.toml").write_text(
        '[model_providers.gateway]\nname = "gateway"\n', encoding="utf-8"
    )
    custom_command, custom_environment, custom_enabled = _codex_invocation(
        **common,
        codex_home=custom_home,
    )
    assert custom_enabled
    assert "--ignore-user-config" not in custom_command
    assert custom_environment["CODEX_HOME"] == str(custom_home.resolve())
    assert any(
        value.startswith("mcp_servers.fastctx.command=")
        for value in custom_command
    )
