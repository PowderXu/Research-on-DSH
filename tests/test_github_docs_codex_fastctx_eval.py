from __future__ import annotations

import json
from pathlib import Path

from kbbench.github_docs_codex_fastctx_eval import _load_items, _trace


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
