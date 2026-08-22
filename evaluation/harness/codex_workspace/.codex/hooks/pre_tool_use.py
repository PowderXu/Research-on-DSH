from __future__ import annotations

import json
import os
import re
import sys
import tempfile
from pathlib import Path


PREFIX = "mcp__techdocs__"
COMPOSITE = f"{PREFIX}techdocs_composite"
SEARCH = f"{PREFIX}techdocs_search"
EXPAND = f"{PREFIX}techdocs_expand"
FETCH = f"{PREFIX}techdocs_fetch"


def deny(reason: str) -> None:
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": reason,
                }
            }
        )
    )


def main() -> None:
    event = json.load(sys.stdin)
    tool = str(event.get("tool_name") or "")
    session_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(event.get("session_id") or "unknown"))
    state_path_value = os.environ.get("KB_EVAL_HOOK_STATE")
    state_path = (
        Path(state_path_value)
        if state_path_value
        else Path(tempfile.gettempdir()) / f"kbbench-codex-hook-{session_id}.json"
    )
    state = {"mode": None, "calls": 0, "search_seen": False, "events": []}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

    mode = str(state.get("mode") or os.environ.get("KB_EVAL_MODE") or "")
    if not mode:
        if tool == COMPOSITE:
            mode = "composite"
        elif tool == SEARCH:
            mode = "primitive"
        state["mode"] = mode or None
    elif not state.get("mode"):
        state["mode"] = mode

    reason = ""
    maximum = 1 if mode == "composite" else 3
    if not tool.startswith(PREFIX):
        reason = "Only the shared techdocs MCP tools are allowed in this eval."
    elif int(state.get("calls", 0)) >= maximum:
        reason = f"Technical-document tool-call limit is {maximum}."
    elif mode == "composite" and tool != COMPOSITE:
        reason = "Composite mode permits only techdocs_composite."
    elif mode == "primitive":
        if tool not in {SEARCH, EXPAND, FETCH}:
            reason = "Tool is not available in primitive mode."
        elif int(state.get("calls", 0)) == 0 and tool != SEARCH:
            reason = "Primitive mode must begin with techdocs_search."
        elif not bool(state.get("search_seen")) and tool != SEARCH:
            reason = "Search must run before expansion or fetch."
    elif mode not in {"composite", "primitive"}:
        reason = f"Unknown KB eval mode: {mode}"

    allowed = not reason
    state.setdefault("events", []).append(
        {
            "tool": tool,
            "allowed": allowed,
            "reason": reason or None,
            "arguments": event.get("tool_input"),
        }
    )
    if allowed:
        state["calls"] = int(state.get("calls", 0)) + 1
        if tool == SEARCH:
            state["search_seen"] = True
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    if reason:
        deny(reason)
        return
    # Per the Codex hook contract, an unchanged allowed call exits 0 without
    # output. `permissionDecision: allow` is reserved for `updatedInput`.


if __name__ == "__main__":
    main()
