from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .shared_harness_eval import (
    _codex_usage,
    _dsh_session_files,
    _dsh_trace,
    _json_lines,
    _load_key,
    _mcp_args,
    _read_dsh_session,
    local_service,
)
from .techdocs_service import TechdocsIndexService


MODEL_PRICING_USD_PER_MILLION = {
    "input_fresh": 0.75,
    "input_cached": 0.075,
    "output": 4.50,
}
DOC_GLOBS = "docs/**/*.txt,docs/**/*.rst,docs/**/*.md,docs/**/*.mdx,README.rst,README.md"
QUOTA_PATTERN = re.compile(
    r"insufficient[_ -]?quota|credit(?:s| balance)?.{0,40}(?:exhausted|exceeded|insufficient)|"
    r"billing.{0,30}(?:limit|hard limit)|(?:monthly|usage) limit.{0,30}(?:reached|exceeded)",
    re.IGNORECASE | re.DOTALL,
)
INFRASTRUCTURE_PATTERN = re.compile(
    r"unknown model|model_not_found|invalid api key|authentication|failed to load|"
    r"configuration error|plugin.+(?:failed|error)|mcp.+startup.+(?:failed|error)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class Arm:
    arm_id: str
    harness: str
    kb: bool
    integration: str
    narration: str = "default"

    @property
    def orchestrated(self) -> bool:
        return self.integration != "none"

    @property
    def automatic(self) -> bool:
        return self.integration == "automatic"

    @property
    def policy(self) -> bool:
        return self.integration in {
            "policy", "policy-generalized", "policy-authority", "policy-corpus-probe"
        }

    @property
    def generalized(self) -> bool:
        return self.integration in {
            "policy-generalized", "policy-authority", "policy-corpus-probe"
        }


ARMS = (
    Arm("C0", "codex", False, "none"),
    Arm("D0", "dsh", False, "none"),
    Arm("C3-S", "codex", True, "skill"),
    Arm("D3-S", "dsh", True, "skill"),
    Arm("C3-A", "codex", True, "automatic"),
    Arm("D3-A", "dsh", True, "automatic"),
)

# A separate, non-destructive treatment that keeps the original six-arm D3
# protocol intact. C3-SF is C3-S plus a bounded FastCtx grep/glob MCP surface;
# D3-S remains the previously defined DSH skill-selected arm.
FASTCTX_D3S_ARMS = (
    Arm("C3-SF", "codex", True, "skill"),
    Arm("D3-S", "dsh", True, "skill"),
)
NARRATION_ABLATION_ARMS = (
    Arm("C3-SF", "codex", True, "skill"),
    Arm("C3-SF-Q", "codex", True, "skill", "silent"),
    Arm("D3-S", "dsh", True, "skill"),
    Arm("D3-S-N", "dsh", True, "skill", "narrated"),
)
ORIGINAL_C3S_PILOT_ARMS = (
    Arm("C3-S", "codex", True, "skill"),
)
D3S_ONLY_ARMS = (
    Arm("D3-S", "dsh", True, "skill"),
)
D3P_PILOT_ARMS = (
    Arm("D3-S", "dsh", True, "skill"),
    Arm("D3-P", "dsh", True, "policy"),
)
D3PG_PILOT_ARMS = (
    Arm("D3-S", "dsh", True, "skill"),
    Arm("D3-PG", "dsh", True, "policy-generalized"),
)
D3PA_HELDOUT_ARMS = (
    Arm("D3-S", "dsh", True, "skill"),
    Arm("D3-PA", "dsh", True, "policy-authority"),
)
D3PC_INTEGRATION_ARMS = (
    Arm("D3-PC", "dsh", True, "policy-corpus-probe"),
)
D3PC_HELDOUT_ARMS = (
    Arm("D3-S", "dsh", True, "skill"),
    Arm("D3-PC", "dsh", True, "policy-corpus-probe"),
)
ARM_SETS = {
    "c3s-pilot": ORIGINAL_C3S_PILOT_ARMS,
    "d3": ARMS,
    "d3s-only": D3S_ONLY_ARMS,
    "d3p-pilot": D3P_PILOT_ARMS,
    "d3pg-pilot": D3PG_PILOT_ARMS,
    "d3pa-heldout": D3PA_HELDOUT_ARMS,
    "d3pc-integration": D3PC_INTEGRATION_ARMS,
    "d3pc-heldout": D3PC_HELDOUT_ARMS,
    "fastctx-d3s": FASTCTX_D3S_ARMS,
    "narration-ablation": NARRATION_ABLATION_ARMS,
}
FASTCTX_VERSION = "0.2.5"
FASTCTX_TOOLS = ("grep", "glob")
SHELL_SEARCH_PATTERN = re.compile(r"(?:^|[\s\"';&|])(?:rg|grep|find)\s")
CODEX_ASSET_ROOT = Path(__file__).resolve().parents[1] / "codex"
CODEX_SILENT_DEVELOPER_INSTRUCTIONS = (
    CODEX_ASSET_ROOT / "silent_narration_instructions.txt"
).read_text(encoding="utf-8").strip()


def coding_prompt(task: dict[str, Any]) -> str:
    return (
        f"Solve SWE-bench instance {task['instance_id']} in the current repository. "
        "Implement the requested fix directly in the working tree. Inspect the repository, "
        "edit the necessary files, and run focused tests or other useful validation. Do not "
        "access the web, search for the original pull request, or ask the user questions. "
        "Finish with a concise summary of the change and tests.\n\n"
        f"Issue:\n{task['problem_statement']}"
    )


def fastctx_coding_prompt(task: dict[str, Any]) -> str:
    return (
        "FILESYSTEM SEARCH POLICY FOR THIS ARM: For repository content search and path "
        "discovery, you must use `mcp__fastctx__grep` and `mcp__fastctx__glob`; do not "
        "run shell `rg`, `grep`, or `find`. The shell remains available for builds, tests, "
        "version-control inspection, and commands that are not repository search.\n\n"
        + coding_prompt(task)
    )


def estimate_paid_usd(usage: dict[str, Any]) -> float:
    return (
        float(usage.get("input_fresh") or 0) * MODEL_PRICING_USD_PER_MILLION["input_fresh"]
        + float(usage.get("input_cached") or 0) * MODEL_PRICING_USD_PER_MILLION["input_cached"]
        + float(usage.get("output") or 0) * MODEL_PRICING_USD_PER_MILLION["output"]
    ) / 1_000_000


def terminal_reason(stdout: str, stderr: str, return_code: int) -> str | None:
    combined = f"{stdout}\n{stderr}"
    if QUOTA_PATTERN.search(combined):
        return "api_credit_or_quota_exhausted"
    if return_code != 0 and INFRASTRUCTURE_PATTERN.search(combined):
        return "model_or_runtime_infrastructure_error"
    return None


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _git(mirror_or_worktree: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = _run(["git", "-C", str(mirror_or_worktree), *args], cwd=mirror_or_worktree.parent)
    if check and completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "git failed")
    return completed


@contextmanager
def detached_worktree(mirror: Path, commit: str, root: Path, label: str) -> Iterable[Path]:
    digest = hashlib.sha256(f"{label}\0{commit}\0{time.time_ns()}".encode()).hexdigest()[:12]
    path = (root / f"{label}-{digest}").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = _run(
        ["git", "-C", str(mirror), "worktree", "add", "--detach", str(path), commit],
        cwd=mirror.parent,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "could not create detached worktree")
    try:
        yield path
    finally:
        _run(
            ["git", "-C", str(mirror), "worktree", "remove", "--force", str(path)],
            cwd=mirror.parent,
        )
        _run(["git", "-C", str(mirror), "worktree", "prune"], cwd=mirror.parent)


def capture_patch(worktree: Path) -> tuple[str, str, list[str]]:
    status = _git(worktree, "status", "--porcelain=v1", "--untracked-files=all").stdout
    untracked = _git(
        worktree,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    ).stdout.split("\0")
    included = [
        value
        for value in untracked
        if value and not value.startswith((".codex/", ".agents/", ".dsh/"))
    ]
    for value in included:
        _git(worktree, "add", "--intent-to-add", "--", value)
    patch = _git(worktree, "diff", "--binary", "--no-ext-diff", "HEAD").stdout
    return patch, status, included


def install_codex_skill(worktree: Path, *, fastctx_search: bool = False) -> None:
    skill_dir = worktree / ".agents/skills/techdocs-research"
    skill_dir.mkdir(parents=True, exist_ok=True)
    fastctx_instructions = (
        "\nFor repository content and path discovery, prefer "
        "`mcp__fastctx__grep` and `mcp__fastctx__glob` over shell `rg`, `grep`, or `find`. "
        "Use the shell normally for builds, tests, and commands that are not repository search.\n"
        if fastctx_search
        else ""
    )
    skill_text = (CODEX_ASSET_ROOT / "techdocs-research/SKILL.md").read_text(
        encoding="utf-8"
    )
    (skill_dir / "SKILL.md").write_text(skill_text + fastctx_instructions, encoding="utf-8")


def add_fastctx_mcp_config(command: list[str], fastctx_bin: Path) -> None:
    command.extend(
        [
            "--config",
            f"mcp_servers.fastctx.command={json.dumps(str(fastctx_bin))}",
            "--config",
            f"mcp_servers.fastctx.args={json.dumps(['serve'])}",
            "--config",
            f"mcp_servers.fastctx.enabled_tools={json.dumps(list(FASTCTX_TOOLS))}",
            "--config",
            "mcp_servers.fastctx.required=true",
            "--config",
            'mcp_servers.fastctx.default_tools_approval_mode="approve"',
            "--config",
            "mcp_servers.fastctx.startup_timeout_sec=30",
            "--config",
            "mcp_servers.fastctx.tool_timeout_sec=30",
        ]
    )


def _text_characters(value: object) -> int:
    if isinstance(value, list):
        return sum(_text_characters(item) for item in value)
    if isinstance(value, dict):
        return sum(
            len(item)
            if key == "text" and isinstance(item, str)
            else _text_characters(item)
            if isinstance(item, (dict, list))
            else 0
            for key, item in value.items()
        )
    return 0


def _codex_repository_search_trace(events_text: str) -> dict[str, Any]:
    sequence: list[str] = []
    fastctx_calls: list[dict[str, Any]] = []
    shell_search_calls: list[dict[str, Any]] = []
    for line in events_text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "item.completed":
            continue
        item = event.get("item") or {}
        item_type = str(item.get("type") or "")
        if item_type == "mcp_tool_call":
            server = str(item.get("server") or "")
            tool = str(item.get("tool") or "")
            name = f"mcp__{server}__{tool}"
            sequence.append(name)
            if server == "fastctx" and tool in FASTCTX_TOOLS:
                fastctx_calls.append(
                    {
                        "tool": tool,
                        "arguments": item.get("arguments") or {},
                        "output_characters": _text_characters(item.get("result")),
                        "error": item.get("error"),
                    }
                )
        elif item_type == "command_execution":
            sequence.append("shell")
            command = str(item.get("command") or "")
            if SHELL_SEARCH_PATTERN.search(command):
                shell_search_calls.append(
                    {
                        "command": command,
                        "output_characters": len(str(item.get("aggregated_output") or "")),
                        "exit_code": item.get("exit_code"),
                    }
                )
        elif item_type == "file_change":
            sequence.append("apply_patch")
    return {
        "session_tool_sequence": sequence,
        "repository_search_backend": "fastctx" if fastctx_calls else "shell-or-none",
        "filesystem_search_calls": len(fastctx_calls),
        "filesystem_search_output_characters": sum(
            int(call["output_characters"]) for call in fastctx_calls
        ),
        "filesystem_search_trace": fastctx_calls,
        "shell_search_calls": len(shell_search_calls),
        "shell_search_output_characters": sum(
            int(call["output_characters"]) for call in shell_search_calls
        ),
        "shell_search_trace": shell_search_calls,
    }


def _codex_response_trace(events_text: str) -> dict[str, int]:
    messages: list[str] = []
    for line in events_text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "item.completed":
            continue
        item = event.get("item") or {}
        if item.get("type") == "agent_message":
            text = str(item.get("text") or "")
            if text.strip():
                messages.append(text)
    progress = messages[:-1]
    final = messages[-1:]
    return {
        "text_messages": len(messages),
        "progress_text_messages": len(progress),
        "progress_text_characters": sum(len(value) for value in progress),
        "final_text_characters": sum(len(value) for value in final),
        "mixed_text_tool_steps": 0,
        "tool_only_steps": 0,
        "multi_tool_steps": 0,
    }


def _dsh_repository_search_trace(events: list[dict[str, Any]]) -> dict[str, Any]:
    calls: dict[str, dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []
    for event in events:
        data = event.get("data") or {}
        if event.get("type") == "tool/call":
            tool = str(data.get("name") or "")
            if tool in (*FASTCTX_TOOLS, "bash"):
                calls[str(data.get("callId") or "")] = {
                    "tool": tool,
                    "arguments": data.get("arguments"),
                    "output_characters": 0,
                }
        elif event.get("type") == "tool/result":
            message = data.get("message") or {}
            call_id = str((message.get("source") or {}).get("callId") or "")
            if call_id in calls:
                call = calls.pop(call_id)
                call["output_characters"] = _text_characters(message.get("content"))
                completed.append(call)
    completed.extend(calls.values())
    filesystem_calls = [call for call in completed if call["tool"] in FASTCTX_TOOLS]
    shell_search_calls: list[dict[str, Any]] = []
    for call in completed:
        if call["tool"] != "bash":
            continue
        arguments = call.get("arguments")
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            parsed = {}
        command = str((parsed or {}).get("command") or "") if isinstance(parsed, dict) else ""
        if SHELL_SEARCH_PATTERN.search(command):
            shell_search_calls.append(call)
    return {
        "repository_search_backend": "dsh-tool-fs-search" if filesystem_calls else "none",
        "filesystem_search_calls": len(filesystem_calls),
        "filesystem_search_output_characters": sum(
            int(call["output_characters"]) for call in filesystem_calls
        ),
        "filesystem_search_trace": filesystem_calls,
        "shell_search_calls": len(shell_search_calls),
        "shell_search_output_characters": sum(
            int(call["output_characters"]) for call in shell_search_calls
        ),
        "shell_search_trace": shell_search_calls,
    }


def _dsh_response_trace(events: list[dict[str, Any]]) -> dict[str, int]:
    steps: list[dict[str, int]] = []
    for event in events:
        if event.get("type") != "assistant/message":
            continue
        message = (event.get("data") or {}).get("message") or {}
        content = message.get("content") or []
        texts = [
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict)
            and block.get("type") == "text"
            and str(block.get("text") or "").strip()
        ]
        tool_calls = sum(
            int(isinstance(block, dict) and block.get("type") == "tool-call")
            for block in content
        )
        steps.append({"text_characters": sum(map(len, texts)), "tool_calls": tool_calls})
    text_steps = [step for step in steps if step["text_characters"] > 0]
    final_step = text_steps[-1:] if text_steps else []
    progress_steps = text_steps[:-1]
    return {
        "text_messages": len(text_steps),
        "progress_text_messages": len(progress_steps),
        "progress_text_characters": sum(
            step["text_characters"] for step in progress_steps
        ),
        "final_text_characters": sum(step["text_characters"] for step in final_step),
        "mixed_text_tool_steps": sum(
            int(step["text_characters"] > 0 and step["tool_calls"] > 0) for step in steps
        ),
        "tool_only_steps": sum(
            int(step["text_characters"] == 0 and step["tool_calls"] > 0) for step in steps
        ),
        "multi_tool_steps": sum(int(step["tool_calls"] > 1) for step in steps),
    }


def _case_artifacts(output_dir: Path, task_id: str, arm_id: str) -> dict[str, Path]:
    raw = (output_dir / "raw" / task_id / arm_id).resolve()
    raw.mkdir(parents=True, exist_ok=True)
    return {
        "raw": raw,
        "stdout": raw / "stdout.txt",
        "stderr": raw / "stderr.txt",
        "mcp_trace": raw / "mcp_trace.jsonl",
        "kb_trace": raw / "kb_trace.jsonl",
        "last_message": raw / "last_message.txt",
        "patch": raw / "model.patch",
        "status": raw / "git_status.txt",
    }


def _mcp_config_args(
    python: Path,
    project_root: Path,
    endpoint: str,
    trace: Path,
    result_limit: int,
    evidence_token_budget: int,
) -> list[str]:
    return _mcp_args(
        python,
        project_root,
        endpoint,
        "composite",
        trace,
        result_limit,
        evidence_token_budget,
    )


def run_codex_case(
    *,
    task: dict[str, Any],
    arm: Arm,
    worktree: Path,
    endpoint: str,
    output_dir: Path,
    project_root: Path,
    python: Path,
    codex_bin: Path,
    fastctx_bin: Path | None,
    model: str,
    reasoning_effort: str,
    timeout_seconds: int,
    result_limit: int,
    evidence_token_budget: int,
) -> dict[str, Any]:
    artifacts = _case_artifacts(output_dir, str(task["instance_id"]), arm.arm_id)
    uses_fastctx = arm.arm_id.startswith("C3-SF")
    if arm.orchestrated:
        install_codex_skill(worktree, fastctx_search=uses_fastctx)
    prompt = fastctx_coding_prompt(task) if uses_fastctx else coding_prompt(task)
    command = [
        str(codex_bin),
        "exec",
        "-C",
        str(worktree),
        "--ephemeral",
        "--sandbox",
        "workspace-write",
        "--color",
        "never",
        "--json",
        "--ignore-user-config",
        "--config",
        'approval_policy="never"',
        "--config",
        f'model_reasoning_effort={json.dumps(reasoning_effort)}',
        "--disable",
        "apps",
        "--disable",
        "browser_use",
        "--disable",
        "computer_use",
        "--model",
        model,
        "--output-last-message",
        str(artifacts["last_message"]),
    ]
    if arm.narration == "silent":
        command.extend(
            [
                "--config",
                f"developer_instructions={json.dumps(CODEX_SILENT_DEVELOPER_INSTRUCTIONS)}",
            ]
        )
    if arm.automatic:
        hook_command = " ".join(
            shlex.quote(value)
            for value in (
                "/usr/bin/env",
                f"PYTHONPATH={project_root}",
                f"KB_EVAL_KB_TRACE={artifacts['kb_trace']}",
                f"KB_ROUTING_RULES={project_root / 'dsh-techdocs-plugin/routing-rules.json'}",
                f"KB_ENDPOINT={endpoint}",
                f"KB_RESULT_LIMIT={result_limit}",
                f"KB_EVIDENCE_TOKEN_BUDGET={evidence_token_budget}",
                str(python),
                "-m",
                "kbbench.codex_techdocs_hook",
            )
        )
        hook_config = (
            "hooks.UserPromptSubmit=[{hooks=[{type=\"command\",command="
            f"{json.dumps(hook_command)},timeout=35,"
            f"additionalContextLimit={max(4000, evidence_token_budget * 6)}"
            "}]}]"
        )
        command.extend(
            [
                "--config",
                hook_config,
                "--enable",
                "hooks",
                "--dangerously-bypass-hook-trust",
            ]
        )
    if arm.kb:
        mcp = _mcp_config_args(
            python,
            project_root,
            endpoint,
            artifacts["mcp_trace"],
            result_limit,
            evidence_token_budget,
        )
        command.extend(
            [
                "--config",
                f"mcp_servers.techdocs.command={json.dumps(mcp[0])}",
                "--config",
                f"mcp_servers.techdocs.args={json.dumps(mcp[1:])}",
                "--config",
                "mcp_servers.techdocs.required=true",
                "--config",
                'mcp_servers.techdocs.default_tools_approval_mode="approve"',
                "--config",
                "mcp_servers.techdocs.startup_timeout_sec=30",
                "--config",
                "mcp_servers.techdocs.tool_timeout_sec=30",
            ]
        )
    if uses_fastctx:
        if fastctx_bin is None or not fastctx_bin.is_file():
            raise FileNotFoundError(
                f"FastCtx {FASTCTX_VERSION} executable was not found: {fastctx_bin}"
            )
        add_fastctx_mcp_config(command, fastctx_bin)
    command.append(prompt)
    started = time.perf_counter()
    try:
        completed = _run(command, cwd=worktree, timeout=timeout_seconds)
        stdout, stderr, return_code = completed.stdout, completed.stderr, completed.returncode
        error = None
    except (subprocess.TimeoutExpired, OSError) as exc:
        stdout = _timeout_text(getattr(exc, "stdout", ""))
        stderr = _timeout_text(getattr(exc, "stderr", ""))
        return_code = -1
        error = f"{type(exc).__name__}: {exc}"
    latency = time.perf_counter() - started
    artifacts["stdout"].write_text(stdout, encoding="utf-8")
    artifacts["stderr"].write_text(stderr, encoding="utf-8")
    patch, status, added = capture_patch(worktree)
    artifacts["patch"].write_text(patch, encoding="utf-8")
    artifacts["status"].write_text(status, encoding="utf-8")
    mcp_events = _json_lines(artifacts["mcp_trace"])
    kb_events = _json_lines(artifacts["kb_trace"])
    usage = _codex_usage(stdout)
    row = _result_row(
        task=task,
        arm=arm,
        model=model,
        reasoning_effort=reasoning_effort,
        return_code=return_code,
        error=error,
        latency=latency,
        stdout=stdout,
        stderr=stderr,
        usage=usage,
        mcp_events=mcp_events,
        kb_events=kb_events,
        patch=patch,
        status=status,
        added=added,
        raw_dir=artifacts["raw"],
        actual_models=[model] if return_code == 0 else [],
        session_tool_sequence=[],
        model_steps=None,
    )
    row.update(_codex_repository_search_trace(stdout))
    row.update(_codex_response_trace(stdout))
    row["configured_filesystem_search_backend"] = "fastctx" if uses_fastctx else "codex-shell"
    row["fastctx_version"] = FASTCTX_VERSION if uses_fastctx else None
    row["filesystem_search_compliant"] = not uses_fastctx or row["shell_search_calls"] == 0
    return row


def _write_dsh_patches(
    *,
    artifacts: dict[str, Path],
    arm: Arm,
    model: str,
    reasoning_effort: str,
    project_root: Path,
    python: Path,
    endpoint: str,
    worktree: Path,
    result_limit: int,
    evidence_token_budget: int,
) -> tuple[Path, Path]:
    model_patch = artifacts["raw"] / "model.patch.yml"
    runtime_patch = artifacts["raw"] / "runtime.patch.yml"
    model_patch.write_text(
        "\n".join(
            [
                "- id: agent-default-model",
                "  config:",
                "    provider: openai",
                f"    model: {json.dumps(model)}",
                f"    reasoningEffort: {json.dumps(reasoning_effort)}",
                "",
                "- id: llm-pi-ai",
                "  config:",
                "    providers:",
                "      openai:",
                "        apiKeyEnv: OPENAI_API_KEY",
                "        models:",
                f"          - id: {json.dumps(model)}",
                "            contextWindow: 400000",
                "            maxTokens: 8192",
                "",
            ]
        ),
        encoding="utf-8",
    )
    lines = [
        "- id: session-title-llm",
        "  disabled: true",
        "- id: approval",
        "  config:",
        "    policy: never",
        "- id: permission",
        "  config:",
        "    defaultPreset: benchmark-workspace-write",
        "    presets:",
        "      benchmark-workspace-write:",
        "        sandbox: workspace-write",
        "        approval: never",
        "        name: benchmark-workspace-write",
        "        description: Non-interactive benchmark inside the task worktree.",
        "- id: web",
        "  disabled: true",
        "- id: web-search-deepseek",
        "  disabled: true",
        "- id: tool-web",
        "  disabled: true",
    ]
    provider_evidence_budget = 600 if arm.policy else evidence_token_budget
    provider_id = "repo-generalized" if arm.generalized else "repo-local"
    lines.extend(
        [
            "- id: kbbench-techdocs-runtime",
            "  disabled: true",
            "- id: kbbench-narration-ablation",
            f"  disabled: {str(arm.narration != 'narrated').lower()}",
            "- id: kbbench-techdocs-capability",
            f"  disabled: {str(not arm.kb).lower()}",
            "  config:",
            f"    provider: {provider_id}",
            "- id: kbbench-techdocs-provider-repo",
            f"  disabled: {str(not arm.kb or arm.generalized).lower()}",
            "  config:",
            f"    endpoint: {json.dumps(endpoint)}",
            f"    resultLimit: {result_limit}",
            f"    evidenceTokenBudget: {provider_evidence_budget}",
            "    graphExpansion: true",
            f"    policyGuided: {str(arm.policy).lower()}",
            f"    tracePath: {json.dumps(str(artifacts['kb_trace']))}",
            "- id: kbbench-techdocs-provider-repo-generalized",
            f"  disabled: {str(not arm.generalized).lower()}",
            "  config:",
            f"    endpoint: {json.dumps(endpoint)}",
            f"    resultLimit: {result_limit}",
            f"    evidenceTokenBudget: {provider_evidence_budget}",
            "    graphExpansion: true",
            f"    generalizedPolicy: {str(arm.generalized).lower()}",
            f"    tracePath: {json.dumps(str(artifacts['kb_trace']))}",
            "- id: kbbench-techdocs-intent-capability",
            f"  disabled: {str(not arm.generalized).lower()}",
            "  config:",
            "    provider: llm-intent",
            "- id: kbbench-techdocs-intent-provider-llm",
            f"  disabled: {str(not arm.generalized).lower()}",
            "  config:",
            "    provider: openai",
            f"    model: {json.dumps(model)}",
            f"    reasoningEffort: {json.dumps(reasoning_effort)}",
            f"    tracePath: {json.dumps(str(artifacts['kb_trace']))}",
            "- id: kbbench-tool-techdocs",
            f"  disabled: {str(not arm.kb or arm.policy).lower()}",
            "  config:",
            "    forceGraph: true",
            "- id: kbbench-techdocs-observation-state",
            f"  disabled: {str(not arm.policy).lower()}",
            "- id: kbbench-tool-techdocs-policy",
            f"  disabled: {str(not arm.policy or arm.generalized).lower()}",
            "  config:",
            "    forceGraph: false",
            "- id: kbbench-skill-techdocs",
            f"  disabled: {str(not arm.kb).lower()}",
            "- id: kbbench-techdocs-routing-capability",
            f"  disabled: {str(not arm.automatic).lower()}",
            "  config:",
            "    provider: rules",
            "- id: kbbench-techdocs-routing-rules",
            f"  disabled: {str(not arm.automatic).lower()}",
            "  config:",
            f"    rulesPath: {json.dumps(str(project_root / 'dsh-techdocs-plugin/routing-rules.json'))}",
            "- id: kbbench-techdocs-coding-consumer",
            f"  disabled: {str(not arm.automatic).lower()}",
            "  config:",
            f"    tracePath: {json.dumps(str(artifacts['kb_trace']))}",
            "- id: kbbench-techdocs-policy-consumer",
            f"  disabled: {str(not arm.policy or arm.generalized).lower()}",
            "  config:",
            f"    tracePath: {json.dumps(str(artifacts['kb_trace']))}",
            "- id: kbbench-techdocs-generalized-consumer",
            f"  disabled: {str(not arm.generalized).lower()}",
            "  config:",
            f"    tracePath: {json.dumps(str(artifacts['kb_trace']))}",
        ]
    )
    runtime_patch.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return model_patch, runtime_patch


def run_dsh_case(
    *,
    task: dict[str, Any],
    arm: Arm,
    worktree: Path,
    endpoint: str,
    output_dir: Path,
    project_root: Path,
    python: Path,
    dsh_bin: Path,
    dsh_home: Path,
    model: str,
    reasoning_effort: str,
    timeout_seconds: int,
    result_limit: int,
    evidence_token_budget: int,
    openai_api_key: str,
) -> dict[str, Any]:
    artifacts = _case_artifacts(output_dir, str(task["instance_id"]), arm.arm_id)
    model_patch, runtime_patch = _write_dsh_patches(
        artifacts=artifacts,
        arm=arm,
        model=model,
        reasoning_effort=reasoning_effort,
        project_root=project_root,
        python=python,
        endpoint=endpoint,
        worktree=worktree,
        result_limit=result_limit,
        evidence_token_budget=evidence_token_budget,
    )
    before = _dsh_session_files(dsh_home)
    command = [
        str(dsh_bin),
        "--profile",
        "headless",
        "--patch",
        str(model_patch),
        "--patch",
        str(runtime_patch),
        coding_prompt(task),
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "DSH_HOME": str(dsh_home),
            "DSH_PERMISSION_MODE": "workspace-write",
            "DSH_TELEMETRY_MODE": "DISABLED",
            "OPENAI_API_KEY": openai_api_key,
        }
    )
    started = time.perf_counter()
    try:
        completed = _run(command, cwd=worktree, env=environment, timeout=timeout_seconds)
        stdout, stderr, return_code = completed.stdout, completed.stderr, completed.returncode
        error = None
    except (subprocess.TimeoutExpired, OSError) as exc:
        stdout = _timeout_text(getattr(exc, "stdout", ""))
        stderr = _timeout_text(getattr(exc, "stderr", ""))
        return_code = -1
        error = f"{type(exc).__name__}: {exc}"
    latency = time.perf_counter() - started
    artifacts["stdout"].write_text(stdout, encoding="utf-8")
    artifacts["stderr"].write_text(stderr, encoding="utf-8")
    new_sessions = sorted(
        _dsh_session_files(dsh_home) - before,
        key=lambda value: value.stat().st_mtime,
    )
    events = _read_dsh_session(new_sessions[-1]) if new_sessions else []
    trace = _dsh_trace(events)
    patch, status, added = capture_patch(worktree)
    artifacts["patch"].write_text(patch, encoding="utf-8")
    artifacts["status"].write_text(status, encoding="utf-8")
    mcp_events = _json_lines(artifacts["mcp_trace"])
    kb_events = _json_lines(artifacts["kb_trace"])
    row = _result_row(
        task=task,
        arm=arm,
        model=model,
        reasoning_effort=reasoning_effort,
        return_code=return_code,
        error=error,
        latency=latency,
        stdout=stdout,
        stderr=stderr,
        usage=dict(trace["usage"]),
        mcp_events=mcp_events,
        kb_events=kb_events,
        patch=patch,
        status=status,
        added=added,
        raw_dir=artifacts["raw"],
        actual_models=list(trace["actual_models"]),
        session_tool_sequence=list(trace["session_tool_sequence"]),
        model_steps=int(trace["model_steps"]),
    )
    row.update(_dsh_repository_search_trace(events))
    row.update(_dsh_response_trace(events))
    row["configured_filesystem_search_backend"] = "dsh-tool-fs-search"
    row["fastctx_version"] = None
    row["filesystem_search_compliant"] = row["shell_search_calls"] == 0
    row["session_file"] = str(new_sessions[-1]) if new_sessions else None
    return row


def _timeout_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _result_row(
    *,
    task: dict[str, Any],
    arm: Arm,
    model: str,
    reasoning_effort: str,
    return_code: int,
    error: str | None,
    latency: float,
    stdout: str,
    stderr: str,
    usage: dict[str, Any],
    mcp_events: list[dict[str, Any]],
    kb_events: list[dict[str, Any]],
    patch: str,
    status: str,
    added: list[str],
    raw_dir: Path,
    actual_models: list[str],
    session_tool_sequence: list[str],
    model_steps: int | None,
) -> dict[str, Any]:
    usage = usage_with_auxiliary_calls(usage, kb_events)
    tools = [str(event.get("tool") or "") for event in mcp_events]
    retrieval_events = [event for event in kb_events if event.get("event") == "retrieval"]
    route_events = [event for event in kb_events if event.get("event") == "route"]
    kb_errors = [
        str(event.get("error"))
        for event in kb_events
        if event.get("error")
    ]
    reason = terminal_reason(stdout, stderr, return_code)
    return {
        "instance_id": str(task["instance_id"]),
        "stratum": str(task["stratum"]),
        "difficulty": str(task.get("difficulty") or ""),
        "base_commit": str(task["base_commit"]),
        "arm": arm.arm_id,
        "harness": arm.harness,
        "kb_available": arm.kb,
        "orchestrated": arm.orchestrated,
        "integration": arm.integration,
        "narration": arm.narration,
        "requested_model": model,
        "actual_models": actual_models,
        "reasoning_effort": reasoning_effort,
        "return_code": return_code,
        "error": error,
        "terminal_reason": reason,
        "latency_seconds": latency,
        "usage": usage,
        "estimated_paid_usd": estimate_paid_usd(usage) if arm.harness == "dsh" else 0.0,
        "mcp_tool_calls": len(mcp_events),
        "mcp_tool_sequence": tools,
        "mcp_errors": [str(event.get("error")) for event in mcp_events if event.get("error")],
        "kb_retrieval_calls": len(mcp_events) + len(retrieval_events),
        "route_decisions": [
            {
                "decision": event.get("decision"),
                "allow_graph": event.get("allowGraph"),
                "reason": event.get("reason"),
            }
            for event in route_events
        ],
        "kb_trace_errors": kb_errors,
        "session_tool_sequence": session_tool_sequence,
        "model_steps": model_steps,
        "patch_sha256": hashlib.sha256(patch.encode("utf-8")).hexdigest(),
        "patch_characters": len(patch),
        "changed_status": status.splitlines(),
        "captured_untracked_files": added,
        "raw_directory": str(raw_dir),
        "completed_at_unix": time.time(),
    }


def usage_with_auxiliary_calls(
    usage: dict[str, Any], kb_events: list[dict[str, Any]]
) -> dict[str, int]:
    """Add direct intent/verifier model calls that are outside the agent loop log."""

    combined = {str(key): int(value or 0) for key, value in usage.items()}
    for event in kb_events:
        if event.get("event") not in {"intent_compile", "evidence_verify"}:
            continue
        auxiliary = event.get("usage") or {}
        fresh = int(auxiliary.get("inputTokens") or 0) + int(
            auxiliary.get("cacheWriteTokens") or 0
        )
        cached = int(auxiliary.get("cacheReadTokens") or 0)
        output = int(auxiliary.get("outputTokens") or 0)
        combined["input_fresh"] = int(combined.get("input_fresh") or 0) + fresh
        combined["input_cached"] = int(combined.get("input_cached") or 0) + cached
        combined["output"] = int(combined.get("output") or 0) + output
        combined["reasoning_output"] = int(combined.get("reasoning_output") or 0) + int(
            auxiliary.get("reasoningTokens") or 0
        )
    combined["input_total"] = int(combined.get("input_fresh") or 0) + int(
        combined.get("input_cached") or 0
    )
    combined["total"] = combined["input_total"] + int(combined.get("output") or 0)
    return combined


def _difficulty_rank(value: str) -> int:
    normalized = value.lower()
    if "<15" in normalized:
        return 0
    if "15 min" in normalized:
        return 1
    if "1-4" in normalized:
        return 2
    return 3


def schedule_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    strata = ("code_only", "latent_mixed", "direct_document")
    grouped: dict[str, list[dict[str, Any]]] = {value: [] for value in strata}
    for task in tasks:
        grouped.setdefault(str(task["stratum"]), []).append(task)
    for values in grouped.values():
        values.sort(
            key=lambda task: (
                _difficulty_rank(str(task.get("difficulty") or "")),
                str(task["instance_id"]),
            )
        )
    ordered: list[dict[str, Any]] = []
    while any(grouped.get(value) for value in strata):
        for stratum in strata:
            if grouped.get(stratum):
                ordered.append(grouped[stratum].pop(0))
    return ordered


def load_tasks(manifest_path: Path, parquet_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import pandas as pd

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_values = [dict(value) for value in manifest["tasks"]]
    parent_name = manifest.get("parent_manifest")
    if parent_name:
        project_root = manifest_path.resolve().parents[2]
        parent_path = (project_root / str(parent_name)).resolve()
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        parent_tasks = {
            str(value["instance_id"]): dict(value) for value in parent["tasks"]
        }
        task_values = [
            {**parent_tasks.get(str(value["instance_id"]), {}), **value}
            for value in task_values
        ]
        manifest.setdefault("dataset", parent.get("dataset"))
    selected = {str(value["instance_id"]): value for value in task_values}
    frame = pd.read_parquet(parquet_path)
    records = {
        str(row["instance_id"]): row
        for row in frame.to_dict(orient="records")
        if str(row["instance_id"]) in selected
    }
    tasks: list[dict[str, Any]] = []
    for instance_id, selection in selected.items():
        if instance_id not in records:
            raise ValueError(f"selected task is missing from parquet: {instance_id}")
        record = records[instance_id]
        tasks.append(
            {
                "instance_id": instance_id,
                "repo": str(record["repo"]),
                "base_commit": str(record["base_commit"]),
                "problem_statement": str(record["problem_statement"]),
                "difficulty": str(selection.get("difficulty") or record.get("difficulty") or ""),
                "stratum": str(selection["stratum"]),
            }
        )
    return manifest, schedule_tasks(tasks)


def _state_path(output_dir: Path) -> Path:
    return output_dir / "run_state.json"


def save_state(output_dir: Path, state: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = _state_path(output_dir)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    predictions: dict[str, list[dict[str, str]]] = {}
    for row in state["rows"]:
        patch_path = Path(str(row["raw_directory"])) / "model.patch"
        patch = patch_path.read_text(encoding="utf-8") if patch_path.exists() else ""
        predictions.setdefault(str(row["arm"]), []).append(
            {
                "instance_id": str(row["instance_id"]),
                "model_name_or_path": f"{row['harness']}-{row['arm']}-{state['model']}",
                "model_patch": patch,
            }
        )
    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for arm, values in predictions.items():
        (prediction_dir / f"{arm}.json").write_text(
            json.dumps(values, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def total_paid_usd(state: dict[str, Any]) -> float:
    rows = list(state.get("rows") or []) + list(state.get("failed_attempts") or [])
    return sum(
        float(row.get("estimated_paid_usd") or 0)
        for row in rows
        if row.get("harness") == "dsh"
    )


def summarize_state(state: dict[str, Any]) -> dict[str, Any]:
    rows = list(state["rows"])
    active_arms = tuple(Arm(**value) for value in state["arms"])
    by_arm: dict[str, dict[str, Any]] = {}
    for arm in active_arms:
        selected = [row for row in rows if row["arm"] == arm.arm_id]
        if not selected:
            continue
        by_arm[arm.arm_id] = {
            "runs": len(selected),
            "successful_processes": sum(int(row["return_code"] == 0) for row in selected),
            "mean_latency_seconds": sum(float(row["latency_seconds"]) for row in selected)
            / len(selected),
            "mean_tokens": sum(float(row["usage"].get("total") or 0) for row in selected)
            / len(selected),
            "mean_kb_calls": sum(
                int(row.get("kb_retrieval_calls", row.get("mcp_tool_calls", 0)))
                for row in selected
            )
            / len(selected),
            "mean_filesystem_search_calls": sum(
                int(row.get("filesystem_search_calls") or 0) for row in selected
            )
            / len(selected),
            "mean_filesystem_search_output_characters": sum(
                int(row.get("filesystem_search_output_characters") or 0) for row in selected
            )
            / len(selected),
            "mean_shell_search_calls": sum(
                int(row.get("shell_search_calls") or 0) for row in selected
            )
            / len(selected),
            "mean_shell_search_output_characters": sum(
                int(row.get("shell_search_output_characters") or 0) for row in selected
            )
            / len(selected),
            "mean_output_tokens": sum(
                int((row.get("usage") or {}).get("output") or 0) for row in selected
            )
            / len(selected),
            "mean_text_messages": sum(
                int(row.get("text_messages") or 0) for row in selected
            )
            / len(selected),
            "mean_progress_text_characters": sum(
                int(row.get("progress_text_characters") or 0) for row in selected
            )
            / len(selected),
            "mean_final_text_characters": sum(
                int(row.get("final_text_characters") or 0) for row in selected
            )
            / len(selected),
            "mean_mixed_text_tool_steps": sum(
                int(row.get("mixed_text_tool_steps") or 0) for row in selected
            )
            / len(selected),
            "mean_tool_only_steps": sum(
                int(row.get("tool_only_steps") or 0) for row in selected
            )
            / len(selected),
            "mean_multi_tool_steps": sum(
                int(row.get("multi_tool_steps") or 0) for row in selected
            )
            / len(selected),
            "filesystem_search_compliance_rate": sum(
                int(bool(row.get("filesystem_search_compliant", True))) for row in selected
            )
            / len(selected),
            "estimated_paid_usd": sum(float(row["estimated_paid_usd"]) for row in selected),
        }
    completed_tasks = [
        task_id
        for task_id in sorted({str(row["instance_id"]) for row in rows})
        if {str(row["arm"]) for row in rows if row["instance_id"] == task_id}
        == {arm.arm_id for arm in active_arms}
    ]
    return {
        "terminal_reason": state.get("terminal_reason"),
        "attempted_runs": len(rows),
        "complete_arm_set_tasks": len(completed_tasks),
        "complete_task_ids": completed_tasks,
        "observed_dsh_paid_usd": total_paid_usd(state),
        "failed_attempts": len(state.get("failed_attempts") or []),
        "by_arm": by_arm,
        "accuracy_status": "pending official SWE-bench Docker scoring",
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run paired real-model full coding-agent KB evaluation on SWE-bench Verified."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "config/swebench_fastctx_pilot5_v1.json",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=project_root / "data/swebench_verified/test.parquet",
    )
    parser.add_argument(
        "--mirror",
        type=Path,
        default=project_root / "data/swebench_verified/repos/django.git",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(os.environ.get("PYTHON") or sys.executable),
    )
    parser.add_argument(
        "--codex-bin",
        type=Path,
        default=Path(os.environ.get("CODEX_BIN") or shutil.which("codex") or "codex"),
    )
    parser.add_argument(
        "--fastctx-bin",
        type=Path,
        default=project_root / "node_modules/.bin/fastctx",
    )
    parser.add_argument(
        "--dsh-bin",
        type=Path,
        default=project_root / "node_modules/.bin/dsh",
    )
    parser.add_argument("--dsh-home", type=Path, default=project_root / "dsh_home")
    parser.add_argument(
        "--env-file", type=Path, default=project_root / ".env"
    )
    # Codex with ChatGPT authentication accepts the public alias rather than
    # the API-only dated snapshot. OpenAI currently maps this alias to the sole
    # gpt-5.4-mini-2026-03-17 snapshot; both harnesses therefore use the same
    # underlying model while remaining valid on their respective auth routes.
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--result-limit", type=int, default=8)
    parser.add_argument("--evidence-token-budget", type=int, default=2200)
    parser.add_argument("--max-paid-usd", type=float, default=20.0)
    parser.add_argument("--task-limit", type=int)
    parser.add_argument("--arm-set", choices=sorted(ARM_SETS), default="d3")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Preserve failed attempt metadata, then rerun only nonzero-return-code arms.",
    )
    args = parser.parse_args()
    active_arms = ARM_SETS[args.arm_set]

    openai_key = os.environ.get("OPENAI_API_KEY") or _load_key(args.env_file, "OPENAI_API_KEY")
    if not openai_key:
        raise SystemExit("OPENAI_API_KEY was not found")
    manifest, tasks = load_tasks(args.manifest.resolve(), args.parquet.resolve())
    if args.task_limit is not None:
        tasks = tasks[: max(0, args.task_limit)]
    output_dir = args.output.resolve()
    state_file = _state_path(output_dir)
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding="utf-8"))
    else:
        state = {
            "schema_version": 9
            if args.arm_set == "d3s-only"
            else 8
            if args.arm_set in {"d3pc-integration", "d3pc-heldout"}
            else 7
            if args.arm_set == "d3pa-heldout"
            else 6
            if args.arm_set == "narration-ablation"
            else 5
            if args.arm_set == "d3pg-pilot"
            else 4
            if args.arm_set == "d3p-pilot"
            else 3
            if args.arm_set == "fastctx-d3s"
            else 2,
            "benchmark": (
                "SWE-bench Verified standalone DSH D3-S validation"
                if args.arm_set == "d3s-only"
                else "SWE-bench Verified DSH corpus-probed authority policy comparison"
                if args.arm_set in {"d3pc-integration", "d3pc-heldout"}
                else "SWE-bench Verified DSH authority-separated policy held-out comparison"
                if args.arm_set == "d3pa-heldout"
                else "SWE-bench Verified narration-policy ablation"
                if args.arm_set == "narration-ablation"
                else "SWE-bench Verified FastCtx versus DSH D3-S comparison"
                if args.arm_set == "fastctx-d3s"
                else (
                    "SWE-bench Verified DSH D3-S versus generalized policy-guided D3-P pilot"
                    if args.arm_set == "d3pg-pilot"
                    else "SWE-bench Verified DSH D3-S versus policy-guided D3-P pilot"
                    if args.arm_set == "d3p-pilot"
                    else "SWE-bench Verified TechDocs D3 integration comparison"
                )
            ),
            "arm_set": args.arm_set,
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "manifest_dataset": manifest["dataset"],
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "dsh_paid_budget_usd": args.max_paid_usd,
            "pricing_usd_per_million_tokens": MODEL_PRICING_USD_PER_MILLION,
            "docs_include_globs": DOC_GLOBS,
            "schedule": [task["instance_id"] for task in tasks],
            "arms": [arm.__dict__ for arm in active_arms],
            "fastctx": {
                "version": FASTCTX_VERSION,
                "binary": str(args.fastctx_bin.resolve()),
                "enabled_tools": list(FASTCTX_TOOLS),
            }
            if args.arm_set in {"fastctx-d3s", "narration-ablation"}
            else None,
            "rows": [],
            "terminal_reason": None,
            "started_at_unix": time.time(),
        }
    expected_arms = [arm.__dict__ for arm in active_arms]
    expected_schema = (
        9
        if args.arm_set == "d3s-only"
        else 8
        if args.arm_set in {"d3pc-integration", "d3pc-heldout"}
        else 7
        if args.arm_set == "d3pa-heldout"
        else 6
        if args.arm_set == "narration-ablation"
        else 5
        if args.arm_set == "d3pg-pilot"
        else 4
        if args.arm_set == "d3p-pilot"
        else 3
        if args.arm_set == "fastctx-d3s"
        else 2
    )
    if state.get("schema_version") != expected_schema or state.get("arms") != expected_arms:
        raise SystemExit(
            "The output directory contains an incompatible evaluation state. "
            "Use a new --output directory for the selected arm set."
        )
    if args.retry_failed:
        failed = [row for row in state["rows"] if int(row.get("return_code") or 0) != 0]
        if failed:
            state.setdefault("failed_attempts", []).extend(failed)
            state["rows"] = [row for row in state["rows"] if int(row.get("return_code") or 0) == 0]
            state["terminal_reason"] = None
    completed = {(str(row["instance_id"]), str(row["arm"])) for row in state["rows"]}
    worktree_root = output_dir / "worktrees"
    stop = state.get("terminal_reason")
    scheduled_keys = {
        (str(task["instance_id"]), arm.arm_id)
        for task in tasks
        for arm in active_arms
    }
    if stop == "selected_schedule_completed" and not scheduled_keys.issubset(completed):
        stop = None
        state["terminal_reason"] = None
        state["schedule"] = [task["instance_id"] for task in tasks]
        save_state(output_dir, state)

    for task_index, task in enumerate(tasks):
        if stop:
            break
        if all((task["instance_id"], arm.arm_id) in completed for arm in active_arms):
            continue
        with detached_worktree(
            args.mirror.resolve(), task["base_commit"], worktree_root, f"{task['instance_id']}-kb"
        ) as kb_worktree:
            service = TechdocsIndexService(
                repository_root=kb_worktree,
                github_repository=task["repo"],
                include_glob=DOC_GLOBS,
                default_method="bm25",
                graph_repository_root=kb_worktree,
            )
            with local_service(service) as endpoint:
                if args.arm_set in {"fastctx-d3s", "narration-ablation"}:
                    offset = task_index % len(active_arms)
                    task_arms = active_arms[offset:] + active_arms[:offset]
                else:
                    task_arms = active_arms
                for arm in task_arms:
                    key = (task["instance_id"], arm.arm_id)
                    if key in completed:
                        continue
                    observed = total_paid_usd(state)
                    if arm.harness == "dsh" and observed >= args.max_paid_usd:
                        stop = "observed_paid_budget_reached"
                        break
                    print(
                        json.dumps(
                            {
                                "event": "case_start",
                                "instance_id": task["instance_id"],
                                "stratum": task["stratum"],
                                "arm": arm.arm_id,
                                "observed_paid_usd": observed,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    with detached_worktree(
                        args.mirror.resolve(),
                        task["base_commit"],
                        worktree_root,
                        f"{task['instance_id']}-{arm.arm_id}",
                    ) as agent_worktree:
                        common = {
                            "task": task,
                            "arm": arm,
                            "worktree": agent_worktree,
                            "endpoint": endpoint,
                            "output_dir": output_dir,
                            "project_root": project_root,
                            # Keep a virtual-environment launcher path intact; resolving the
                            # symlink would silently select the base interpreter.
                            "python": Path(os.path.abspath(args.python)),
                            "model": args.model,
                            "reasoning_effort": args.reasoning_effort,
                            "timeout_seconds": args.timeout_seconds,
                            "result_limit": args.result_limit,
                            "evidence_token_budget": args.evidence_token_budget,
                        }
                        if arm.harness == "codex":
                            row = run_codex_case(
                                codex_bin=args.codex_bin.resolve(),
                                fastctx_bin=args.fastctx_bin.resolve(),
                                **common,
                            )
                        else:
                            row = run_dsh_case(
                                dsh_bin=args.dsh_bin.resolve(),
                                dsh_home=args.dsh_home.resolve(),
                                openai_api_key=openai_key,
                                **common,
                            )
                    state["rows"].append(row)
                    completed.add(key)
                    save_state(output_dir, state)
                    print(
                        json.dumps(
                            {
                                "event": "case_end",
                                "instance_id": task["instance_id"],
                                "arm": arm.arm_id,
                                "return_code": row["return_code"],
                                "latency_seconds": round(float(row["latency_seconds"]), 3),
                                "tokens": row["usage"].get("total", 0),
                                "estimated_paid_usd": row["estimated_paid_usd"],
                                "kb_calls": row["kb_retrieval_calls"],
                                "terminal_reason": row["terminal_reason"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    if row["terminal_reason"]:
                        stop = row["terminal_reason"]
                        break
    state["terminal_reason"] = stop or "selected_schedule_completed"
    state["finished_at_unix"] = time.time()
    state["summary"] = summarize_state(state)
    save_state(output_dir, state)
    (output_dir / "summary.json").write_text(
        json.dumps(state["summary"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "evaluation_end", **state["summary"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
