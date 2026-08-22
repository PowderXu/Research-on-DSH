from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .dsh_hotpot_eval import answer_f1, deterministic_sample, normalize_answer, parse_agent_json
from .harness_datasets import FORMATS, load_harness_questions
from .techdocs_service import RESOURCE_ROOT, TechdocsIndexService, TechdocsRequestHandler


HARNESSES = ("codex", "dsh")
MODES = ("composite", "primitive")


def common_prompt(question: dict[str, object]) -> str:
    answer_instruction = (
        "Give a concise but complete technical answer, preserving necessary commands, versions, "
        "configuration values, and ordered steps."
        if question.get("answer_style") == "technical"
        else "Give a short answer."
    )
    return (
        f"SHARED KB HARNESS EVAL {question['id']}. Follow the installed technical-document "
        "benchmark workflow. Return exactly one JSON object with this schema: "
        '{"answer":"short answer","sources":["viking:// evidence URI"]}. '
        "Do not add Markdown or explanation. If the available evidence is insufficient, "
        f"use NOT FOUND. {answer_instruction}\n"
        f"Question: {question['question']}"
    )


def _load_key(path: Path | None, name: str) -> str | None:
    if path is None:
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip("'\"")
    return None


def _json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    output: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            output.append(value)
    return output


def _codex_usage(events_text: str) -> dict[str, int]:
    usage: dict[str, int] = {}
    for line in events_text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "turn.completed" or not isinstance(event.get("usage"), dict):
            continue
        usage = {
            str(key): int(value)
            for key, value in event["usage"].items()
            if isinstance(value, int)
        }
    total_input = usage.get("input_tokens", 0)
    cached = usage.get("cached_input_tokens", 0)
    output = usage.get("output_tokens", 0)
    return {
        "input_fresh": max(0, total_input - cached),
        "input_cached": cached,
        "input_total": total_input,
        "output": output,
        "total": total_input + output,
        "reasoning_output": usage.get("reasoning_output_tokens", 0),
    }


def _dsh_session_files(dsh_home: Path) -> set[Path]:
    return set(dsh_home.glob("sessions/**/session.jsonl.zstd"))


def _read_dsh_session(path: Path) -> list[dict[str, Any]]:
    raw = b""
    if shutil.which("zstdcat") is not None:
        completed = subprocess.run(
            ["zstdcat", str(path)], capture_output=True, check=False
        )
        if completed.returncode == 0:
            raw = completed.stdout
    if not raw:
        # Fall back to the `zstandard` Python package when the zstd CLI is absent.
        try:
            import zstandard
        except ImportError:
            return []
        try:
            with open(path, "rb") as handle:
                dctx = zstandard.ZstdDecompressor()
                raw = dctx.stream_reader(handle).read()
        except (OSError, zstandard.ZstdError):
            return []
    text = raw.decode("utf-8", errors="replace")
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def _dsh_trace(events: list[dict[str, Any]]) -> dict[str, object]:
    fresh = cached = output = 0
    models: set[str] = set()
    tool_names: list[str] = []
    steps: set[tuple[int, int]] = set()
    for event in events:
        event_type = event.get("type")
        data = event.get("data") or {}
        if event_type == "assistant/message":
            message = data.get("message") or {}
            # DSH persists usage beside `message` in current session traces. Keep
            # the older nested location as a compatibility fallback.
            usage = data.get("usage") or message.get("usage") or {}
            fresh += int(usage.get("inputTokens") or 0)
            cached += int(usage.get("cacheReadTokens") or 0)
            output += int(usage.get("outputTokens") or 0)
            source = message.get("source") or {}
            if source.get("model"):
                models.add(str(source["model"]))
        elif event_type == "tool/call":
            tool_names.append(str(data.get("name") or ""))
        elif event_type == "step/start":
            steps.add((int(data.get("turn") or 0), int(data.get("step") or 0)))
    return {
        "usage": {
            "input_fresh": fresh,
            "input_cached": cached,
            "input_total": fresh + cached,
            "output": output,
            "total": fresh + cached + output,
            "reasoning_output": 0,
        },
        "actual_models": sorted(models),
        "session_tool_sequence": tool_names,
        "model_steps": len(steps),
    }


def _source_id(source: str) -> str:
    value = source.split("#", 1)[0].rstrip("/")
    prefix = f"{RESOURCE_ROOT}/"
    return value[len(prefix) :] if value.startswith(prefix) else value.lstrip("/")


def score_prediction(
    prediction: str,
    sources: list[str],
    gold_answer: str,
    relevant_ids: list[str],
) -> dict[str, float]:
    cited = {_source_id(source) for source in sources}
    relevant = set(relevant_ids)
    return {
        "exact_match": float(normalize_answer(prediction) == normalize_answer(gold_answer)),
        "f1": answer_f1(prediction, gold_answer),
        "citation_present": float(bool(sources)),
        "citation_recall": len(cited & relevant) / len(relevant) if relevant else 0.0,
        "citation_all_support": float(bool(relevant) and relevant.issubset(cited)),
    }


def score_evidence(sources: list[str], relevant_ids: list[str]) -> dict[str, float]:
    retrieved = {_source_id(source) for source in sources}
    relevant = set(relevant_ids)
    return {
        "evidence_recall": len(retrieved & relevant) / len(relevant) if relevant else 0.0,
        "evidence_all_support": float(bool(relevant) and relevant.issubset(retrieved)),
    }


@contextmanager
def local_service(service: TechdocsIndexService) -> Iterable[str]:
    handler = type("SharedEvalHandler", (TechdocsRequestHandler,), {"service": service})
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _mcp_args(
    python: Path,
    project_root: Path,
    endpoint: str,
    mode: str,
    trace: Path,
    result_limit: int,
    evidence_token_budget: int,
    fixed_query: str | None = None,
) -> list[str]:
    args = [
        str(python),
        str(project_root / "kbbench/techdocs_mcp.py"),
        "--endpoint",
        endpoint,
        "--mode",
        mode,
        "--result-limit",
        str(result_limit),
        "--evidence-token-budget",
        str(evidence_token_budget),
        "--trace",
        str(trace),
    ]
    if fixed_query:
        args.extend(["--fixed-query", fixed_query])
    return args


def run_codex_case(
    question: dict[str, object],
    mode: str,
    endpoint: str,
    output_dir: Path,
    project_root: Path,
    python: Path,
    codex_bin: Path,
    model: str,
    timeout_seconds: int,
    result_limit: int,
    evidence_token_budget: int,
) -> dict[str, object]:
    case_name = f"{question['id']}-{mode}-codex"
    raw_dir = (output_dir / "raw" / case_name).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    mcp_trace = raw_dir / "mcp_trace.jsonl"
    hook_state = raw_dir / "hook_state.json"
    last_message = raw_dir / "last_message.json"
    events_path = raw_dir / "events.jsonl"
    stderr_path = raw_dir / "stderr.txt"
    schema = project_root / "evaluation/harness/answer_schema.json"
    workspace = project_root / "evaluation/harness/codex_workspace"
    args = _mcp_args(
        python,
        project_root,
        endpoint,
        mode,
        mcp_trace,
        result_limit,
        evidence_token_budget,
        str(question["question"]) if mode == "composite" else None,
    )
    skill = "techdocs-composite-eval" if mode == "composite" else "techdocs-primitive-eval"
    prompt = f"Use ${skill}.\n\n{common_prompt(question)}"
    hook_script = workspace / ".codex/hooks/pre_tool_use.py"
    hook_command = " ".join(
        shlex.quote(value)
        for value in (
            "/usr/bin/env",
            f"KB_EVAL_MODE={mode}",
            f"KB_EVAL_HOOK_STATE={hook_state}",
            "/usr/bin/python3",
            str(hook_script),
        )
    )
    hook_config = (
        'hooks.PreToolUse=[{matcher="*",hooks=[{type="command",command='
        f"{json.dumps(hook_command)},timeout=5"
        "}]}]"
    )
    command = [
        str(codex_bin),
        "exec",
        "-C",
        str(workspace),
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--color",
        "never",
        "--json",
        "--ignore-user-config",
        "--config",
        'approval_policy="never"',
        "--config",
        hook_config,
        "--enable",
        "hooks",
        "--dangerously-bypass-hook-trust",
        "--model",
        model,
        "--output-schema",
        str(schema),
        "--output-last-message",
        str(last_message),
        "--config",
        f"mcp_servers.techdocs.command={json.dumps(args[0])}",
        "--config",
        f"mcp_servers.techdocs.args={json.dumps(args[1:])}",
        "--config",
        "mcp_servers.techdocs.required=true",
        "--config",
        'mcp_servers.techdocs.default_tools_approval_mode="approve"',
        "--config",
        "mcp_servers.techdocs.startup_timeout_sec=30",
        "--config",
        "mcp_servers.techdocs.tool_timeout_sec=30",
        prompt,
    ]
    environment = dict(os.environ)
    environment["KB_EVAL_MODE"] = mode
    environment["KB_EVAL_HOOK_STATE"] = str(hook_state)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=workspace,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        latency_seconds = time.perf_counter() - started
        events_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        parsed = json.loads(last_message.read_text(encoding="utf-8")) if completed.returncode == 0 else {}
        parse_error = None if isinstance(parsed, dict) else "Codex output was not an object"
    except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError) as error:
        latency_seconds = time.perf_counter() - started
        completed = None
        parsed = {}
        parse_error = str(error)
    sources = [str(value) for value in parsed.get("sources", [])] if isinstance(parsed, dict) else []
    mcp_events = _json_lines(mcp_trace)
    tool_sequence = [str(event.get("tool") or "") for event in mcp_events]
    evidence_sources = sorted(
        {
            str(source)
            for event in mcp_events
            for source in event.get("output_uris") or []
        }
    )
    return {
        "harness": "codex",
        "mode": mode,
        "requested_model": model,
        "actual_models": [model] if completed and completed.returncode == 0 else [],
        "prediction": str(parsed.get("answer") or "") if isinstance(parsed, dict) else "",
        "sources": sources,
        "latency_seconds": latency_seconds,
        "return_code": completed.returncode if completed else -1,
        "parse_error": parse_error,
        "usage": _codex_usage(completed.stdout if completed else ""),
        "tool_calls": len(mcp_events),
        "tool_sequence": tool_sequence,
        "evidence_output_hashes": [
            str(event.get("output_sha256") or "") for event in mcp_events
        ],
        "evidence_sources": evidence_sources,
        "workflow_compliant": workflow_compliant(mode, mcp_events),
        "model_steps": None,
        "raw_directory": str(raw_dir),
    }


def _write_dsh_patches(
    raw_dir: Path,
    mode: str,
    model: str,
    project_root: Path,
    python: Path,
    endpoint: str,
    mcp_trace: Path,
    result_limit: int,
    evidence_token_budget: int,
    fixed_query: str | None,
) -> tuple[Path, Path]:
    model_patch = raw_dir / "model.patch.yml"
    runtime_patch = raw_dir / "runtime.patch.yml"
    model_patch.write_text(
        "\n".join(
            [
                "- id: agent-default-model",
                "  config:",
                "    provider: openai",
                f"    model: {json.dumps(model)}",
                "",
                "- id: llm-pi-ai",
                "  config:",
                "    providers:",
                "      openai:",
                "        apiKeyEnv: OPENAI_API_KEY",
                "        models:",
                f"          - id: {json.dumps(model)}",
                "            contextWindow: 400000",
                "            maxTokens: 4096",
                "",
            ]
        ),
        encoding="utf-8",
    )
    args = _mcp_args(
        python,
        project_root,
        endpoint,
        mode,
        mcp_trace,
        result_limit,
        evidence_token_budget,
        fixed_query,
    )
    runtime_patch.write_text(
        "\n".join(
            [
                "- id: kbbench-techdocs-runtime",
                "  config:",
                "    adapter: mcp",
                f"    workflowMode: {mode}",
                f"    maxToolCalls: {1 if mode == 'composite' else 3}",
                "    restrictTools: true",
                "",
                "- insert:",
                "    - id: kbbench-shared-mcp",
                "      name: '@deepseek-ai/dsh-mcp-client'",
                "      config:",
                "        serverName: techdocs",
                "        transport: stdio",
                f"        command: {json.dumps(args[0])}",
                f"        args: {json.dumps(args[1:])}",
                "        env: {}",
                f"        cwd: {json.dumps(str(project_root))}",
                "        toolCallTimeoutMs: 30000",
                "        failOnStartupError: true",
                "        reconnect:",
                "          enabled: false",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return model_patch, runtime_patch


def run_dsh_case(
    question: dict[str, object],
    mode: str,
    endpoint: str,
    output_dir: Path,
    project_root: Path,
    python: Path,
    dsh_bin: Path,
    dsh_home: Path,
    plugin_dir: Path,
    model: str,
    timeout_seconds: int,
    result_limit: int,
    evidence_token_budget: int,
    openai_api_key: str | None,
) -> dict[str, object]:
    case_name = f"{question['id']}-{mode}-dsh"
    raw_dir = (output_dir / "raw" / case_name).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    mcp_trace = raw_dir / "mcp_trace.jsonl"
    stdout_path = raw_dir / "stdout.txt"
    stderr_path = raw_dir / "stderr.txt"
    model_patch, runtime_patch = _write_dsh_patches(
        raw_dir,
        mode,
        model,
        project_root,
        python,
        endpoint,
        mcp_trace,
        result_limit,
        evidence_token_budget,
        str(question["question"]) if mode == "composite" else None,
    )
    before_sessions = _dsh_session_files(dsh_home)
    command = [
        str(dsh_bin),
        "--profile",
        "headless",
        "--patch",
        str(model_patch.resolve()),
        "--patch",
        str((plugin_dir / "trial-kb-minimal.patch.yml").resolve()),
        "--patch",
        str(runtime_patch.resolve()),
        common_prompt(question),
    ]
    environment = dict(os.environ)
    environment["DSH_HOME"] = str(dsh_home)
    if openai_api_key:
        environment["OPENAI_API_KEY"] = openai_api_key
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=plugin_dir,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        latency_seconds = time.perf_counter() - started
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        parsed = parse_agent_json(completed.stdout) if completed.returncode == 0 else {}
        parse_error = None
    except (subprocess.TimeoutExpired, OSError, ValueError, json.JSONDecodeError) as error:
        latency_seconds = time.perf_counter() - started
        completed = None
        parsed = {}
        parse_error = str(error)
    new_sessions = sorted(
        _dsh_session_files(dsh_home) - before_sessions,
        key=lambda path: path.stat().st_mtime,
    )
    session_events = _read_dsh_session(new_sessions[-1]) if new_sessions else []
    trace = _dsh_trace(session_events)
    sources = [str(value) for value in parsed.get("sources", [])] if isinstance(parsed, dict) else []
    mcp_events = _json_lines(mcp_trace)
    tool_sequence = [str(event.get("tool") or "") for event in mcp_events]
    evidence_sources = sorted(
        {
            str(source)
            for event in mcp_events
            for source in event.get("output_uris") or []
        }
    )
    return {
        "harness": "dsh",
        "mode": mode,
        "requested_model": model,
        "actual_models": trace["actual_models"],
        "prediction": str(parsed.get("answer") or "") if isinstance(parsed, dict) else "",
        "sources": sources,
        "latency_seconds": latency_seconds,
        "return_code": completed.returncode if completed else -1,
        "parse_error": parse_error,
        "usage": trace["usage"],
        "tool_calls": len(mcp_events),
        "tool_sequence": tool_sequence,
        "evidence_output_hashes": [
            str(event.get("output_sha256") or "") for event in mcp_events
        ],
        "evidence_sources": evidence_sources,
        "workflow_compliant": workflow_compliant(mode, mcp_events),
        "model_steps": trace["model_steps"],
        "session_tool_sequence": trace["session_tool_sequence"],
        "session_file": str(new_sessions[-1]) if new_sessions else None,
        "raw_directory": str(raw_dir),
    }


def workflow_compliant(mode: str, mcp_events: list[dict[str, Any]]) -> bool:
    if any(event.get("error") for event in mcp_events):
        return False
    tools = [str(event.get("tool") or "") for event in mcp_events]
    if mode == "composite":
        return tools == ["techdocs_composite"]
    return (
        1 <= len(tools) <= 3
        and tools[0] == "techdocs_search"
        and set(tools).issubset({"techdocs_search", "techdocs_expand", "techdocs_fetch"})
    )


def summarize(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    metrics = (
        "exact_match",
        "f1",
        "citation_present",
        "citation_recall",
        "citation_all_support",
        "evidence_recall",
        "evidence_all_support",
        "latency_seconds",
        "tool_calls",
    )
    output: dict[str, dict[str, float]] = {}
    for mode in MODES:
        for harness in HARNESSES:
            selected = [row for row in rows if row["mode"] == mode and row["harness"] == harness]
            if not selected:
                continue
            key = f"{mode}/{harness}"
            output[key] = {
                metric: float(np.mean([float(row[metric]) for row in selected]))
                for metric in metrics
            }
            latencies = np.asarray([float(row["latency_seconds"]) for row in selected])
            output[key]["latency_p50_seconds"] = float(np.percentile(latencies, 50))
            output[key]["latency_p95_seconds"] = float(np.percentile(latencies, 95))
            for token_name in ("input_fresh", "input_cached", "input_total", "output", "total"):
                output[key][f"tokens_{token_name}"] = float(
                    np.mean(
                        [
                            float(
                                (row["usage"] or {}).get(
                                    token_name,
                                    (
                                        float((row["usage"] or {}).get("input_total", 0))
                                        + float((row["usage"] or {}).get("output", 0))
                                        if token_name == "total"
                                        else 0
                                    ),
                                )
                            )
                            for row in selected
                        ]
                    )
                )
            successful = sum(_row_succeeded(row) for row in selected)
            output[key]["successful"] = float(successful)
            output[key]["success_rate"] = successful / len(selected)
    return output


def _row_succeeded(row: dict[str, object]) -> bool:
    return (
        int(row.get("return_code") or 0) == 0
        and not row.get("parse_error")
        and bool(row.get("workflow_compliant", True))
    )


def composite_evidence_identical(rows: list[dict[str, object]]) -> bool:
    composite = [row for row in rows if row.get("mode") == "composite"]
    if not composite:
        return True
    keys = {(str(row["question_id"]), int(row["repeat"])) for row in composite}
    for key in keys:
        pair = [
            row
            for row in composite
            if (str(row["question_id"]), int(row["repeat"])) == key
        ]
        if {str(row.get("harness")) for row in pair} != set(HARNESSES):
            return False
        hashes = [tuple(row.get("evidence_output_hashes") or []) for row in pair]
        if not hashes[0] or len(set(hashes)) != 1:
            return False
    return True


def paired_differences(
    rows: list[dict[str, object]], bootstrap_samples: int = 2000
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    rng = np.random.default_rng(71)
    for mode in MODES:
        by_key = {
            (str(row["question_id"]), int(row["repeat"])): row
            for row in rows
            if row["mode"] == mode
            and row["harness"] == "codex"
            and _row_succeeded(row)
        }
        pairs = [
            (by_key[(str(row["question_id"]), int(row["repeat"]))], row)
            for row in rows
            if row["mode"] == mode
            and row["harness"] == "dsh"
            and _row_succeeded(row)
            and (str(row["question_id"]), int(row["repeat"])) in by_key
        ]
        if not pairs:
            continue
        mode_output: dict[str, object] = {"pairs": len(pairs), "direction": "DSH minus Codex"}
        for metric in (
            "exact_match",
            "f1",
            "citation_recall",
            "citation_all_support",
            "evidence_recall",
            "latency_seconds",
            "tool_calls",
        ):
            differences = np.asarray(
                [float(dsh[metric]) - float(codex[metric]) for codex, dsh in pairs],
                dtype=np.float64,
            )
            draws = np.asarray(
                [
                    float(rng.choice(differences, size=len(differences), replace=True).mean())
                    for _ in range(bootstrap_samples)
                ]
            )
            mode_output[metric] = {
                "mean": float(differences.mean()),
                "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            }
        for token_name in ("input_fresh", "input_total", "output", "total"):
            differences = np.asarray(
                [
                    float((dsh["usage"] or {}).get(token_name, 0))
                    - float((codex["usage"] or {}).get(token_name, 0))
                    for codex, dsh in pairs
                ],
                dtype=np.float64,
            )
            draws = np.asarray(
                [
                    float(rng.choice(differences, size=len(differences), replace=True).mean())
                    for _ in range(bootstrap_samples)
                ]
            )
            mode_output[f"tokens_{token_name}"] = {
                "mean": float(differences.mean()),
                "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            }
        output[mode] = mode_output
    return output


def estimate_dsh_api_cost(
    rows: list[dict[str, object]],
    input_per_million: float,
    cached_input_per_million: float,
    output_per_million: float,
) -> dict[str, float]:
    selected = [row for row in rows if row.get("harness") == "dsh"]
    fresh = sum(float((row.get("usage") or {}).get("input_fresh", 0)) for row in selected)
    cached = sum(float((row.get("usage") or {}).get("input_cached", 0)) for row in selected)
    output = sum(float((row.get("usage") or {}).get("output", 0)) for row in selected)
    estimated = (
        fresh * input_per_million
        + cached * cached_input_per_million
        + output * output_per_million
    ) / 1_000_000
    return {
        "input_fresh_tokens": fresh,
        "input_cached_tokens": cached,
        "output_tokens": output,
        "input_usd_per_million": input_per_million,
        "cached_input_usd_per_million": cached_input_per_million,
        "output_usd_per_million": output_per_million,
        "estimated_usd": estimated,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired Codex-skill/hooks versus DSH-plugin evaluation over one shared MCP KB."
    )
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--graph-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include")
    parser.add_argument("--question-format", choices=FORMATS, default="auto")
    parser.add_argument("--include-unanswerable", action="store_true")
    parser.add_argument("--github-repo", default="")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--modes", default="composite,primitive")
    parser.add_argument("--harnesses", default="codex,dsh")
    parser.add_argument("--codex-bin", type=Path, default=Path("codex"))
    parser.add_argument("--dsh-bin", type=Path, required=True)
    parser.add_argument("--dsh-home", type=Path, required=True)
    parser.add_argument("--plugin-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--codex-model", default="gpt-5.4-mini")
    parser.add_argument("--dsh-model", default="gpt-5.4-mini")
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--result-limit", type=int, default=8)
    parser.add_argument("--evidence-token-budget", type=int, default=2200)
    parser.add_argument("--minimum-claim-pairs", type=int, default=30)
    parser.add_argument("--dsh-input-price-per-million", type=float)
    parser.add_argument("--dsh-cached-input-price-per-million", type=float)
    parser.add_argument("--dsh-output-price-per-million", type=float)
    parser.add_argument("--env-file", type=Path)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    questions, dataset_manifest = load_harness_questions(
        args.questions,
        question_format=args.question_format,
        answerable_only=not args.include_unanswerable,
    )
    sample = deterministic_sample(questions, args.sample_size)
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    harnesses = [value.strip() for value in args.harnesses.split(",") if value.strip()]
    if not set(modes).issubset(MODES) or not set(harnesses).issubset(HARNESSES):
        raise ValueError("unknown mode or harness")
    api_key = _load_key(args.env_file, "OPENAI_API_KEY")
    include_glob = args.include or (
        "*.txt" if dataset_manifest["question_format"] == "techqa" else "*.md"
    )
    service = TechdocsIndexService(
        repository_root=args.repo,
        graph_repository_root=args.graph_repo,
        github_repository=args.github_repo,
        include_glob=include_glob,
        dense_dimensions=64,
        route_count=32,
        default_method="hybrid",
        title_repeats=1,
    )
    title_to_id = {
        html.unescape(document.title): document.doc_id for document in service.corpus.documents
    }
    for question in sample:
        if "relevant_ids" in question:
            missing = sorted(set(question.get("relevant_ids") or []) - set(service.documents))
        else:
            missing = [
                str(title)
                for title in question.get("supporting", [])
                if html.unescape(str(title)) not in title_to_id
            ]
        if missing:
            raise ValueError(
                f"question {question['id']} has scoring documents absent from the corpus: "
                + ", ".join(missing[:10])
            )
    rows: list[dict[str, object]] = []
    with local_service(service) as endpoint:
        for mode in modes:
            for repeat in range(args.repeats):
                for question in sample:
                    order = list(harnesses)
                    if len(order) == 2 and int(
                        hashlib.sha256(
                            f"{mode}\0{repeat}\0{question['id']}".encode("utf-8")
                        ).hexdigest(),
                        16,
                    ) % 2:
                        order.reverse()
                    relevant_ids = (
                        [str(value) for value in question.get("relevant_ids", [])]
                        if "relevant_ids" in question
                        else [
                            title_to_id[html.unescape(str(title))]
                            for title in question.get("supporting", [])
                        ]
                    )
                    for harness in order:
                        print(
                            f"[{mode}] repeat {repeat + 1}: {harness} {question['id']}",
                            flush=True,
                        )
                        shared = {
                            "question": question,
                            "mode": mode,
                            "endpoint": endpoint,
                            "output_dir": args.output,
                            "project_root": project_root,
                            "python": args.python.resolve(),
                            "timeout_seconds": args.timeout_seconds,
                            "result_limit": args.result_limit,
                            "evidence_token_budget": args.evidence_token_budget,
                        }
                        if harness == "codex":
                            row = run_codex_case(
                                codex_bin=args.codex_bin,
                                model=args.codex_model,
                                **shared,
                            )
                        else:
                            row = run_dsh_case(
                                dsh_bin=args.dsh_bin.resolve(),
                                dsh_home=args.dsh_home.resolve(),
                                plugin_dir=args.plugin_dir.resolve(),
                                model=args.dsh_model,
                                openai_api_key=api_key,
                                **shared,
                            )
                        row.update(
                            {
                                "question_id": str(question["id"]),
                                "question": str(question["question"]),
                                "question_type": str(question.get("type") or ""),
                                "repeat": repeat,
                                "gold_answer": str(question["answer"]),
                                "relevant_ids": relevant_ids,
                                **score_prediction(
                                    str(row["prediction"]),
                                    list(row["sources"]),
                                    str(question["answer"]),
                                    relevant_ids,
                                ),
                                **score_evidence(
                                    list(row.get("evidence_sources") or []),
                                    relevant_ids,
                                ),
                            }
                        )
                        rows.append(row)
                        (args.output / "partial_rows.json").write_text(
                            json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8",
                        )

    same_requested_model = args.codex_model == args.dsh_model
    paired_observations_per_mode = len(sample) * args.repeats
    adequate_paired_observations = paired_observations_per_mode >= args.minimum_claim_pairs
    identical_composite_evidence = composite_evidence_identical(rows)
    complete_pairs = all(
        any(
            row["question_id"] == str(question["id"])
            and row["mode"] == mode
            and row["harness"] == harness
            and int(row["repeat"]) == repeat
            and _row_succeeded(row)
            for row in rows
        )
        for question in sample
        for mode in modes
        for harness in harnesses
        for repeat in range(args.repeats)
    )
    report = {
        "benchmark": "Shared-MCP Codex versus DSH harness evaluation",
        "claim_boundary": (
            "The KB service, MCP server, tool schemas, corpus, retrieval settings, and evidence "
            "limits are identical. Composite mode measures harness/answer overhead around one "
            "fixed tool; primitive mode measures model-plus-harness orchestration."
        ),
        "selection": "lowest SHA-256(question id), independent of outcomes",
        "dataset": {
            **dataset_manifest,
            "include_glob": include_glob,
            "selected_question_ids": [str(question["id"]) for question in sample],
            "selected_qrel_documents": len(
                {
                    str(value)
                    for question in sample
                    for value in question.get("relevant_ids", [])
                }
            ),
        },
        "sample_size": len(sample),
        "repeats": args.repeats,
        "modes": modes,
        "harnesses": harnesses,
        "models": {"codex": args.codex_model, "dsh": args.dsh_model},
        "minimum_claim_pairs_per_mode": args.minimum_claim_pairs,
        "paired_observations_per_mode": paired_observations_per_mode,
        "harness_claim_eligible": bool(
            set(harnesses) == set(HARNESSES)
            and same_requested_model
            and complete_pairs
            and adequate_paired_observations
            and identical_composite_evidence
        ),
        "eligibility_checks": {
            "same_requested_model_identifier": same_requested_model,
            "same_mcp_server": True,
            "same_backend_process": True,
            "same_tool_schemas_per_mode": True,
            "same_evidence_limits": True,
            "identical_composite_evidence": identical_composite_evidence,
            "workflow_compliance_required": True,
            "complete_successful_pairs": complete_pairs,
            "minimum_paired_observations_met": adequate_paired_observations,
        },
        "token_accounting": (
            "tokens_total is input_total plus output. input_total includes cached input. Codex "
            "fresh input is input_tokens minus its cached subset; DSH reports fresh inputTokens "
            "and cacheReadTokens separately."
        ),
        "service": service.health(),
        "summary": summarize(rows),
        "paired_dsh_minus_codex": paired_differences(rows),
        "rows": rows,
    }
    prices = (
        args.dsh_input_price_per_million,
        args.dsh_cached_input_price_per_million,
        args.dsh_output_price_per_million,
    )
    if all(value is not None for value in prices):
        report["dsh_api_cost_estimate"] = {
            "basis": "Observed token counts multiplied by caller-supplied prices; excludes Codex ChatGPT-account usage.",
            **estimate_dsh_api_cost(rows, *[float(value) for value in prices]),
        }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
