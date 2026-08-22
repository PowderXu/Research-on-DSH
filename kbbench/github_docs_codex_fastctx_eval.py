"""Run a real Codex + FastCtx MCP retrieval arm on frozen GitHub Docs questions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable

from .skillopt_github_docs.scorer import GitHubDocsSourceResolver, score_ranked_sources


DEFAULT_QUESTION_IDS = ("108045", "122713", "180308", "26749", "57244")
FASTCTX_TOOLS = ("grep", "glob", "read")
MARKDOWN_PATH = re.compile(r"(?:[A-Za-z]:)?/?(?:[A-Za-z0-9_.@+-]+/)+[A-Za-z0-9_.@+ -]+\.md")


def _normalize_item(raw: dict[str, Any]) -> dict[str, Any]:
    item_id = str(raw.get("question_id") or raw.get("id") or "").strip()
    question = str(raw.get("query") or raw.get("question") or "").strip()
    qrels = [str(value) for value in raw.get("qrel_ids") or [] if str(value)]
    if not item_id or not question or not qrels:
        raise ValueError("GitHub Docs item requires question_id, query, and qrel_ids")
    return {
        **raw,
        "id": item_id,
        "question": question,
        "qrel_ids": qrels,
        "task_type": str(raw.get("evidence_category") or "github_docs"),
        "intent_category": str(raw.get("intent_category") or "unknown"),
        "evidence_structure": str(raw.get("evidence_structure") or "unknown"),
        "qrel_count": int(raw.get("qrel_count") or len(set(qrels))),
    }


def _load_items(split_root: Path, question_ids: Iterable[str]) -> list[dict[str, Any]]:
    wanted = list(dict.fromkeys(str(value) for value in question_ids))
    by_id: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        path = split_root / split / "items.json"
        for raw in json.loads(path.read_text(encoding="utf-8")):
            item = _normalize_item(raw)
            if item["id"] in wanted:
                by_id[item["id"]] = item
    missing = [item_id for item_id in wanted if item_id not in by_id]
    if missing:
        raise ValueError(f"question IDs not found in frozen split: {missing}")
    return [by_id[item_id] for item_id in wanted]


def _prompt(item: dict[str, Any]) -> str:
    return (
        f"Use $github-docs-fastctx.\n\n"
        f"GITHUB DOCS CODEX FASTCTX RETRIEVAL EVAL {item['id']}. "
        "Load and follow the installed GitHub Docs retrieval skill before searching. "
        "Use only the pinned local documentation corpus. Use FastCtx MCP grep, glob, and read "
        "for every filesystem discovery and document read; do not use shell commands. "
        "Return exactly one JSON object and no Markdown, with this schema: "
        '{"answer":"concise evidence-grounded answer or NOT FOUND",'
        '"sources":["ordered canonical content/... .md path"]}. '
        "List at most 10 distinct sources in decreasing evidence relevance. Do not invent a source.\n"
        f"Question:\n{item['question']}"
    )


def _usage(events_text: str) -> dict[str, int]:
    usage: dict[str, int] = {}
    for line in events_text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
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
    }


def _text_blocks(value: object) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "text" and isinstance(child, str):
                yield child
            else:
                yield from _text_blocks(child)
    elif isinstance(value, list):
        for child in value:
            yield from _text_blocks(child)


def _trace(events_text: str) -> dict[str, Any]:
    tool_sequence: list[str] = []
    visible_sources: list[str] = []
    seen_sources: set[str] = set()
    fastctx_calls: list[dict[str, Any]] = []
    shell_calls: list[str] = []
    for line in events_text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "item.completed":
            continue
        item = event.get("item") or {}
        if item.get("type") == "mcp_tool_call":
            server = str(item.get("server") or "")
            tool = str(item.get("tool") or "")
            if server != "fastctx" or tool not in FASTCTX_TOOLS:
                continue
            tool_sequence.append(tool)
            texts = list(_text_blocks(item.get("result")))
            for text in texts:
                for match in MARKDOWN_PATH.findall(text):
                    source = match.strip().lstrip("./")
                    if source not in seen_sources:
                        visible_sources.append(source)
                        seen_sources.add(source)
            fastctx_calls.append(
                {
                    "tool": tool,
                    "arguments": item.get("arguments") or {},
                    "output_characters": sum(len(text) for text in texts),
                    "error": item.get("error"),
                }
            )
        elif item.get("type") == "command_execution":
            shell_calls.append(str(item.get("command") or ""))
    return {
        "tool_sequence": tool_sequence,
        "visible_sources": visible_sources,
        "fastctx_calls": fastctx_calls,
        "shell_calls": shell_calls,
    }


def _install_skill(workspace: Path, skill_path: Path) -> Path:
    installed = workspace / ".agents/skills/github-docs-fastctx/SKILL.md"
    installed.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(skill_path, installed)
    return installed


def run_case(
    item: dict[str, Any],
    *,
    workspace: Path,
    corpus_jsonl: Path,
    skill_path: Path,
    codex_bin: Path,
    fastctx_bin: Path,
    schema: Path,
    output_root: Path,
    model: str,
    reasoning_effort: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    case_dir = output_root / "raw" / str(item["id"])
    case_dir.mkdir(parents=True, exist_ok=True)
    events_path = case_dir / "events.jsonl"
    stderr_path = case_dir / "stderr.txt"
    last_message = case_dir / "last_message.json"
    prompt_path = case_dir / "prompt.txt"
    prompt = _prompt(item)
    skill_content = skill_path.read_text(encoding="utf-8")
    prompt_path.write_text(prompt, encoding="utf-8")
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
        "--ignore-rules",
        "--config",
        'approval_policy="never"',
        "--config",
        f"model_reasoning_effort={json.dumps(reasoning_effort)}",
        "--config",
        f"developer_instructions={json.dumps(skill_content)}",
        "--disable",
        "apps",
        "--disable",
        "browser_use",
        "--disable",
        "computer_use",
        "--model",
        model,
        "--output-schema",
        str(schema),
        "--output-last-message",
        str(last_message),
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
        "mcp_servers.fastctx.tool_timeout_sec=60",
        prompt,
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        latency = time.perf_counter() - started
        stdout = completed.stdout
        stderr = completed.stderr
        return_code = completed.returncode
        execution_error = ""
    except (OSError, subprocess.TimeoutExpired) as error:
        latency = time.perf_counter() - started
        stdout = str(getattr(error, "stdout", "") or "")
        stderr = str(getattr(error, "stderr", "") or "")
        return_code = -1
        execution_error = str(error)
    events_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")

    try:
        parsed = json.loads(last_message.read_text(encoding="utf-8")) if return_code == 0 else {}
        parse_error = "" if isinstance(parsed, dict) else "Codex output was not an object"
    except (OSError, json.JSONDecodeError) as error:
        parsed = {}
        parse_error = str(error)
    sources = [str(value) for value in parsed.get("sources", [])] if isinstance(parsed, dict) else []
    trace = _trace(stdout)
    resolver = GitHubDocsSourceResolver(corpus_jsonl)
    visible_ranked_ids = resolver.resolve_ranked(trace["visible_sources"])
    ranked_ids = resolver.resolve_ranked(sources)
    citation_ranked_ids = ranked_ids
    metrics = score_ranked_sources(ranked_ids, item["qrel_ids"])
    citation_metrics = score_ranked_sources(citation_ranked_ids, item["qrel_ids"])
    compliant = bool(trace["fastctx_calls"]) and not trace["shell_calls"] and not any(
        call.get("error") for call in trace["fastctx_calls"]
    )
    agent_ok = return_code == 0 and not parse_error and compliant
    if not agent_ok:
        metrics["hard"] = 0.0
        metrics["soft"] = 0.0
    return {
        "id": str(item["id"]),
        "question": item["question"],
        "task_type": item["task_type"],
        "intent_category": item.get("intent_category"),
        "evidence_structure": item.get("evidence_structure"),
        "qrel_count": int(item.get("qrel_count") or len(set(item["qrel_ids"]))),
        "predicted_answer": str(parsed.get("answer") or "") if isinstance(parsed, dict) else "",
        "ranked_ids": ranked_ids,
        "citation_ranked_ids": citation_ranked_ids,
        "visible_ranked_ids": visible_ranked_ids,
        "visible_sources": trace["visible_sources"],
        "agent_sources": sources,
        "agent_ok": agent_ok,
        "fail_reason": execution_error or parse_error or ("workflow_noncompliant" if not compliant else ""),
        "latency_seconds": latency,
        "usage": _usage(stdout),
        "tool_sequence": trace["tool_sequence"],
        "actual_models": [model] if return_code == 0 else [],
        "skill_invoked": True,
        "skill_sha256": hashlib.sha256(skill_path.read_bytes()).hexdigest(),
        "fastctx_version": "0.2.5",
        "fastctx_calls": trace["fastctx_calls"],
        "shell_calls": trace["shell_calls"],
        "return_code": return_code,
        **{key: float(value) for key, value in metrics.items()},
        **{
            f"citation_{key}": float(value)
            for key, value in citation_metrics.items()
            if key not in {"hard", "soft"}
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--split-root", type=Path, default=root / "evaluation/skillopt/github_docs_v2_split"
    )
    parser.add_argument(
        "--workspace", type=Path, default=root / "data/github-docs"
    )
    parser.add_argument(
        "--corpus-jsonl", type=Path, default=root / "evaluation/github_docs_v2/corpus.jsonl"
    )
    parser.add_argument(
        "--skill",
        type=Path,
        default=root / "codex-techdocs-plugin/skills/github-docs-fastctx/SKILL.md",
    )
    parser.add_argument("--codex-bin", type=Path, default=Path("codex"))
    parser.add_argument(
        "--fastctx-bin", type=Path, default=root / ".fastctx-runtime/node_modules/.bin/fastctx"
    )
    parser.add_argument(
        "--schema", type=Path, default=root / "evaluation/harness/answer_schema.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--question-ids", default=",".join(DEFAULT_QUESTION_IDS))
    args = parser.parse_args()

    for required in (
        args.split_root,
        args.workspace,
        args.corpus_jsonl,
        args.skill,
        args.codex_bin,
        args.fastctx_bin,
        args.schema,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    installed_skill = _install_skill(args.workspace, args.skill)
    items = _load_items(args.split_root, args.question_ids.split(","))
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] Codex + FastCtx: {item['id']}", flush=True)
        row = run_case(
            item,
            workspace=args.workspace.resolve(),
            corpus_jsonl=args.corpus_jsonl.resolve(),
            skill_path=args.skill.resolve(),
            codex_bin=args.codex_bin.resolve(),
            fastctx_bin=args.fastctx_bin.resolve(),
            schema=args.schema.resolve(),
            output_root=args.output.resolve(),
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            timeout_seconds=args.timeout_seconds,
        )
        rows.append(row)
        (args.output / "partial_rollouts.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"  ok={row['agent_ok']} Hit@10={row['hit_at_10']:.3f} "
            f"nDCG@10={row['ndcg_at_10']:.3f} latency={row['latency_seconds']:.2f}s",
            flush=True,
        )
    (args.output / "rollouts.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "arm": "codex-fastctx",
        "evaluation_layer": "Codex agent + translated matched skill + FastCtx MCP",
        "question_ids": [item["id"] for item in items],
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "codex_cli": subprocess.run(
            [str(args.codex_bin.resolve()), "--version"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "fastctx_version": subprocess.run(
            [str(args.fastctx_bin.resolve()), "--version"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "skill": str(args.skill.resolve()),
        "installed_skill": str(installed_skill.resolve()),
        "skill_sha256": hashlib.sha256(args.skill.read_bytes()).hexdigest(),
        "corpus": str(args.workspace.resolve()),
        "corpus_jsonl": str(args.corpus_jsonl.resolve()),
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
