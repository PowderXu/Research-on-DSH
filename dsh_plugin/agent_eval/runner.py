"""Run matched real-model DSH episodes over a frozen benchmark split."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml

from kbbench.scoring import GitHubDocsSourceResolver, score_ranked_sources
from dsh_plugin.backend.corpus_workspace import validate_corpus_workspace
from dsh_plugin.backend.retrieval_policy import retrieval_contract

from .credentials import load_openai_key_from_configured_env


ARM_SKILL_PLUGIN_IDS = {
    "fs": "kbbench-skill-docsqa-fs",
    "hybrid": "kbbench-skill-docsqa-hybrid",
    "neo4j": "kbbench-skill-docsqa-neo4j",
}
ARM_SKILL_NAMES = {
    "fs": "docsqa-fs",
    "hybrid": "docsqa-hybrid",
    "neo4j": "docsqa-neo4j",
}

ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _validate_agent_json(
    output: str,
) -> tuple[dict[str, object], list[str], list[dict[str, object]]]:
    """Validate the exact JSON response contract without repairing model output."""

    cleaned = ANSI_PATTERN.sub("", output).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as error:
        return {}, [f"invalid_json:{error.msg}"], []
    if not isinstance(value, dict):
        return {}, ["output_not_object"], []

    errors: list[str] = []
    invalid_sources: list[dict[str, object]] = []
    expected_keys = {"answer", "sources"}
    missing = sorted(expected_keys - set(value))
    unexpected = sorted(set(value) - expected_keys)
    if missing:
        errors.append(f"missing_keys:{','.join(missing)}")
    if unexpected:
        errors.append(f"unexpected_keys:{','.join(unexpected)}")

    answer = value.get("answer")
    if not isinstance(answer, str):
        errors.append("answer_not_string")
        answer = ""

    raw_sources = value.get("sources")
    sources: list[str] = []
    if not isinstance(raw_sources, list):
        errors.append("sources_not_array")
        raw_sources = []
    if len(raw_sources) > 10:
        errors.append("too_many_sources")
    seen: set[str] = set()
    for index, source in enumerate(raw_sources):
        if not isinstance(source, str):
            invalid_sources.append(
                {
                    "index": index,
                    "reason": "not_string",
                    "value": repr(source)[:200],
                }
            )
            continue
        if not source.strip():
            invalid_sources.append(
                {"index": index, "reason": "empty", "value": source}
            )
            continue
        if source in seen:
            invalid_sources.append(
                {"index": index, "reason": "duplicate", "value": source[:500]}
            )
            continue
        seen.add(source)
        if len(sources) < 10:
            sources.append(source)
    if invalid_sources:
        errors.append("invalid_sources")
    return {"answer": answer, "sources": sources}, errors, invalid_sources


def parse_agent_json(output: str) -> dict[str, object]:
    """Parse and strictly validate the DSH final response."""

    value, errors, _invalid_sources = _validate_agent_json(output)
    if errors:
        raise ValueError(";".join(errors))
    return value


def _session_files(dsh_home: Path) -> set[Path]:
    return set(dsh_home.glob("sessions/**/session.jsonl.zstd"))


def _read_session(path: Path) -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["zstdcat", str(path)], capture_output=True, text=True, check=False
    )
    if completed.returncode:
        return []
    events: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _parse_tool_arguments(value: object) -> tuple[object, str]:
    if isinstance(value, dict):
        return value, ""
    if not isinstance(value, str):
        return value, "arguments_not_json_object"
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        return value, f"invalid_arguments_json:{error.msg}"
    if not isinstance(parsed, dict):
        return parsed, "arguments_not_json_object"
    return parsed, ""


def _text_from_content(value: object) -> str:
    texts: list[str] = []

    def visit(node: object) -> None:
        if isinstance(node, dict):
            if node.get("type") == "text" and isinstance(node.get("text"), str):
                texts.append(str(node["text"]))
            for child in node.values():
                if isinstance(child, (dict, list)):
                    visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)
    return "\n".join(texts)


def _compact_result_meta(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    compact: dict[str, object] = {}
    for key in ("shape", "total", "truncated", "path", "offset", "lang"):
        candidate = value.get(key)
        if isinstance(candidate, (str, int, float, bool)) or candidate is None:
            compact[key] = candidate
    for key in ("files", "paths", "lines"):
        candidate = value.get(key)
        if isinstance(candidate, list):
            compact[f"{key}_count"] = len(candidate)
    return compact


def _normalize_fs_path(value: object) -> str:
    path = str(value or "").strip().replace("\\", "/")
    path = re.sub(r"^<path>|</path>$", "", path).strip()
    path = re.sub(r"/+", "/", path)
    while path.startswith("./"):
        path = path[2:]
    return path.rstrip("/")


def _discovered_fs_paths(meta: object, result_text: str) -> list[str]:
    """Extract Markdown paths returned by glob/grep without guessing filenames."""

    candidates: list[object] = []
    if isinstance(meta, dict):
        paths = meta.get("paths")
        if isinstance(paths, list):
            candidates.extend(paths)
        files = meta.get("files")
        if isinstance(files, list):
            candidates.extend(
                row.get("path")
                for row in files
                if isinstance(row, dict) and row.get("path")
            )
    candidates.extend(
        match.group(1)
        for match in re.finditer(
            r"(?<![A-Za-z0-9_.-])((?:\.?\.?/)?(?:[A-Za-z0-9_.-]+/)+"
            r"[A-Za-z0-9_.-]+\.mdx?)",
            result_text,
            flags=re.IGNORECASE,
        )
    )
    for line in result_text.splitlines():
        candidate = line.strip()
        if re.fullmatch(r"(?:\.?\.?/)?[A-Za-z0-9_.-]+\.mdx?", candidate, re.I):
            candidates.append(candidate)

    paths: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_fs_path(candidate)
        if not normalized.casefold().endswith((".md", ".mdx")):
            continue
        if normalized and normalized not in seen:
            seen.add(normalized)
            paths.append(normalized)
    return paths


def _read_fs_path(outcome: Mapping[str, Any]) -> str:
    arguments = outcome.get("arguments")
    if not isinstance(arguments, dict):
        return ""
    for key in ("file_path", "path"):
        if isinstance(arguments.get(key), str):
            return _normalize_fs_path(arguments[key])
    return ""


def _same_fs_document(discovered: str, read_path: str) -> bool:
    discovered_path = _normalize_fs_path(discovered)
    target_path = _normalize_fs_path(read_path)
    if not discovered_path or not target_path:
        return False
    return (
        discovered_path == target_path
        or discovered_path.endswith("/" + target_path)
        or target_path.endswith("/" + discovered_path)
    )


def _tool_outcomes(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match tool results to calls by ID and retain auditable arguments/status."""

    event_list = list(events)
    outcomes: list[dict[str, Any]] = []
    by_call_id: dict[str, dict[str, Any]] = {}
    for event in event_list:
        if event.get("type") != "tool/call":
            continue
        data = event.get("data") or {}
        call_id = str(data.get("callId") or "")
        arguments, argument_error = _parse_tool_arguments(data.get("arguments"))
        outcome = {
            "call_id": call_id,
            "name": str(data.get("name") or ""),
            "arguments": arguments,
            "arguments_error": argument_error,
            "result_matched": False,
            "is_error": None,
            "successful": False,
            "result_character_count": 0,
            "result_preview": "",
            "result_meta": {},
            "discovered_path_count": 0,
            "discovered_paths": [],
            "_all_discovered_paths": [],
        }
        outcomes.append(outcome)
        if call_id:
            by_call_id[call_id] = outcome

    for event in event_list:
        if event.get("type") != "tool/result":
            continue
        data = event.get("data") or {}
        message = data.get("message") or {}
        source = message.get("source") or {}
        source_call_id = str(source.get("callId") or "")
        blocks = [
            block
            for block in message.get("content") or []
            if isinstance(block, dict) and block.get("type") == "tool-result"
        ]
        for block in blocks or [{}]:
            call_id = str(block.get("toolCallId") or source_call_id)
            outcome = by_call_id.get(call_id)
            if outcome is None:
                continue
            content = block.get("content") or []
            result_text = _text_from_content(content)
            is_error = block.get("isError")
            discovered_paths = (
                _discovered_fs_paths(data.get("meta"), result_text)
                if outcome.get("name") in {"glob", "grep"}
                else []
            )
            outcome.update(
                {
                    "result_matched": True,
                    "is_error": is_error if isinstance(is_error, bool) else None,
                    "successful": is_error is False,
                    "result_character_count": len(result_text),
                    "result_preview": result_text[:2_000],
                    "result_meta": _compact_result_meta(data.get("meta")),
                    "discovered_path_count": len(discovered_paths),
                    "discovered_paths": discovered_paths[:50],
                    "_all_discovered_paths": discovered_paths,
                }
            )
    return outcomes


def _outcome_retrieved_evidence(outcome: Mapping[str, Any]) -> bool:
    if not outcome.get("successful") or outcome.get("arguments_error"):
        return False
    name = str(outcome.get("name") or "")
    text = str(outcome.get("result_preview") or "")
    meta = outcome.get("result_meta") or {}
    if name in {"glob", "grep"}:
        return bool(
            int(outcome.get("discovered_path_count") or 0) > 0
            and (int(meta.get("total") or 0) > 0 or bool(text.strip()))
        )
    if name == "read":
        return bool(int(meta.get("lines_count") or 0) > 0 or "<path>" in text)
    if name == "docsqa_search":
        return "<docsqa-evidence" in text and "URI:" in text
    return False


def _fs_discovery_read_links(
    outcomes: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Prove that an earlier discovery result supplied each successful read."""

    links: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for read_index, read_outcome in enumerate(outcomes):
        if read_outcome.get("name") != "read" or not _outcome_retrieved_evidence(
            read_outcome
        ):
            continue
        read_path = _read_fs_path(read_outcome)
        if not read_path:
            continue
        for discovery in outcomes[:read_index]:
            if discovery.get("name") not in {"glob", "grep"}:
                continue
            if not _outcome_retrieved_evidence(discovery):
                continue
            for discovered_path in discovery.get("_all_discovered_paths") or []:
                if not _same_fs_document(str(discovered_path), read_path):
                    continue
                key = (str(discovery.get("call_id") or ""), str(read_outcome.get("call_id") or ""))
                if key in seen:
                    continue
                seen.add(key)
                links.append(
                    {
                        "discovery_call_id": key[0],
                        "discovery_tool": str(discovery.get("name") or ""),
                        "discovered_path": str(discovered_path),
                        "read_call_id": key[1],
                        "read_path": read_path,
                    }
                )
    return links


def _trace_summary(
    events: Iterable[dict[str, Any]], expected_arm: str | None = None
) -> dict[str, Any]:
    fresh = cached = output = 0
    models: set[str] = set()
    model_steps: set[tuple[int, int]] = set()
    event_list = list(events)
    for event in event_list:
        event_type = event.get("type")
        data = event.get("data") or {}
        if event_type == "assistant/message":
            message = data.get("message") or {}
            usage = data.get("usage") or message.get("usage") or {}
            fresh += int(usage.get("inputTokens") or 0)
            cached += int(usage.get("cacheReadTokens") or 0)
            output += int(usage.get("outputTokens") or 0)
            source = message.get("source") or {}
            if source.get("model"):
                models.add(str(source["model"]))
        elif event_type == "step/start":
            model_steps.add((int(data.get("turn") or 0), int(data.get("step") or 0)))

    tool_outcomes = _tool_outcomes(event_list)
    tool_names = [str(outcome["name"]) for outcome in tool_outcomes]
    skill_loaded = any(
        outcome["name"] == "skill" and outcome["successful"]
        for outcome in tool_outcomes
    )
    expected_skill_name = ARM_SKILL_NAMES.get(str(expected_arm or ""), "")
    expected_skill_loaded = any(
        outcome["name"] == "skill"
        and outcome["successful"]
        and not outcome["arguments_error"]
        and isinstance(outcome["arguments"], dict)
        and outcome["arguments"].get("name") == expected_skill_name
        and f'<skill_content name="{expected_skill_name}">' in outcome["result_preview"]
        for outcome in tool_outcomes
    ) if expected_skill_name else skill_loaded
    retrieved_tools = [
        str(outcome["name"])
        for outcome in tool_outcomes
        if _outcome_retrieved_evidence(outcome)
    ]
    fs_retrieval_links = _fs_discovery_read_links(tool_outcomes)
    if expected_arm == "fs":
        retrieval_succeeded = bool(fs_retrieval_links)
    elif expected_arm in {"hybrid", "neo4j"}:
        retrieval_succeeded = "docsqa_search" in retrieved_tools
    else:
        retrieval_succeeded = bool(retrieved_tools)
    for outcome in tool_outcomes:
        outcome.pop("_all_discovered_paths", None)
    return {
        "usage": {
            "input_fresh": fresh,
            "input_cached": cached,
            "input_total": fresh + cached,
            "output": output,
            "total": fresh + cached + output,
        },
        "actual_models": sorted(models),
        "tool_sequence": tool_names,
        "tool_outcomes": tool_outcomes,
        "model_steps": len(model_steps),
        "skill_loaded": skill_loaded,
        "expected_skill_name": expected_skill_name,
        "expected_skill_loaded": expected_skill_loaded,
        "retrieval_succeeded": retrieval_succeeded,
        "successful_retrieval_tools": retrieved_tools,
        "fs_retrieval_links": fs_retrieval_links,
    }


def _compact_conversation(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for event in events:
        event_type = str(event.get("type") or "")
        if event_type not in {
            "user/message",
            "assistant/message",
            "tool/call",
            "tool/result",
        }:
            continue
        data = event.get("data") or {}
        encoded = json.dumps(data, ensure_ascii=False)
        if len(encoded) > 16_000:
            encoded = encoded[:16_000] + "...[truncated]"
        kept.append({"type": event_type, "data": encoded})
    return kept


def _visible_sources(events: Iterable[dict[str, Any]]) -> list[str]:
    """Extract source identifiers in the order they were visible to the model."""

    # The project namespace is part of the canonical ID (for example,
    # ``github-docs::/actions/...``); the two colons must not be truncated.
    uri_pattern = re.compile(
        r"viking://resources/docsqa/[A-Za-z0-9_.:/%#@?=&+~-]+"
    )
    path_pattern = re.compile(r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.md")
    sources: list[str] = []
    seen: set[str] = set()
    for event in events:
        if event.get("type") != "tool/result":
            continue
        text = _text_from_content(event.get("data") or {})
        for match in [*uri_pattern.findall(text), *path_pattern.findall(text)]:
            value = match.rstrip(".,:;)")
            if value not in seen:
                sources.append(value)
                seen.add(value)
    return sources


def build_prompt(item: dict[str, Any], arm: str) -> str:
    dataset = str(item.get("dataset") or "pinned documentation")
    return (
        f"DOCSQA DSH RETRIEVAL EVAL {item['id']} (corpus={dataset}). "
        "Load and follow the installed documentation retrieval skill before searching. "
        "Use only the pinned local documentation corpus and the tools available in this profile. "
        "Return exactly one JSON object and no Markdown, with this schema: "
        '{"answer":"concise evidence-grounded answer or NOT FOUND",'
        '"sources":["ordered canonical doc ID or repository .md path"]}. '
        "List at most 10 distinct sources in decreasing evidence relevance. Do not invent a source.\n"
        f"Documentation project: {dataset}.\nQuestion:\n{item['question']}"
    )


def _dsh_gateway_url(environment: Mapping[str, str]) -> str:
    return str(
        environment.get("DSH_OPENAI_BASE_URL")
        or environment.get("OPENAI_BASE_URL")
        or ""
    ).strip()


def _write_dsh_gateway_patch(
    case_dir: Path,
    base_url: str,
    model_patch: Path,
) -> Path | None:
    """Derive a complete provider patch so DSH's replace semantics keep the model."""

    if not base_url:
        return None
    document = yaml.safe_load(model_patch.read_text(encoding="utf-8"))
    if not isinstance(document, list):
        raise ValueError(f"DSH model patch must be a list: {model_patch}")
    provider = next(
        (
            entry
            for entry in document
            if isinstance(entry, dict) and entry.get("id") == "llm-pi-ai"
        ),
        None,
    )
    if not isinstance(provider, dict):
        raise ValueError(f"DSH model patch has no llm-pi-ai entry: {model_patch}")
    providers = provider.get("config", {}).get("providers", {})
    openai = providers.get("openai") if isinstance(providers, dict) else None
    if not isinstance(openai, dict):
        raise ValueError(f"DSH model patch has no OpenAI provider: {model_patch}")
    openai["baseURL"] = base_url
    patch = case_dir / "provider-model.patch.yml"
    patch.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return patch


def _dsh_invocation(
    config: "DshCommandConfig",
    item: dict[str, Any],
    candidate_patch: Path,
    gateway_patch: Path | None,
) -> list[str]:
    return [
        str(config.dsh_binary.resolve()),
        "--profile",
        config.profile,
        "--patch",
        str((gateway_patch or config.model_patch).resolve()),
        "--patch",
        str(config.common_patch.resolve()),
        "--patch",
        str(config.arm_patch.resolve()),
        "--patch",
        str(candidate_patch.resolve()),
        build_prompt(item, config.arm),
    ]


@dataclass(frozen=True)
class DshCommandConfig:
    arm: str
    dsh_binary: Path
    dsh_home: Path
    workspace: Path
    model_patch: Path
    common_patch: Path
    arm_patch: Path
    corpus_path: Path
    timeout_seconds: int = 180
    profile: str = "headless"

    def validate(self) -> None:
        if self.arm not in ARM_SKILL_PLUGIN_IDS:
            raise ValueError(f"unknown DSH arm: {self.arm}")
        for path in (
            self.dsh_binary,
            self.dsh_home,
            self.workspace,
            self.model_patch,
            self.common_patch,
            self.arm_patch,
            self.corpus_path,
        ):
            if not path.exists():
                raise FileNotFoundError(path)


def preflight_corpus_workspace(config: DshCommandConfig) -> dict[str, Any]:
    """Verify corpus bytes and actual official tools without a model or QA labels."""
    corpus = [
        json.loads(line)
        for line in config.corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    snapshot = validate_corpus_workspace(corpus, config.workspace)
    script = Path(__file__).resolve().parents[1] / "scripts/preflight_fs.ts"
    try:
        completed = subprocess.run(
            ["node", "--experimental-strip-types", str(script)],
            input=json.dumps({
                "workspace": str(config.workspace.resolve()),
                "documents": [
                    {"path": row["source_path"], "text": row["rendered_text"]}
                    for row in corpus
                ],
            }),
            cwd=script.parent.parent,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(f"official FS workspace preflight could not run: {error}") from error
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()[-4000:]
        raise ValueError(f"official FS workspace preflight failed: {detail}")
    try:
        tool_check = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("official FS workspace preflight returned invalid JSON") from error
    if not isinstance(tool_check, dict) or tool_check.get("ok") is not True:
        raise ValueError("official FS workspace preflight did not confirm success")
    count = snapshot["document_count"]
    for field in ("expected_document_count", "searchable_document_count", "glob_document_count"):
        if tool_check.get(field) != count:
            raise ValueError(f"official FS workspace preflight inventory mismatch: {field}")
    if not isinstance(tool_check.get("probe_count"), int) or tool_check["probe_count"] < 1:
        raise ValueError("official FS workspace preflight did not verify search/read probes")
    # Recheck after probing to catch content changes during the startup check.
    if validate_corpus_workspace(corpus, config.workspace) != snapshot:
        raise ValueError("corpus workspace changed during preflight")
    return {"protocol": "exact-corpus-official-fs-v1", **snapshot, "official_tools": tool_check}


class DshCommandRunner:
    """Launch the pinned DSH headless profile for one benchmark question."""

    def __init__(
        self,
        config: DshCommandConfig,
        trace_provider: Callable[[], list[dict[str, Any]]] | None = None,
    ) -> None:
        config.validate()
        self.preflight = preflight_corpus_workspace(config)
        self.config = config
        self.trace_provider = trace_provider

    def __call__(
        self,
        item: dict[str, Any],
        skill_path: Path,
        case_dir: Path,
    ) -> dict[str, Any]:
        case_dir.mkdir(parents=True, exist_ok=True)
        candidate_patch = case_dir / "candidate-skill.patch.yml"
        candidate_patch.write_text(
            "\n".join(
                (
                    f"- id: {ARM_SKILL_PLUGIN_IDS[self.config.arm]}",
                    "  disabled: false",
                    "  config:",
                    f"    skillPath: {json.dumps(str(skill_path.resolve()))}",
                    "",
                )
            ),
            encoding="utf-8",
        )
        environment = dict(os.environ)
        environment["DSH_HOME"] = str(self.config.dsh_home.resolve())
        environment["DSH_PERMISSION_MODE"] = "read-only"
        gateway_patch = _write_dsh_gateway_patch(
            case_dir,
            _dsh_gateway_url(environment),
            self.config.model_patch,
        )
        command = _dsh_invocation(
            self.config, item, candidate_patch, gateway_patch
        )
        before = _session_files(self.config.dsh_home)
        backend_before = len(self.trace_provider()) if self.trace_provider else 0
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=self.config.workspace,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.config.timeout_seconds,
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

        (case_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
        (case_dir / "stderr.txt").write_text(stderr, encoding="utf-8")
        new_sessions = sorted(
            _session_files(self.config.dsh_home) - before,
            key=lambda path: path.stat().st_mtime,
        )
        session_path = new_sessions[-1] if new_sessions else None
        session_events = _read_session(session_path) if session_path else []
        trace = _trace_summary(session_events, expected_arm=self.config.arm)
        backend_events = (
            self.trace_provider()[backend_before:] if self.trace_provider else []
        )
        if return_code == 0:
            parsed, validation_errors, invalid_sources = _validate_agent_json(stdout)
        else:
            parsed, validation_errors, invalid_sources = {}, [], []
        parse_error = ";".join(validation_errors)
        sources = parsed.get("sources") if isinstance(parsed, dict) else []
        if not isinstance(sources, list):
            sources = []
        return {
            "answer": str(parsed.get("answer") or "") if isinstance(parsed, dict) else "",
            "sources": [str(value) for value in sources],
            "latency_seconds": latency,
            "return_code": return_code,
            "execution_error": execution_error,
            "parse_error": parse_error,
            "output_validation_errors": validation_errors,
            "invalid_sources": invalid_sources,
            "session_file": str(session_path) if session_path else None,
            "conversation": _compact_conversation(session_events),
            "visible_sources": _visible_sources(session_events),
            "backend_events": backend_events,
            "provider_mode": "custom_gateway" if gateway_patch else "default",
            **trace,
        }


Runner = Callable[[dict[str, Any], Path, Path], dict[str, Any]]


def _logical_relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"runtime file is outside logical root: {path}") from error


def _files_sha256(paths: Iterable[Path], *, logical_root: Path) -> str:
    """Hash file bytes under stable logical paths, never host-absolute paths."""

    digest = hashlib.sha256()
    resolved = sorted(
        (value.resolve() for value in paths),
        key=lambda path: _logical_relative_path(path, logical_root),
    )
    for path in resolved:
        digest.update(_logical_relative_path(path, logical_root).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _file_identity(path: Path, logical_path: str) -> dict[str, str]:
    resolved = path.resolve()
    return {
        "path": logical_path,
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def _repo_file_bundle_identity(
    paths: Iterable[Path], project_root: Path
) -> dict[str, object]:
    files = sorted(
        (path.resolve() for path in paths),
        key=lambda path: _logical_relative_path(path, project_root),
    )
    return {
        "sha256": _files_sha256(files, logical_root=project_root),
        "files": [
            _file_identity(path, _logical_relative_path(path, project_root))
            for path in files
        ],
    }


def _provider_identity(
    model_patch: Path, environment: Mapping[str, str]
) -> dict[str, str]:
    document = yaml.safe_load(model_patch.read_text(encoding="utf-8"))
    default = next(
        (
            entry.get("config") or {}
            for entry in document or []
            if isinstance(entry, dict) and entry.get("id") == "agent-default-model"
        ),
        {},
    )
    gateway_url = _dsh_gateway_url(environment)
    return {
        "provider": str(default.get("provider") or ""),
        "model": str(default.get("model") or ""),
        "adapter": "llm-pi-ai",
        "mode": "custom_gateway" if gateway_url else "default",
        "gateway_identity": hashlib.sha256(gateway_url.encode("utf-8")).hexdigest()
        if gateway_url
        else "",
    }


def _installed_dependency_versions(names: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _stable_service_identity(health: Mapping[str, Any]) -> dict[str, Any]:
    """Select deterministic service-health fields for resume fingerprints."""

    return {
        "status": health.get("status"),
        "arm": health.get("arm"),
        "revision": health.get("revision"),
        "documents": health.get("documents"),
        "chunks": health.get("chunks"),
        "method": health.get("method"),
        "retrieval_contract": health.get("retrievalContract"),
        "graph_snapshot": health.get("graphSnapshot"),
    }


def run_batch(
    *,
    items: list[dict[str, Any]],
    skill_path: Path,
    out_root: str,
    arm: str,
    resolver: GitHubDocsSourceResolver,
    runner: Runner,
    resume: bool = False,
    evaluation_contract: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run real DSH episodes and persist trajectories plus deterministic scores."""

    root = Path(out_root).resolve()
    prediction_dir = root / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_path.resolve()
    if not skill_path.is_file():
        raise FileNotFoundError(skill_path)
    skill_sha256 = hashlib.sha256(skill_path.read_bytes()).hexdigest()

    results: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        user_prompt = build_prompt(item, arm)
        evaluation_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "arm": arm,
                    "question_id": str(item["id"]),
                    "prompt": user_prompt,
                    "qrel_ids": sorted(set(map(str, item["qrel_ids"]))),
                    "task_type": str(item["task_type"]),
                    "evidence_structure": item.get("evidence_structure"),
                    "skill_sha256": skill_sha256,
                    "evaluation_contract": dict(evaluation_contract or {}),
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        item_dir = prediction_dir / str(item["id"])
        item_dir.mkdir(parents=True, exist_ok=True)
        result_path = item_dir / "result.json"
        if resume and result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if str(result.get("id")) != str(item["id"]):
                raise ValueError(f"resumed result ID mismatch: {result_path}")
            if (
                bool(result.get("agent_ok"))
                and result.get("evaluation_fingerprint") == evaluation_fingerprint
            ):
                results.append(result)
                print(f"agent progress: {index}/{len(items)} (resumed)", flush=True)
                continue
            print(f"agent progress: {index}/{len(items)} (retrying stale/failed)", flush=True)
        execution = runner(item, skill_path, item_dir)
        visible_sources = execution.get("visible_sources") or []
        if execution.get("backend_events"):
            visible_sources = [
                doc_id
                for event in execution["backend_events"]
                for doc_id in event.get("ranked_ids") or []
            ]
        agent_sources = list(execution.get("sources") or [])
        visible_ranked_ids = resolver.resolve_ranked(visible_sources, limit=10)
        ranked_ids = resolver.resolve_ranked(agent_sources, limit=10)
        unresolved_sources = [
            source for source in agent_sources if resolver.resolve(str(source)) is None
        ]
        unresolved_visible_sources = [
            source for source in visible_sources if resolver.resolve(str(source)) is None
        ]
        metrics = score_ranked_sources(ranked_ids, item["qrel_ids"])
        expected_skill_loaded = bool(execution.get("expected_skill_loaded"))
        retrieval_succeeded = bool(execution.get("retrieval_succeeded"))
        invalid_sources = list(execution.get("invalid_sources") or [])
        output_validation_errors = list(
            execution.get("output_validation_errors") or []
        )
        agent_ok = (
            execution.get("return_code") == 0
            and not execution.get("parse_error")
            and expected_skill_loaded
            and retrieval_succeeded
            and not invalid_sources
            and not unresolved_sources
        )
        if not agent_ok:
            metrics["hard"] = 0.0
            metrics["soft"] = 0.0

        conversation = execution.get("conversation") or [
            {"type": "assistant/output", "content": execution.get("answer", "")}
        ]
        (item_dir / "conversation.json").write_text(
            json.dumps(conversation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (item_dir / "target_user_prompt.txt").write_text(user_prompt, encoding="utf-8")
        reference_text = (
            "Hidden deterministic retrieval score (document identities withheld): "
            f"Hit@10={metrics['hit_at_10']:.3f}, "
            f"Recall@10={metrics['recall_at_10']:.3f}, "
            f"nDCG@10={metrics['ndcg_at_10']:.3f}; "
            f"evidence_category={item['task_type']}; "
            f"evidence_structure={item.get('evidence_structure', 'unknown')}; "
            f"qrel_count={item.get('qrel_count', len(set(item['qrel_ids'])))}."
        )
        failure_reasons: list[str] = []
        if execution.get("return_code") != 0:
            failure_reasons.append("execution_failed")
        if execution.get("parse_error"):
            failure_reasons.append("invalid_output_json")
        if not expected_skill_loaded:
            failure_reasons.append("expected_skill_not_loaded")
        if not retrieval_succeeded:
            failure_reasons.append("retrieval_not_successful")
        if invalid_sources:
            failure_reasons.append("invalid_sources")
        if unresolved_sources:
            failure_reasons.append("unresolved_sources")

        result = {
            "id": str(item["id"]),
            "arm": arm,
            "hard": int(metrics["hard"]),
            "soft": float(metrics["soft"]),
            "question": item["question"],
            "task_description": item["question"],
            "task_type": item["task_type"],
            "intent_category": item.get("intent_category"),
            "project": str(
                item.get("dataset") or str(item["id"]).split("::", 1)[0]
            ),
            "qrel_ids": list(map(str, item["qrel_ids"])),
            "evidence_structure": item.get("evidence_structure"),
            "qrel_count": int(item.get("qrel_count") or len(set(item["qrel_ids"]))),
            "question_has_image": bool(item.get("question_images")),
            "question_image_group": "image"
            if item.get("question_images")
            else "text_only",
            "predicted_answer": str(execution.get("answer") or ""),
            "ranked_ids": ranked_ids,
            "citation_ranked_ids": ranked_ids,
            "visible_ranked_ids": visible_ranked_ids,
            "visible_sources": list(visible_sources),
            "agent_sources": agent_sources,
            "invalid_sources": invalid_sources,
            "unresolved_sources": unresolved_sources,
            "unresolved_visible_sources": unresolved_visible_sources,
            "output_validation_errors": output_validation_errors,
            "agent_ok": agent_ok,
            "fail_reason": ";".join(failure_reasons),
            "execution_error": str(execution.get("execution_error") or ""),
            "parse_error": str(execution.get("parse_error") or ""),
            "skill_path": str(skill_path),
            "skill_sha256": skill_sha256,
            "evaluation_contract": dict(evaluation_contract or {}),
            "evaluation_fingerprint": evaluation_fingerprint,
            "target_user_prompt": user_prompt,
            "reference_text": reference_text,
            "n_turns": int(execution.get("model_steps") or 0),
            "latency_seconds": float(execution.get("latency_seconds") or 0.0),
            "usage": execution.get("usage") or {},
            "tool_sequence": execution.get("tool_sequence") or [],
            "tool_outcomes": execution.get("tool_outcomes") or [],
            "expected_skill_name": execution.get("expected_skill_name")
            or ARM_SKILL_NAMES[arm],
            "expected_skill_loaded": expected_skill_loaded,
            "retrieval_succeeded": retrieval_succeeded,
            "successful_retrieval_tools": execution.get(
                "successful_retrieval_tools"
            )
            or [],
            "fs_retrieval_links": execution.get("fs_retrieval_links") or [],
            "backend_events": execution.get("backend_events") or [],
            "provider_mode": str(execution.get("provider_mode") or "unknown"),
            "actual_models": execution.get("actual_models") or [],
            "session_file": execution.get("session_file"),
            **{key: float(value) for key, value in metrics.items()},
        }
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        results.append(result)
        print(f"agent progress: {index}/{len(items)}", flush=True)

    (root / "rollouts.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return results


def load_split_items(
    split_path: Path,
    *,
    limit: int | None = None,
    question_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"expected a JSON array in {split_path}")
    items: list[dict[str, Any]] = []
    for row in rows:
        question_id = str(row["question_id"])
        if question_ids and question_id not in question_ids:
            continue
        items.append(
            {
                **row,
                "id": question_id,
                "question": str(row["query"]),
                "task_type": str(row["evidence_category"]),
            }
        )
        if limit and len(items) >= limit:
            break
    if question_ids:
        missing = sorted(question_ids - {item["id"] for item in items})
        if missing:
            raise ValueError(f"question IDs not present in {split_path}: {', '.join(missing)}")
    return items


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    from dsh_plugin.backend.data_paths import arm_data_layout

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARM_SKILL_PLUGIN_IDS), required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--question-id", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="Prepared corpus/splits; defaults to this arm's local data/corpus",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Documentation workspace; defaults to this arm's local data/documents",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Local retrieval indexes; defaults to this arm's data/indexes",
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--skill", type=Path)
    parser.add_argument("--dsh-binary", type=Path, default=project_root / "dsh_plugin/node_modules/.bin/dsh")
    parser.add_argument("--dsh-home", type=Path, default=project_root / "dsh_plugin/dsh_home")
    parser.add_argument("--model-patch", type=Path, default=project_root / "dsh_plugin/harness/model-openai.patch.yml")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed per-question result.json files in the output directory",
    )
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--neo4j-uri", default=os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7688"))
    parser.add_argument("--neo4j-username", default=os.environ.get("NEO4J_USERNAME", "neo4j"))
    parser.add_argument("--neo4j-database", default=os.environ.get("NEO4J_DATABASE", "neo4j"))
    args = parser.parse_args()
    layout = arm_data_layout(args.arm, args.data_root).ensure()
    dataset_dir = args.dataset_dir or layout.corpus
    workspace = args.workspace or layout.documents
    cache_dir = args.cache_dir or layout.indexes

    split_path = dataset_dir / "splits" / f"{args.split}.json"
    corpus_path = dataset_dir / "corpus.jsonl"
    items = load_split_items(
        split_path,
        limit=args.limit,
        question_ids=set(args.question_id),
    )
    skill_path = args.skill or project_root / f"dsh_plugin/plugin/skills/{args.arm}/initial_skill.md"
    config = DshCommandConfig(
        arm=args.arm,
        dsh_binary=args.dsh_binary,
        dsh_home=args.dsh_home,
        workspace=workspace,
        model_patch=args.model_patch,
        common_patch=project_root / "dsh_plugin/harness/docsqa_dsh_common.patch.yml",
        arm_patch=project_root / f"dsh_plugin/harness/docsqa_{args.arm}_system.patch.yml",
        corpus_path=corpus_path,
        timeout_seconds=args.timeout_seconds,
    )
    command_runner = DshCommandRunner(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "preflight.json").write_text(
        json.dumps(command_runner.preflight, indent=2) + "\n", encoding="utf-8"
    )
    if not load_openai_key_from_configured_env():
        raise SystemExit(
            "OPENAI_API_KEY is unset; set it directly or point KBBENCH_OPENAI_ENV_FILE to an env file"
        )

    service_host = None
    trace_provider = None
    service_identity: dict[str, Any] = {
        "status": "not_applicable",
        "arm": args.arm,
        "retrieval_contract": retrieval_contract(args.arm),
        "graph_snapshot": None,
    }
    if args.arm in {"hybrid", "neo4j"}:
        from dsh_plugin.backend.service import GitHubDocsPluginService, GitHubDocsServiceHost

        service = GitHubDocsPluginService(
            dataset_dir=dataset_dir,
            cache_dir=cache_dir,
            arm=args.arm,
            embedding_model_name=args.embedding_model,
            device=args.device,
            local_files_only=not args.allow_model_download,
            neo4j_uri=args.neo4j_uri,
            neo4j_username=args.neo4j_username,
            neo4j_password=os.environ.get("NEO4J_PASSWORD", "secretgraph"),
            neo4j_database=args.neo4j_database,
            trace_path=layout.traces / "agent-eval.jsonl",
            data_root=layout.root,
        )
        service_host = GitHubDocsServiceHost(
            service, "127.0.0.1", 1935 if args.arm == "hybrid" else 1936
        )
        service_host.start()
        trace_provider = lambda: list(service.events)
        command_runner.trace_provider = trace_provider
        service_identity = _stable_service_identity(service.health())

    try:
        compiled_plugin_files = sorted(
            (project_root / "dsh_plugin/plugin/lib").rglob("*.js")
        )
        if not compiled_plugin_files:
            raise FileNotFoundError(
                "compiled DSH plugin runtime is missing; run npm run build in dsh_plugin/plugin"
            )
        runtime_files = [
            Path(__file__),
            project_root / "dsh_plugin/backend/corpus_workspace.py",
            project_root / "dsh_plugin/backend/prepare_plugin_data.py",
            project_root / "dsh_plugin/scripts/preflight_fs.ts",
            project_root / "dsh_plugin/backend/http_contract.py",
            project_root / "dsh_plugin/backend/retrieval_policy.py",
            project_root / "dsh_plugin/backend/service.py",
            project_root / "dsh_plugin/plugin/package.json",
            project_root / "evaluation/kbbench/plugin_eval.py",
            project_root / "evaluation/kbbench/retrieval.py",
            config.model_patch,
            config.common_patch,
            config.arm_patch,
            *compiled_plugin_files,
        ]
        dependency_files = [
            project_root / "dsh_plugin/package.json",
            project_root / "dsh_plugin/package-lock.json",
            project_root / "dsh_plugin/plugin/package.json",
            project_root / "dsh_plugin/plugin/package-lock.json",
            project_root / "evaluation/pyproject.toml",
            project_root / "evaluation/requirements.txt",
            project_root / "evaluation/requirements-graph.txt",
        ]
        manifest_path = dataset_dir / "manifest.json"
        corpus_identity: dict[str, object] = {
            "corpus": _file_identity(corpus_path, "dataset/corpus.jsonl"),
            "searchable_workspace": {
                field: command_runner.preflight[field]
                for field in ("protocol", "document_count", "corpus_sha256", "workspace_sha256", "ignore_sha256")
            },
        }
        if manifest_path.is_file():
            corpus_identity["manifest"] = _file_identity(
                manifest_path, "dataset/manifest.json"
            )
        provider_identity = _provider_identity(config.model_patch, os.environ)
        rows = run_batch(
            items=items,
            skill_path=skill_path,
            out_root=str(args.output_dir),
            arm=args.arm,
            resolver=GitHubDocsSourceResolver(corpus_path),
            runner=command_runner,
            resume=args.resume,
            evaluation_contract={
                "retrieval": retrieval_contract(args.arm),
                "backend_service": service_identity,
                "embedding_model": args.embedding_model,
                "device": args.device,
                "corpus": corpus_identity,
                "split": {
                    "name": args.split,
                    "file": _file_identity(
                        split_path, f"dataset/splits/{args.split}.json"
                    ),
                    "questions": len(items),
                },
                "scoring": {
                    "protocol": "binary-qrels-ir-at-1-5-10-20-v1",
                    "citation_limit": 10,
                    "implementation": _file_identity(
                        project_root / "evaluation/kbbench/scoring.py",
                        "evaluation/kbbench/scoring.py",
                    ),
                },
                "dependencies": {
                    "locked_files": _repo_file_bundle_identity(
                        dependency_files, project_root
                    ),
                    "python_packages": _installed_dependency_versions(
                        ("neo4j", "numpy", "sentence-transformers", "PyYAML")
                    ),
                },
                "provider": provider_identity,
                "runtime": _repo_file_bundle_identity(runtime_files, project_root),
                "compiled_plugin": _repo_file_bundle_identity(
                    compiled_plugin_files, project_root
                ),
            },
        )
    finally:
        if service_host is not None:
            service_host.close()
    print(json.dumps({"arm": args.arm, "questions": len(rows), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
