"""Real DSH rollout runner used by the GitHub Docs SkillOpt adapter."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml

from .scorer import GitHubDocsSourceResolver, score_ranked_sources
from .validation import SkillCandidateError, validate_skill_candidate


ARM_SKILL_PLUGIN_IDS = {
    "fs": "kbbench-skill-github-docs-fs",
    "hybrid": "kbbench-skill-github-docs-hybrid",
    "neo4j": "kbbench-skill-github-docs-neo4j",
}

ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def parse_agent_json(output: str) -> dict[str, object]:
    """Parse the DSH final response, tolerating a surrounding code fence."""

    cleaned = ANSI_PATTERN.sub("", output).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE
        )
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("agent output is not a JSON object")
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


def _trace_summary(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    fresh = cached = output = 0
    models: set[str] = set()
    tool_names: list[str] = []
    model_steps: set[tuple[int, int]] = set()
    skill_loaded = False
    for event in events:
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
        elif event_type == "tool/call":
            name = str(data.get("name") or "")
            tool_names.append(name)
            skill_loaded = skill_loaded or name == "skill"
        elif event_type == "step/start":
            model_steps.add((int(data.get("turn") or 0), int(data.get("step") or 0)))
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
        "model_steps": len(model_steps),
        "skill_loaded": skill_loaded,
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

    uri_pattern = re.compile(r"viking://resources/techdocs(?:/[A-Za-z0-9_./-]+)?")
    path_pattern = re.compile(r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.md")
    sources: list[str] = []
    seen: set[str] = set()
    for event in events:
        if event.get("type") != "tool/result":
            continue
        text = json.dumps(event.get("data") or {}, ensure_ascii=False)
        for match in [*uri_pattern.findall(text), *path_pattern.findall(text)]:
            value = match.rstrip(".,:;)")
            if value not in seen:
                sources.append(value)
                seen.add(value)
    return sources


def build_prompt(item: dict[str, Any], arm: str) -> str:
    return (
        f"GITHUB DOCS DSH RETRIEVAL EVAL {item['id']} ({arm}). "
        "Load and follow the installed GitHub Docs retrieval skill before searching. "
        "Use only the pinned local documentation corpus and the tools available in this profile. "
        "Return exactly one JSON object and no Markdown, with this schema: "
        '{"answer":"concise evidence-grounded answer or NOT FOUND",'
        '"sources":["ordered canonical doc ID or repository .md path"]}. '
        "List at most 10 distinct sources in decreasing evidence relevance. Do not invent a source.\n"
        f"Question:\n{item['question']}"
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
        ):
            if not path.exists():
                raise FileNotFoundError(path)


class DshCommandRunner:
    """Launch the pinned DSH headless profile for one benchmark question."""

    def __init__(
        self,
        config: DshCommandConfig,
        trace_provider: Callable[[], list[dict[str, Any]]] | None = None,
    ) -> None:
        config.validate()
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
        trace = _trace_summary(session_events)
        backend_events = (
            self.trace_provider()[backend_before:] if self.trace_provider else []
        )
        try:
            parsed = parse_agent_json(stdout) if return_code == 0 else {}
            parse_error = ""
        except (ValueError, json.JSONDecodeError) as error:
            parsed = {}
            parse_error = str(error)
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
            "session_file": str(session_path) if session_path else None,
            "conversation": _compact_conversation(session_events),
            "visible_sources": _visible_sources(session_events),
            "backend_events": backend_events,
            "provider_mode": "custom_gateway" if gateway_patch else "default",
            **trace,
        }


Runner = Callable[[dict[str, Any], Path, Path], dict[str, Any]]


def _invalid_result(
    item: dict[str, Any], error: Exception, prediction_dir: Path, skill_content: str
) -> dict[str, Any]:
    item_dir = prediction_dir / str(item["id"])
    item_dir.mkdir(parents=True, exist_ok=True)
    conversation = [
        {
            "type": "validation/error",
            "content": str(error),
        }
    ]
    (item_dir / "conversation.json").write_text(
        json.dumps(conversation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (item_dir / "target_system_prompt.txt").write_text(skill_content, encoding="utf-8")
    return {
        "id": str(item["id"]),
        "hard": 0,
        "soft": 0.0,
        "question": item["question"],
        "task_description": item["question"],
        "task_type": item["task_type"],
        "predicted_answer": "",
        "ranked_ids": [],
        "agent_ok": False,
        "fail_reason": f"invalid_skill_candidate: {error}",
        "target_system_prompt": skill_content,
        "target_user_prompt": build_prompt(item, str(item.get("arm") or "unknown")),
        "reference_text": "Candidate rejected before rollout; no qrels were exposed.",
        "n_turns": 0,
    }


def run_batch(
    *,
    items: list[dict[str, Any]],
    skill_content: str,
    out_root: str,
    arm: str,
    resolver: GitHubDocsSourceResolver,
    runner: Runner,
    expected_name: str,
    leakage_markers: dict[str, set[str]],
) -> list[dict[str, Any]]:
    """Run real DSH episodes and persist the trajectories SkillOpt reflects on."""

    root = Path(out_root).resolve()
    prediction_dir = root / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    skill_hash = hashlib.sha256(skill_content.encode()).hexdigest()
    candidate_dir = root / "candidate_skills" / skill_hash
    candidate_dir.mkdir(parents=True, exist_ok=True)
    skill_path = candidate_dir / "skill.md"
    skill_path.write_text(skill_content, encoding="utf-8")

    try:
        validate_skill_candidate(
            skill_content,
            arm=arm,
            expected_name=expected_name,
            leakage_markers=leakage_markers,
        )
        validation_error: SkillCandidateError | None = None
    except SkillCandidateError as error:
        validation_error = error

    results: list[dict[str, Any]] = []
    for item in items:
        if validation_error:
            results.append(
                _invalid_result(item, validation_error, prediction_dir, skill_content)
            )
            continue
        item_dir = prediction_dir / str(item["id"])
        item_dir.mkdir(parents=True, exist_ok=True)
        execution = runner(item, skill_path, item_dir)
        visible_sources = execution.get("visible_sources") or []
        if execution.get("backend_events"):
            visible_sources = [
                doc_id
                for event in execution["backend_events"]
                for doc_id in event.get("ranked_ids") or []
            ]
        visible_ranked_ids = resolver.resolve_ranked(visible_sources)
        ranked_ids = resolver.resolve_ranked(execution.get("sources") or [])
        metrics = score_ranked_sources(ranked_ids, item["qrel_ids"])
        agent_ok = (
            execution.get("return_code") == 0
            and not execution.get("parse_error")
            and bool(execution.get("skill_loaded"))
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
        (item_dir / "target_system_prompt.txt").write_text(
            skill_content, encoding="utf-8"
        )
        user_prompt = build_prompt(item, arm)
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
        result = {
            "id": str(item["id"]),
            "hard": int(metrics["hard"]),
            "soft": float(metrics["soft"]),
            "question": item["question"],
            "task_description": item["question"],
            "task_type": item["task_type"],
            "intent_category": item.get("intent_category"),
            "evidence_structure": item.get("evidence_structure"),
            "qrel_count": int(item.get("qrel_count") or len(set(item["qrel_ids"]))),
            "predicted_answer": str(execution.get("answer") or ""),
            "ranked_ids": ranked_ids,
            "citation_ranked_ids": ranked_ids,
            "visible_ranked_ids": visible_ranked_ids,
            "visible_sources": list(visible_sources),
            "agent_sources": list(execution.get("sources") or []),
            "agent_ok": agent_ok,
            "fail_reason": str(
                execution.get("execution_error")
                or execution.get("parse_error")
                or ("skill_not_loaded" if not execution.get("skill_loaded") else "")
            ),
            "target_system_prompt": skill_content,
            "target_user_prompt": user_prompt,
            "reference_text": reference_text,
            "n_turns": int(execution.get("model_steps") or 0),
            "latency_seconds": float(execution.get("latency_seconds") or 0.0),
            "usage": execution.get("usage") or {},
            "tool_sequence": execution.get("tool_sequence") or [],
            "actual_models": execution.get("actual_models") or [],
            "session_file": execution.get("session_file"),
            **{key: float(value) for key, value in metrics.items()},
        }
        (item_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        results.append(result)

    (root / "rollouts.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return results
