from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen


EVIDENCE_URI_PATTERN = re.compile(r"viking://resources/techdocs/[^\s\"'<>\\]+")


def load_routing_rules(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("defaultDecision") not in {"retrieve", "skip"}:
        raise ValueError("routing rules defaultDecision must be retrieve or skip")
    return {
        "version": int(value.get("version") or 1),
        "defaultDecision": str(value["defaultDecision"]),
        "maxQueryCharacters": max(200, min(8000, int(value.get("maxQueryCharacters") or 1600))),
        "retrievePatterns": _compile(value.get("retrievePatterns")),
        "codeOnlyPatterns": _compile(value.get("codeOnlyPatterns")),
        "graphPatterns": _compile(value.get("graphPatterns")),
    }


def routing_decision(task: str, rules: dict[str, Any]) -> dict[str, Any]:
    text = str(task or "").strip()
    query = _issue_text(text)[: int(rules["maxQueryCharacters"])].strip()
    retrieve_matches = _matching_sources(text, rules["retrievePatterns"])
    code_only_matches = _matching_sources(text, rules["codeOnlyPatterns"])
    graph_matches = _matching_sources(text, rules["graphPatterns"])
    if retrieve_matches:
        decision = "retrieve"
        reason = f"documentation-signal:{retrieve_matches[0]}"
    elif code_only_matches:
        decision = "skip"
        reason = f"code-signal:{code_only_matches[0]}"
    else:
        decision = str(rules["defaultDecision"])
        reason = f"default:{decision}"
    return {
        "decision": decision,
        "query": query,
        "allowGraph": decision == "retrieve" and bool(graph_matches),
        "confidence": 0.9 if retrieve_matches or code_only_matches else 0.55,
        "reason": reason,
        "ruleVersion": int(rules["version"]),
    }


def run_hook(payload: dict[str, Any], environment: dict[str, str]) -> dict[str, Any] | None:
    trace_path = Path(environment["KB_EVAL_KB_TRACE"])
    rules = load_routing_rules(Path(environment["KB_ROUTING_RULES"]))
    prompt = str(payload.get("prompt") or "")
    route = routing_decision(prompt, rules)
    _record(trace_path, {"event": "route", "harness": "codex", **route})
    if route["decision"] != "retrieve" or not route["query"]:
        return None

    endpoint = _local_endpoint(environment["KB_ENDPOINT"])
    started = time.perf_counter()
    try:
        result = _post(
            f"{endpoint}/v1/search",
            {
                "query": route["query"],
                "result_limit": _bounded_int(environment.get("KB_RESULT_LIMIT"), 1, 30, 8),
                "evidence_token_budget": _bounded_int(
                    environment.get("KB_EVIDENCE_TOKEN_BUDGET"), 200, 12000, 2200
                ),
                "graph": {"enabled": bool(route["allowGraph"])},
            },
        )
        evidence = str(result.get("evidenceText") or "No technical-document evidence found.")
        _record(
            trace_path,
            {
                "event": "retrieval",
                "activation": "automatic-consumer",
                "provider": "repo-local",
                "query": route["query"],
                "allow_graph": bool(route["allowGraph"]),
                "latency_ms": (time.perf_counter() - started) * 1000,
                "output_sha256": hashlib.sha256(evidence.encode("utf-8")).hexdigest(),
                "output_characters": len(evidence),
                "output_uris": sorted(set(EVIDENCE_URI_PATTERN.findall(evidence))),
                "error": None,
                "backend_trace": result.get("trace"),
            },
        )
    except Exception as error:  # Hook failure must not block the coding agent.
        _record(
            trace_path,
            {
                "event": "retrieval",
                "activation": "automatic-consumer",
                "provider": "repo-local",
                "query": route["query"],
                "allow_graph": bool(route["allowGraph"]),
                "latency_ms": (time.perf_counter() - started) * 1000,
                "error": f"{type(error).__name__}: {error}",
            },
        )
        return None

    context = "\n\n".join(
        [
            "<techdocs-context>",
            "This is bounded supplementary evidence selected before the coding step. "
            "Inspect code and run tests normally; ignore passages that are not relevant.",
            evidence,
            "</techdocs-context>",
        ]
    )
    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }
    }


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise ValueError("hook input must be an object")
        output = run_hook(payload, dict(os.environ))
        if output is not None:
            sys.stdout.write(json.dumps(output, separators=(",", ":")))
        return 0
    except Exception as error:  # Invalid hook configuration also fails open.
        trace = os.environ.get("KB_EVAL_KB_TRACE")
        if trace:
            _record(Path(trace), {"event": "hook_error", "harness": "codex", "error": str(error)})
        return 0


def _compile(values: object) -> list[tuple[str, re.Pattern[str]]]:
    if not isinstance(values, list):
        raise ValueError("routing rule patterns must be arrays")
    return [(str(source), re.compile(str(source), re.IGNORECASE)) for source in values]


def _matching_sources(text: str, patterns: list[tuple[str, re.Pattern[str]]]) -> list[str]:
    return [source for source, pattern in patterns if pattern.search(text)]


def _issue_text(task: str) -> str:
    match = re.search(r"\nIssue:\s*\n", task, re.IGNORECASE)
    value = task[match.end() :] if match else task
    return re.sub(r"\s+", " ", value)


def _local_endpoint(value: str) -> str:
    parsed = urlparse(str(value).rstrip("/"))
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("KB_ENDPOINT must be a local HTTP endpoint")
    return parsed.geturl().rstrip("/")


def _post(url: str, body: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:  # noqa: S310 - endpoint is validated local.
        payload = json.loads(response.read())
    result = payload.get("result", payload)
    if not isinstance(result, dict):
        raise ValueError("KB service returned a non-object result")
    return result


def _bounded_int(value: object, minimum: int, maximum: int, fallback: int) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        return fallback
    return max(minimum, min(maximum, parsed))


def _record(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_unix": time.time(), **event}, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
