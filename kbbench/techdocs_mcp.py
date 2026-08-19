from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SERVER_NAME = "kbbench-techdocs"
SERVER_VERSION = "0.2.0"
COMPOSITE_TOOL = "techdocs_composite"
SEARCH_TOOL = "techdocs_search"
EXPAND_TOOL = "techdocs_expand"
FETCH_TOOL = "techdocs_fetch"
EVIDENCE_URI_PATTERN = re.compile(r"viking://resources/techdocs/[^\s\"'<>\\]+")


def tool_definitions(mode: str, fixed_query: bool = False) -> list[dict[str, object]]:
    query_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The unchanged user question or a concise reformulation.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    if mode == "composite":
        composite_schema = (
            {"type": "object", "properties": {}, "additionalProperties": False}
            if fixed_query
            else query_schema
        )
        return [
            {
                "name": COMPOSITE_TOOL,
                "description": (
                    "Run the fixed technical-document retrieval pipeline and return one "
                    "citation-ready evidence package."
                ),
                "inputSchema": composite_schema,
            }
        ]
    if mode != "primitive":
        raise ValueError(f"unknown workflow mode: {mode}")
    return [
        {
            "name": SEARCH_TOOL,
            "description": (
                "Run first-stage technical-document retrieval without graph expansion and "
                "return citation-ready evidence."
            ),
            "inputSchema": query_schema,
        },
        {
            "name": EXPAND_TOOL,
            "description": (
                "Expand explicit Markdown-link neighbors from previously returned seed URIs, "
                "then rank the neighbors against the question."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The unchanged question or a concise reformulation.",
                    },
                    "seed_uris": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "In-scope viking:// URIs returned by techdocs_search.",
                    },
                },
                "required": ["query", "seed_uris"],
                "additionalProperties": False,
            },
        },
        {
            "name": FETCH_TOOL,
            "description": (
                "Fetch bounded full-document evidence for in-scope URIs returned by search "
                "or expansion."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "uris": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    }
                },
                "required": ["uris"],
                "additionalProperties": False,
            },
        },
    ]


class TechdocsMcpServer:
    def __init__(
        self,
        endpoint: str,
        mode: str,
        result_limit: int = 8,
        evidence_token_budget: int = 2200,
        trace_path: Path | None = None,
        fixed_query: str | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.mode = mode
        self.result_limit = max(1, min(30, int(result_limit)))
        self.evidence_token_budget = max(200, min(12000, int(evidence_token_budget)))
        self.trace_path = trace_path
        self.fixed_query = str(fixed_query or "").strip() or None
        self.tools = {
            str(tool["name"]): tool
            for tool in tool_definitions(mode, fixed_query=self.fixed_query is not None)
        }

    def handle(self, request: dict[str, Any]) -> dict[str, object] | None:
        method = str(request.get("method") or "")
        request_id = request.get("id")
        if method.startswith("notifications/"):
            return None
        if method == "initialize":
            protocol = str((request.get("params") or {}).get("protocolVersion") or "2024-11-05")
            return self._result(
                request_id,
                {
                    "protocolVersion": protocol,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                },
            )
        if method == "ping":
            return self._result(request_id, {})
        if method == "tools/list":
            return self._result(request_id, {"tools": list(self.tools.values())})
        if method == "tools/call":
            params = request.get("params") or {}
            name = str(params.get("name") or "")
            arguments = params.get("arguments") or {}
            try:
                text = self.call_tool(name, arguments)
                return self._result(
                    request_id,
                    {"content": [{"type": "text", "text": text}], "isError": False},
                )
            except (ValueError, HTTPError, URLError, OSError, json.JSONDecodeError) as error:
                self._trace(name, arguments, None, error=str(error))
                return self._result(
                    request_id,
                    {
                        "content": [{"type": "text", "text": f"KB tool error: {error}"}],
                        "isError": True,
                    },
                )
        if request_id is None:
            return None
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    def call_tool(self, name: str, arguments: object) -> str:
        if name not in self.tools:
            raise ValueError(f"tool is unavailable in {self.mode} mode: {name}")
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be an object")
        started = time.perf_counter()
        if name in {COMPOSITE_TOOL, SEARCH_TOOL}:
            query = (
                self.fixed_query
                if name == COMPOSITE_TOOL and self.fixed_query is not None
                else _required_string(arguments, "query")
            )
            result = self._post(
                "/v1/search",
                {
                    "query": query,
                    "result_limit": self.result_limit,
                    "evidence_token_budget": self.evidence_token_budget,
                    "graph": {"enabled": name == COMPOSITE_TOOL},
                },
            )
            text = str(result.get("evidenceText") or "No technical-document evidence found.")
        elif name == EXPAND_TOOL:
            query = _required_string(arguments, "query")
            seed_uris = _required_string_list(arguments, "seed_uris")
            result = self._post(
                "/v1/expand",
                {
                    "query": query,
                    "seed_uris": seed_uris,
                    "result_limit": self.result_limit,
                    "evidence_token_budget": self.evidence_token_budget,
                },
            )
            text = str(result.get("evidenceText") or "No technical-document evidence found.")
        else:
            uris = _required_string_list(arguments, "uris")
            result = self._post(
                "/v1/fetch",
                {"uris": uris, "token_budget": self.evidence_token_budget},
            )
            text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
        self._trace(
            name,
            arguments,
            text,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        return text

    def _post(self, path: str, body: dict[str, object]) -> dict[str, Any]:
        data = json.dumps(body, separators=(",", ":")).encode("utf-8")
        request = Request(
            f"{self.endpoint}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=30) as response:  # noqa: S310 - local pinned endpoint
            payload = json.loads(response.read())
        result = payload.get("result", payload)
        if not isinstance(result, dict):
            raise ValueError("KB service returned a non-object result")
        return result

    def _trace(
        self,
        name: str,
        arguments: object,
        output: str | None,
        latency_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        if self.trace_path is None:
            return
        event = {
            "time_unix": time.time(),
            "mode": self.mode,
            "tool": name,
            "arguments": arguments,
            "latency_ms": latency_ms,
            "output_sha256": hashlib.sha256((output or "").encode("utf-8")).hexdigest(),
            "output_characters": len(output or ""),
            "output_uris": sorted(
                {
                    value.rstrip(".,;:)]}")
                    for value in EVIDENCE_URI_PATTERN.findall(output or "")
                }
            ),
            "error": error,
        }
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _result(request_id: object, value: object) -> dict[str, object]:
        return {"jsonrpc": "2.0", "id": request_id, "result": value}


def _required_string(arguments: dict[str, object], key: str) -> str:
    value = str(arguments.get(key) or "").strip()
    if not value:
        raise ValueError(f"{key} is required")
    return value


def _required_string_list(arguments: dict[str, object], key: str) -> list[str]:
    raw = arguments.get(key)
    if not isinstance(raw, list):
        raise ValueError(f"{key} must be an array")
    values = [str(value).strip() for value in raw if str(value).strip()]
    if not values:
        raise ValueError(f"{key} must contain at least one value")
    return values


def run_stdio(server: TechdocsMcpServer) -> None:
    for raw_line in sys.stdin:
        try:
            request = json.loads(raw_line)
            if not isinstance(request, dict):
                raise ValueError("JSON-RPC request must be an object")
            response = server.handle(request)
        except (ValueError, json.JSONDecodeError) as error:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": str(error)},
            }
        if response is not None:
            sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
            sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared stdio MCP adapter for the KB harness eval.")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--mode", choices=("composite", "primitive"), required=True)
    parser.add_argument("--result-limit", type=int, default=8)
    parser.add_argument("--evidence-token-budget", type=int, default=2200)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--fixed-query")
    args = parser.parse_args()
    run_stdio(
        TechdocsMcpServer(
            endpoint=args.endpoint,
            mode=args.mode,
            result_limit=args.result_limit,
            evidence_token_budget=args.evidence_token_budget,
            trace_path=args.trace,
            fixed_query=args.fixed_query,
        )
    )


if __name__ == "__main__":
    main()
