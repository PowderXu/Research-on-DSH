"""HTTP and citation helpers shared by the DSH plugin backend."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit


RESOURCE_ROOT = "viking://resources/techdocs"


class PluginService(Protocol):
    def health(self) -> dict[str, Any]: ...
    def search(self, request: dict[str, Any]) -> dict[str, Any]: ...
    def expand(self, request: dict[str, Any]) -> dict[str, Any]: ...
    def fetch(self, request: dict[str, Any]) -> dict[str, Any]: ...


class TechdocsRequestHandler(BaseHTTPRequestHandler):
    service: PluginService

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if urlsplit(self.path).path != "/health":
            self._write_error(404, "not found")
            return
        self._write_json(200, {"result": self.service.health()})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        try:
            request = self._read_json()
            path = urlsplit(self.path).path
            if path == "/v1/search":
                result = self.service.search(request)
            elif path == "/v1/expand":
                result = self.service.expand(request)
            elif path == "/v1/fetch":
                result = self.service.fetch(request)
            else:
                self._write_error(404, "not found")
                return
            self._write_json(200, {"result": result})
        except (ValueError, json.JSONDecodeError) as error:
            self._write_error(400, str(error))
        except Exception as error:  # pragma: no cover - defensive HTTP boundary
            self._write_error(500, str(error))

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 2_000_000:
            raise ValueError("invalid request body size")
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _write_error(self, status: int, message: str) -> None:
        self._write_json(status, {"error": {"message": message}})

    def _write_json(self, status: int, value: object) -> None:
        body = json.dumps(value, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def render_evidence_text(
    query_id: str,
    results: list[dict[str, Any]],
    token_budget: int,
) -> str:
    character_budget = max(400, token_budget * 4)
    blocks: list[str] = []
    used = 0
    for index, result in enumerate(results, start=1):
        source = str(result.get("repoPath") or "")
        line_start = result.get("lineStart")
        line_end = result.get("lineEnd")
        if source and isinstance(line_start, int):
            source += f":{line_start}"
            if isinstance(line_end, int) and line_end != line_start:
                source += f"-{line_end}"
        commit = str(result.get("commit") or "")
        if source and commit:
            source += f" @ {commit}"
        section = str(result.get("section") or "")
        expanded_from = [str(value) for value in result.get("expandedFrom") or []]
        lines = [
            f"[{index}] {result.get('title') or result.get('uri') or ''}",
            f"URI: {result.get('uri') or ''}{f'#{section}' if section else ''}",
        ]
        if source:
            lines.append(f"Source: {source}")
        if expanded_from:
            lines.append(f"Expanded from: {', '.join(expanded_from)}")
        signals = ", ".join(str(value) for value in result.get("signals") or [])
        lines.extend(
            [
                f"Score: {float(result.get('score') or 0):.4f}; signals: {signals or 'unspecified'}",
                str(result.get("snippet") or ""),
            ]
        )
        block = "\n".join(line for line in lines if line)
        if blocks and used + len(block) > character_budget:
            break
        blocks.append(block[: max(0, character_budget - used)])
        used += len(blocks[-1])
    if not blocks:
        return "No technical-document evidence found."
    return "\n\n".join(
        [f'<techdocs-evidence query-id="{query_id}">', *blocks, "</techdocs-evidence>"]
    )


def document_uri(doc_id: str) -> str:
    normalized = doc_id.lstrip("/")
    if not normalized or normalized.startswith("../"):
        raise ValueError("invalid document id")
    return f"{RESOURCE_ROOT}/{normalized}"


def document_id_from_uri(uri: str) -> str:
    value = uri.split("#", 1)[0].rstrip("/")
    prefix = f"{RESOURCE_ROOT}/"
    if not value.startswith(prefix):
        raise ValueError(f"URI must be below {RESOURCE_ROOT}")
    doc_id = unquote(value[len(prefix) :])
    if not doc_id or doc_id.startswith("../") or "/../" in f"/{doc_id}":
        raise ValueError("invalid technical-document URI")
    return doc_id
