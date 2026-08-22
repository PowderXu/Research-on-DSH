"""Fail-closed validation for SkillOpt-produced DSH skill candidates."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ARMS = frozenset({"fs", "hybrid", "neo4j"})
TOOLS = {
    "fs": ("grep", "read"),
    "hybrid": ("techdocs_search", "techdocs_fetch"),
    "neo4j": ("techdocs_search", "techdocs_expand", "techdocs_fetch"),
}
REQUIRED_HEADINGS = ("## Contract", "## Strategy", "## Failure recovery")


class SkillCandidateError(ValueError):
    """The candidate violates the protected DSH skill contract."""


def _parse_frontmatter(source: str) -> tuple[dict[str, str], str]:
    match = re.match(r"^---\n(.*?)\n---\n?(.*)$", source, flags=re.DOTALL)
    if not match:
        raise SkillCandidateError("skill document requires YAML-style frontmatter")
    metadata: dict[str, str] = {}
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        field = re.match(r"^([A-Za-z][A-Za-z0-9_-]*):\s*(.*?)\s*$", line)
        if not field:
            raise SkillCandidateError(f"unsupported frontmatter line: {raw_line}")
        key, value = field.groups()
        if key in metadata:
            raise SkillCandidateError(f"duplicate frontmatter key: {key}")
        metadata[key] = value.strip("'\"")
    return metadata, match.group(2).strip()


def load_leakage_markers(split_dir: Path) -> dict[str, set[str]]:
    markers = {"test_queries": set(), "all_qrel_ids": set()}
    for name in ("train", "val", "test"):
        path = split_dir / name / "items.json"
        rows: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        for row in rows:
            for qrel in row.get("qrel_ids") or []:
                qrel_text = str(qrel).casefold().rstrip("/") or "/"
                # The home-page qrel is just "/" and appears in ordinary prose;
                # it is not a meaningful memorization marker.
                if qrel_text != "/":
                    markers["all_qrel_ids"].add(qrel_text)
            if name == "test":
                query = " ".join(str(row.get("query") or "").casefold().split())
                if len(query) >= 40:
                    markers["test_queries"].add(query)
    return markers


def validate_skill_candidate(
    source: str,
    *,
    arm: str,
    expected_name: str | None = None,
    max_lines: int = 180,
    max_characters: int = 16_000,
    leakage_markers: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    normalized = str(source or "").replace("\r\n", "\n").replace("\r", "\n")
    if arm not in ARMS:
        raise SkillCandidateError(f"unknown arm: {arm}")
    if "\0" in normalized:
        raise SkillCandidateError("skill document contains a NUL byte")
    if len(normalized) > max_characters:
        raise SkillCandidateError(f"skill exceeds {max_characters} characters")
    if len(normalized.splitlines()) > max_lines:
        raise SkillCandidateError(f"skill exceeds {max_lines} lines")

    metadata, body = _parse_frontmatter(normalized)
    for key in ("name", "description", "arm", "version"):
        if not metadata.get(key):
            raise SkillCandidateError(f"skill frontmatter requires {key}")
    if metadata["arm"] != arm:
        raise SkillCandidateError(
            f"skill arm {metadata['arm']!r} does not match configured arm {arm!r}"
        )
    if expected_name and metadata["name"] != expected_name:
        raise SkillCandidateError(
            f"skill name {metadata['name']!r} does not match {expected_name!r}"
        )
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", metadata["name"]):
        raise SkillCandidateError("skill name must be lowercase kebab-case")
    if not 20 <= len(metadata["description"]) <= 300:
        raise SkillCandidateError("skill description must contain 20 to 300 characters")
    for heading in REQUIRED_HEADINGS:
        if heading not in body:
            raise SkillCandidateError(f"skill body requires {heading}")
    for tool in TOOLS[arm]:
        if f"`{tool}`" not in body:
            raise SkillCandidateError(f"skill must preserve protected tool name {tool}")

    lowered = normalized.casefold()
    forbidden = (
        "ignore the tool contract",
        "bypass the tool contract",
        "disable the evaluator",
        "change the benchmark",
        "read the qrel",
        "read qrel",
        "openai_api_key",
        "neo4j_password",
        "curl http",
        "wget http",
    )
    if any(marker in lowered for marker in forbidden):
        raise SkillCandidateError("skill contains a forbidden bypass, secret, or network instruction")

    if leakage_markers:
        leaked_qrels = sorted(
            marker
            for marker in leakage_markers.get("all_qrel_ids", set())
            if marker and marker in lowered
        )
        if leaked_qrels:
            raise SkillCandidateError(
                f"skill contains benchmark document IDs (first: {leaked_qrels[0]})"
            )
        compact = " ".join(lowered.split())
        if any(
            query in compact
            for query in leakage_markers.get("test_queries", set())
            if query
        ):
            raise SkillCandidateError("skill contains a held-out test question")

    return {
        "metadata": metadata,
        "body": body,
        "line_count": len(normalized.splitlines()),
        "character_count": len(normalized),
    }
