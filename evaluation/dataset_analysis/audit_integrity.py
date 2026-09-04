"""Audit deterministic duplicate and split-overlap risks in the local dataset.

This module intentionally performs only exact and lexical checks.  Its
near-duplicate output is a diagnostic candidate list, not a semantic-duplicate
decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import SplitResult, urlsplit, urlunsplit


SCHEMA_VERSION = "dataset-integrity-v1"
DEVELOPMENT_SPLIT_ALIASES = {"test": "development"}
LEXICAL_SHINGLE_SIZE = 3
LEXICAL_MIN_TOKENS = 8
LEXICAL_JACCARD_THRESHOLD = 0.80
_WORD_RE = re.compile(r"(?u)\b\w+\b")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_split(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    return DEVELOPMENT_SPLIT_ALIASES.get(normalized, normalized or "unknown")


def normalize_identifier(value: Any) -> str:
    """Apply a deliberately narrow syntactic normalization to an identifier."""

    normalized = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    return re.sub(r"\s+", "", normalized)


def normalize_url(value: Any, *, keep_fragment: bool) -> str:
    """Normalize URL syntax without claiming two destinations are semantic peers."""

    raw = unicodedata.normalize("NFKC", str(value or "")).strip()
    if not raw:
        return ""
    parsed = urlsplit(raw)
    if not parsed.scheme or not parsed.netloc:
        return raw

    scheme = parsed.scheme.casefold()
    hostname = (parsed.hostname or "").casefold()
    try:
        port = parsed.port
    except ValueError:
        return raw
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        hostname = f"{hostname}:{port}"
    if parsed.username:
        userinfo = parsed.username
        if parsed.password:
            userinfo += f":{parsed.password}"
        hostname = f"{userinfo}@{hostname}"

    path = parsed.path
    if path != "/":
        path = path.rstrip("/")
    fragment = parsed.fragment if keep_fragment else ""
    return urlunsplit(SplitResult(scheme, hostname, path, parsed.query, fragment))


def lexical_tokens(value: Any) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return tuple(_WORD_RE.findall(normalized))


def lexical_shingles(tokens: tuple[str, ...]) -> frozenset[tuple[str, ...]]:
    if len(tokens) < LEXICAL_SHINGLE_SIZE:
        return frozenset()
    return frozenset(
        tokens[index : index + LEXICAL_SHINGLE_SIZE]
        for index in range(len(tokens) - LEXICAL_SHINGLE_SIZE + 1)
    )


def jaccard(left: frozenset[Any], right: frozenset[Any]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _record(row: dict[str, Any], index: int) -> dict[str, Any]:
    project = normalize_identifier(row.get("project"))
    source_id = normalize_identifier(row.get("source_question_id"))
    return {
        "record_index": index,
        "question_id": str(row.get("question_id") or ""),
        "split": canonical_split(row.get("split")),
        "project": str(row.get("project") or ""),
        "query": str(row.get("query") or ""),
        "normalized_question_id": normalize_identifier(row.get("question_id")),
        "normalized_source_question_id": (
            f"{project}::{source_id}" if project and source_id else ""
        ),
        "normalized_source_url": normalize_url(
            row.get("source_url"), keep_fragment=False
        ),
        "normalized_accepted_answer_url": normalize_url(
            row.get("accepted_answer_url"), keep_fragment=True
        ),
    }


def _group_duplicates(
    records: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        value = record[key]
        if value != "":
            grouped[str(value)].append(record)

    groups: list[dict[str, Any]] = []
    for value, members in grouped.items():
        if len(members) < 2:
            continue
        ordered = sorted(
            members,
            key=lambda item: (item["question_id"], item["record_index"]),
        )
        splits = sorted({str(item["split"]) for item in ordered})
        groups.append(
            {
                "normalized_value": value,
                "record_count": len(ordered),
                "question_ids": [item["question_id"] for item in ordered],
                "record_indices": [item["record_index"] for item in ordered],
                "splits": splits,
                "cross_split": len(splits) > 1,
            }
        )
    groups.sort(
        key=lambda item: (
            item["normalized_value"],
            item["question_ids"],
            item["record_indices"],
        )
    )
    return {
        "group_count": len(groups),
        "record_count": sum(group["record_count"] for group in groups),
        "cross_split_group_count": sum(group["cross_split"] for group in groups),
        "groups": groups,
    }


def _lexical_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[tuple[dict[str, Any], tuple[str, ...], frozenset[tuple[str, ...]]]] = []
    for record in records:
        tokens = lexical_tokens(record["query"])
        prepared.append((record, tokens, lexical_shingles(tokens)))

    pairs: list[dict[str, Any]] = []
    for (left, left_tokens, left_shingles), (
        right,
        right_tokens,
        right_shingles,
    ) in combinations(prepared, 2):
        if left["query"] == right["query"]:
            continue
        if min(len(left_tokens), len(right_tokens)) < LEXICAL_MIN_TOKENS:
            continue
        similarity = jaccard(left_shingles, right_shingles)
        if similarity + 1e-12 < LEXICAL_JACCARD_THRESHOLD:
            continue
        pairs.append(
            {
                "left_question_id": left["question_id"],
                "left_record_index": left["record_index"],
                "left_split": left["split"],
                "right_question_id": right["question_id"],
                "right_record_index": right["record_index"],
                "right_split": right["split"],
                "jaccard": round(similarity, 6),
                "left_token_count": len(left_tokens),
                "right_token_count": len(right_tokens),
                "cross_split": left["split"] != right["split"],
            }
        )
    pairs.sort(
        key=lambda item: (
            -item["jaccard"],
            item["left_question_id"],
            item["left_record_index"],
            item["right_question_id"],
            item["right_record_index"],
        )
    )
    return pairs


def _pairwise_split_counts(
    duplicate_sections: dict[str, dict[str, Any]],
    lexical_pairs: list[dict[str, Any]],
    splits: Iterable[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left, right in combinations(sorted(set(splits)), 2):
        section_counts: dict[str, int] = {}
        for name, section in duplicate_sections.items():
            section_counts[name] = sum(
                left in group["splits"] and right in group["splits"]
                for group in section["groups"]
            )
        rows.append(
            {
                "splits": [left, right],
                **section_counts,
                "lexical_candidate_pairs": sum(
                    {pair["left_split"], pair["right_split"]} == {left, right}
                    for pair in lexical_pairs
                ),
            }
        )
    return rows


def audit(dataset_dir: Path) -> dict[str, Any]:
    questions_path = dataset_dir / "questions.jsonl"
    rows = _read_jsonl(questions_path)
    records = [_record(row, index) for index, row in enumerate(rows)]

    duplicate_sections = {
        "exact_question": _group_duplicates(records, "query"),
        "normalized_question_id": _group_duplicates(
            records, "normalized_question_id"
        ),
        "normalized_source_question_id": _group_duplicates(
            records, "normalized_source_question_id"
        ),
        "normalized_source_url": _group_duplicates(records, "normalized_source_url"),
        "normalized_accepted_answer_url": _group_duplicates(
            records, "normalized_accepted_answer_url"
        ),
    }
    lexical_pairs = _lexical_pairs(records)
    split_counts = dict(sorted(Counter(record["split"] for record in records).items()))

    return {
        "schema_version": SCHEMA_VERSION,
        "input": {
            "dataset_dir": str(dataset_dir),
            "questions_path": str(questions_path),
            "questions_sha256": _sha256(questions_path),
            "question_count": len(records),
            "split_counts": split_counts,
        },
        "method": {
            "split_aliases": DEVELOPMENT_SPLIT_ALIASES,
            "exact_question": "Python string equality on the normalized dataset's query field.",
            "identifier_normalization": (
                "Unicode NFKC, trim, Unicode case-fold, then remove all whitespace. "
                "source_question_id is namespaced by normalized project."
            ),
            "url_normalization": (
                "Unicode NFKC and trim; lowercase scheme and host; remove default "
                "HTTP(S) ports and a non-root trailing slash. Query order is preserved. "
                "source_url fragments are removed; accepted_answer_url fragments are preserved."
            ),
            "lexical_near_duplicate_diagnostic": {
                "label": "lexical candidate only; not a semantic-duplicate judgment",
                "unicode": "NFKC",
                "case": "Unicode case-fold",
                "tokenizer": r"Python Unicode regex (?u)\b\w+\b",
                "shingle_size_tokens": LEXICAL_SHINGLE_SIZE,
                "minimum_tokens_per_question": LEXICAL_MIN_TOKENS,
                "similarity": "set Jaccard over contiguous token shingles",
                "threshold_inclusive": LEXICAL_JACCARD_THRESHOLD,
                "exact_raw_query_pairs_excluded": True,
                "stemming_or_stopword_removal": False,
            },
        },
        "duplicates": duplicate_sections,
        "lexical_near_duplicate_diagnostic": {
            "candidate_pair_count": len(lexical_pairs),
            "cross_split_candidate_pair_count": sum(
                pair["cross_split"] for pair in lexical_pairs
            ),
            "pairs": lexical_pairs,
        },
        "split_overlap": {
            "pairwise": _pairwise_split_counts(
                duplicate_sections,
                lexical_pairs,
                split_counts,
            ),
            "note": (
                "The legacy input label 'test' is reported as 'development' because "
                "the current cohort has been inspected during benchmark development."
            ),
        },
        "limitations": [
            "This deterministic audit cannot establish that no semantic duplicates exist.",
            "Lexical candidate pairs require manual or independently specified semantic review.",
            "Pairs below the fixed lexical threshold, paraphrases, and translations may be missed.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    input_info = report["input"]
    lines = [
        "# Dataset integrity diagnostic",
        "",
        "This is a deterministic exact/lexical audit of the normalized local dataset. "
        "It does **not** establish semantic-duplicate absence.",
        "",
        "## Input",
        "",
        f"- Questions: {input_info['question_count']}",
        f"- SHA-256: `{input_info['questions_sha256']}`",
        "- Splits: "
        + ", ".join(
            f"{name}={count}" for name, count in input_info["split_counts"].items()
        ),
        "",
        "The legacy `test` label is reported as `development` because this cohort has "
        "been inspected during benchmark development.",
        "",
        "## Exact and normalized duplicates",
        "",
        "| Check | Duplicate groups | Records in groups | Cross-split groups |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "exact_question": "Exact query string",
        "normalized_question_id": "Normalized question ID",
        "normalized_source_question_id": "Project-scoped normalized source ID",
        "normalized_source_url": "Normalized source URL",
        "normalized_accepted_answer_url": "Normalized accepted-answer URL",
    }
    for key, label in labels.items():
        section = report["duplicates"][key]
        lines.append(
            f"| {label} | {section['group_count']} | {section['record_count']} | "
            f"{section['cross_split_group_count']} |"
        )

    lexical = report["lexical_near_duplicate_diagnostic"]
    method = report["method"]["lexical_near_duplicate_diagnostic"]
    lines.extend(
        [
            "",
            "## Lexical near-duplicate diagnostic",
            "",
            f"Candidate pairs: {lexical['candidate_pair_count']} total, "
            f"{lexical['cross_split_candidate_pair_count']} cross-split.",
            "",
            "Method: Unicode NFKC + case-folding, Unicode word tokens, contiguous "
            f"{method['shingle_size_tokens']}-token shingle sets, Jaccard "
            f"≥ {method['threshold_inclusive']:.2f}, and at least "
            f"{method['minimum_tokens_per_question']} tokens per question. No stemming "
            "or stopword removal is used. Exact raw-query duplicates are listed above "
            "and excluded here.",
            "",
            "These are lexical review candidates only, not semantic-duplicate labels.",
            "",
            "| Left | Left split | Right | Right split | Jaccard |",
            "|---|---|---|---|---:|",
        ]
    )
    if lexical["pairs"]:
        for pair in lexical["pairs"]:
            lines.append(
                f"| `{pair['left_question_id']}` | {pair['left_split']} | "
                f"`{pair['right_question_id']}` | {pair['right_split']} | "
                f"{pair['jaccard']:.6f} |"
            )
    else:
        lines.append("| _None at the fixed threshold_ |  |  |  |  |")

    lines.extend(
        [
            "",
            "## Cross-split overlap",
            "",
            "| Splits | Exact query | Question ID | Source ID | Source URL | "
            "Accepted-answer URL | Lexical candidates |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["split_overlap"]["pairwise"]:
        lines.append(
            f"| {' / '.join(row['splits'])} | {row['exact_question']} | "
            f"{row['normalized_question_id']} | "
            f"{row['normalized_source_question_id']} | "
            f"{row['normalized_source_url']} | "
            f"{row['normalized_accepted_answer_url']} | "
            f"{row['lexical_candidate_pairs']} |"
        )

    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- This audit cannot establish that no semantic duplicates exist.",
            "- Paraphrases, translations, and pairs below the threshold may be missed.",
            "- Every lexical candidate requires a separately specified review decision.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    report = audit(args.dataset_dir)
    write_report(report, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
