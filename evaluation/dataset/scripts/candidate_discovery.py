"""Discover and audit public GitHub Discussion candidates.

Discovery deliberately stops at a minimal identifier manifest. Question and
accepted-answer text are downloaded later into an ignored cache by
``build_all`` and are not written by this command.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .build_discussion_corpus import parse_discussion_html


ORG_DISCUSSION_LINK = re.compile(r'href="/orgs/([^/]+)/discussions/(\d+)"')
REPO_DISCUSSION_LINK = re.compile(r'href="/([^/]+)/([^/]+)/discussions/(\d+)"')
PAGE_LINK = re.compile(r"(?:&amp;|&)page=(\d+)")

# GitHub's searchable web listing has historically stopped around 1,000
# results. The lower threshold is an operational fail-closed guard: a range
# that approaches the ceiling must be divided before it can be frozen.
GITHUB_LISTING_RESULT_CAP = 1_000
CAP_GUARD_THRESHOLD = 950


class ListingCapRisk(RuntimeError):
    """Raised when a date partition is too close to the listing ceiling."""


@dataclass(frozen=True)
class DiscussionScope:
    organization: str | None = None
    repository: str | None = None

    def __post_init__(self) -> None:
        if (self.organization is None) == (self.repository is None):
            raise ValueError("provide exactly one of organization or repository")
        if self.repository is not None and self.repository.count("/") != 1:
            raise ValueError("repository must use owner/name form")

    @property
    def base_url(self) -> str:
        if self.organization is not None:
            return f"https://github.com/orgs/{self.organization}/discussions"
        return f"https://github.com/{self.repository}/discussions"

    @property
    def label(self) -> str:
        if self.organization is not None:
            return f"organization:{self.organization}"
        return f"repository:{self.repository}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def discussion_numbers_from_listing(
    page_html: str, scope: DiscussionScope
) -> set[int]:
    if scope.organization is not None:
        return {
            int(number)
            for organization, number in ORG_DISCUSSION_LINK.findall(page_html)
            if organization.casefold() == scope.organization.casefold()
        }
    assert scope.repository is not None
    owner, repository = scope.repository.split("/", 1)
    return {
        int(number)
        for found_owner, found_repository, number in REPO_DISCUSSION_LINK.findall(
            page_html
        )
        if found_owner.casefold() == owner.casefold()
        and found_repository.casefold() == repository.casefold()
    }


def max_listing_page(page_html: str) -> int:
    return max((int(value) for value in PAGE_LINK.findall(page_html)), default=1)


def ensure_below_listing_cap(count: int, created_range: str) -> None:
    if count >= CAP_GUARD_THRESHOLD:
        raise ListingCapRisk(
            f"created:{created_range} returned {count} unique IDs, too close to "
            f"the {GITHUB_LISTING_RESULT_CAP}-result listing ceiling; split the "
            "date range and retry"
        )


def _fetch(url: str, attempts: int = 4) -> str:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml",
            "User-Agent": "docsqa-public-candidate-discovery/1.0",
        },
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def collect_partition(
    scope: DiscussionScope, created_range: str
) -> tuple[list[int], dict[str, Any]]:
    query = f"is:answered created:{created_range}"
    first_url = scope.base_url + "?" + urllib.parse.urlencode(
        {"discussions_q": query, "page": 1}
    )
    first_page = _fetch(first_url)
    page_count = max_listing_page(first_page)
    numbers = discussion_numbers_from_listing(first_page, scope)
    urls = [
        scope.base_url
        + "?"
        + urllib.parse.urlencode({"discussions_q": query, "page": page})
        for page in range(2, page_count + 1)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for page_html in executor.map(_fetch, urls):
            numbers.update(discussion_numbers_from_listing(page_html, scope))
    ensure_below_listing_cap(len(numbers), created_range)
    return sorted(numbers), {
        "created_range": created_range,
        "listing_pages": page_count,
        "unique_discussion_ids": len(numbers),
        "query": query,
    }


def _allowed_docs_link(
    url: str, docs_hosts: set[str], docs_path_prefixes: tuple[str, ...]
) -> bool:
    parsed = urllib.parse.urlsplit(html.unescape(url))
    host = (parsed.hostname or "").casefold().rstrip(".")
    if host not in {value.casefold().rstrip(".") for value in docs_hosts}:
        return False
    path = re.sub(r"/+", "/", "/" + parsed.path.lstrip("/")).rstrip("/") or "/"
    return any(
        prefix == "/"
        or path == prefix.rstrip("/")
        or path.startswith(prefix.rstrip("/") + "/")
        for prefix in docs_path_prefixes
    )


def candidate_from_qapage(
    page_html: str,
    *,
    dataset: str,
    discussion_number: int,
    source_url: str,
    docs_hosts: set[str],
    docs_path_prefixes: tuple[str, ...],
) -> dict[str, Any] | None:
    parsed = parse_discussion_html(page_html)
    direct_links = [
        str(url)
        for url in parsed.get("accepted_answer_links") or []
        if _allowed_docs_link(str(url), docs_hosts, docs_path_prefixes)
    ]
    if not direct_links:
        return None
    return {
        "dataset": dataset,
        "discussion_number": discussion_number,
        "source_url": source_url,
        "accepted_answer_url": str(parsed["accepted_answer_url"]),
        "docs_host_links": direct_links,
    }


def fetch_candidates(
    *,
    dataset: str,
    scope: DiscussionScope,
    discussion_numbers: Iterable[int],
    docs_hosts: set[str],
    docs_path_prefixes: tuple[str, ...],
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def fetch_one(number: int) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        source_url = f"{scope.base_url}/{number}"
        try:
            return (
                candidate_from_qapage(
                    _fetch(source_url),
                    dataset=dataset,
                    discussion_number=number,
                    source_url=source_url,
                    docs_hosts=docs_hosts,
                    docs_path_prefixes=docs_path_prefixes,
                ),
                None,
            )
        except Exception as error:
            return None, {
                "discussion_number": number,
                "source_url": source_url,
                "error": str(error),
            }

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for row, failure in executor.map(fetch_one, sorted(set(discussion_numbers))):
            if row is not None:
                rows.append(row)
            if failure is not None:
                failures.append(failure)
    rows.sort(key=lambda row: int(row["discussion_number"]))
    failures.sort(key=lambda row: int(row["discussion_number"]))
    return rows, failures


def discover(args: argparse.Namespace) -> dict[str, Any]:
    scope = DiscussionScope(args.organization, args.repository)
    all_numbers: set[int] = set()
    partitions: list[dict[str, Any]] = []
    total_before_deduplication = 0
    for created_range in args.created_range:
        numbers, report = collect_partition(scope, created_range)
        partitions.append(report)
        total_before_deduplication += len(numbers)
        all_numbers.update(numbers)
    rows, failures = fetch_candidates(
        dataset=args.dataset,
        scope=scope,
        discussion_numbers=all_numbers,
        docs_hosts=set(args.docs_host),
        docs_path_prefixes=tuple(args.docs_path_prefix or ["/"]),
        workers=args.workers,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output / "candidate_discussions.jsonl", rows)
    _write_json(args.output / "failures.json", failures)
    summary = {
        "schema_version": 1,
        "dataset": args.dataset,
        "discussion_scope": scope.label,
        "docs_hosts": sorted(set(args.docs_host)),
        "docs_path_prefixes": sorted(set(args.docs_path_prefix or ["/"])),
        "created_ranges": args.created_range,
        "partitions": partitions,
        "partition_ids_before_deduplication": total_before_deduplication,
        "answered_listing_discussions": len(all_numbers),
        "accepted_answers_with_direct_docs_links": len(rows),
        "fetch_or_parse_failures": len(failures),
        "listing_result_cap": GITHUB_LISTING_RESULT_CAP,
        "cap_guard_threshold": CAP_GUARD_THRESHOLD,
        "measurement": (
            "acceptedAnswer direct-link screen only; pinned-corpus resolution and "
            "source-package validation occur in build_all"
        ),
    }
    _write_json(args.output / "summary.json", summary)
    return summary


def _question_resolution_report(
    project_dir: Path, frozen_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    if not (project_dir / "questions.jsonl").exists():
        return {"status": "not_available"}
    questions = _load_jsonl(project_dir / "questions.jsonl")
    question_urls = {
        str(row.get("source_url") or "").rstrip("/") for row in questions
    }
    frozen_urls = {
        str(row.get("source_url") or "").rstrip("/") for row in frozen_rows
    }
    corpus_ids = {
        str(row["doc_id"])
        for row in _load_jsonl(project_dir / "corpus.jsonl")
    }
    resolved = [row for row in questions if row.get("qrel_ids")]
    dangling = sorted(
        {
            str(qrel)
            for row in resolved
            for qrel in row.get("qrel_ids") or []
            if str(qrel) not in corpus_ids
        }
    )
    return {
        "status": "available",
        "question_rows": len(questions),
        "frozen_rows_found_by_source_url": len(frozen_urls & question_urls),
        "frozen_rows_missing_from_project_package": sorted(
            frozen_urls - question_urls
        ),
        "project_rows_not_in_frozen_manifest": sorted(question_urls - frozen_urls),
        "rows_with_at_least_one_resolved_qrel": len(resolved),
        "dangling_qrel_ids": dangling,
    }


def audit_frozen_manifest(
    *,
    frozen_manifest: Path,
    lineage: dict[str, Any],
    project_data_root: Path | None = None,
    historical_root: Path | None = None,
) -> dict[str, Any]:
    rows = _load_jsonl(frozen_manifest)
    duplicate_keys: list[str] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        key = (str(row.get("dataset")), int(row.get("discussion_number")))
        if key in seen:
            duplicate_keys.append(f"{key[0]}:{key[1]}")
        seen.add(key)
    by_dataset = {
        dataset: [row for row in rows if row.get("dataset") == dataset]
        for dataset in lineage["datasets"]
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "frozen_manifest": str(frozen_manifest),
        "frozen_manifest_sha256": _sha256(frozen_manifest),
        "expected_frozen_manifest_sha256": lineage["frozen_manifest"]["sha256"],
        "frozen_manifest_hash_matches": (
            _sha256(frozen_manifest)
            == lineage["frozen_manifest"]["sha256"]
        ),
        "frozen_rows": len(rows),
        "expected_frozen_rows": int(lineage["frozen_manifest"]["rows"]),
        "duplicate_dataset_discussion_keys": duplicate_keys,
        "datasets": {},
    }
    for dataset, spec in lineage["datasets"].items():
        frozen_rows = by_dataset[dataset]
        frozen_by_number = {
            int(row["discussion_number"]): row for row in frozen_rows
        }
        dataset_report: dict[str, Any] = {
            "frozen_rows": len(frozen_rows),
            "expected_frozen_rows": int(spec["frozen_rows"]),
            "selection_reproducibility": spec["selection_reproducibility"],
            "irrecoverable_step": spec.get("irrecoverable_step"),
        }
        if project_data_root is not None:
            dataset_report["pinned_corpus_resolution"] = _question_resolution_report(
                project_data_root / str(spec["project_id"]), frozen_rows
            )
        artifact = spec.get("historical_host_screen")
        if artifact and historical_root is not None:
            artifact_dir = historical_root / str(artifact["directory"])
            artifact_rows_path = artifact_dir / "host_link_answers.jsonl"
            if artifact_rows_path.exists():
                artifact_rows = _load_jsonl(artifact_rows_path)
                artifact_by_number = {
                    int(row["discussion_number"]): row for row in artifact_rows
                }
                missing = sorted(set(frozen_by_number) - set(artifact_by_number))
                mismatches: list[int] = []
                for number in sorted(set(frozen_by_number) & set(artifact_by_number)):
                    frozen = frozen_by_number[number]
                    old = artifact_by_number[number]
                    filtered_links = [
                        str(url)
                        for url in old.get("docs_host_links") or []
                        if _allowed_docs_link(
                            str(url),
                            set(spec["docs_hosts"]),
                            tuple(spec["docs_path_prefixes"]),
                        )
                    ]
                    if any(
                        (
                            frozen.get("source_url") != old.get("source_url"),
                            frozen.get("accepted_answer_url")
                            != old.get("accepted_answer_url"),
                            frozen.get("docs_host_links") != filtered_links,
                        )
                    ):
                        mismatches.append(number)
                exact_screen = {
                    int(row["discussion_number"])
                    for row in artifact_rows
                    if any(
                        _allowed_docs_link(
                            str(url),
                            set(spec["docs_hosts"]),
                            tuple(spec["docs_path_prefixes"]),
                        )
                        for url in row.get("docs_host_links") or []
                    )
                }
                dataset_report["historical_host_screen"] = {
                    "status": "available",
                    "sha256": _sha256(artifact_rows_path),
                    "expected_sha256": artifact["rows_sha256"],
                    "hash_matches": (
                        _sha256(artifact_rows_path) == artifact["rows_sha256"]
                    ),
                    "base_host_rows": len(artifact_rows),
                    "exact_host_and_path_rows": len(exact_screen),
                    "frozen_rows_missing_from_screen": missing,
                    "frozen_field_mismatches_after_exact_filter": mismatches,
                    "exact_screen_rows_not_in_frozen_manifest": sorted(
                        exact_screen - set(frozen_by_number)
                    ),
                }
                summary_path = artifact_dir / "summary.json"
                if summary_path.exists():
                    dataset_report["historical_host_screen"].update(
                        {
                            "summary_sha256": _sha256(summary_path),
                            "expected_summary_sha256": artifact[
                                "summary_sha256"
                            ],
                            "summary_hash_matches": (
                                _sha256(summary_path)
                                == artifact["summary_sha256"]
                            ),
                        }
                    )
            else:
                dataset_report["historical_host_screen"] = {
                    "status": "not_available",
                    "expected_sha256": artifact["rows_sha256"],
                }
        else:
            dataset_report["historical_host_screen"] = {
                "status": "not_retained"
            }
        report["datasets"][dataset] = dataset_report
    return report


def audit(args: argparse.Namespace) -> dict[str, Any]:
    lineage = json.loads(args.lineage.read_text(encoding="utf-8"))
    report = audit_frozen_manifest(
        frozen_manifest=args.frozen_manifest,
        lineage=lineage,
        project_data_root=args.project_data_root,
        historical_root=args.historical_root,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write_json(args.output, report)
    return report


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover_parser = subparsers.add_parser(
        "discover", help="run a new date-partitioned public candidate screen"
    )
    discover_parser.add_argument("--dataset", required=True)
    scope = discover_parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--organization")
    scope.add_argument("--repository")
    discover_parser.add_argument("--docs-host", action="append", required=True)
    discover_parser.add_argument("--docs-path-prefix", action="append")
    discover_parser.add_argument(
        "--created-range",
        action="append",
        required=True,
        help="repeatable YYYY-MM-DD..YYYY-MM-DD range; cap-risk ranges fail closed",
    )
    discover_parser.add_argument("--workers", type=int, default=6)
    discover_parser.add_argument("--output", type=Path, required=True)

    audit_parser = subparsers.add_parser(
        "audit", help="audit the frozen manifest and any retained local lineage"
    )
    audit_parser.add_argument(
        "--frozen-manifest",
        type=Path,
        default=project_root
        / "evaluation/dataset/templates/discussion_sources.jsonl",
    )
    audit_parser.add_argument(
        "--lineage",
        type=Path,
        default=project_root
        / "evaluation/dataset/templates/candidate_discovery.json",
    )
    audit_parser.add_argument(
        "--project-data-root",
        type=Path,
        default=project_root / "evaluation/dataset/evaluation_data/projects",
    )
    audit_parser.add_argument(
        "--historical-root", type=Path, default=project_root.parent
    )
    audit_parser.add_argument("--output", type=Path)

    args = parser.parse_args()
    if getattr(args, "workers", 1) < 1:
        raise SystemExit("--workers must be positive")
    result = discover(args) if args.command == "discover" else audit(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
