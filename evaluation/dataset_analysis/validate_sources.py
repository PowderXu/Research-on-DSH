"""Deterministically validate DocsQA source packages before semantic judging.

The accepted evaluation unit is the question and accepted-answer text/images,
plus documentation links that resolve to the pinned local corpus. A reference
answer that links another GitHub issue or discussion is expanded only when the
target is another frozen QA source with an accepted answer; otherwise the case
is rejected. External pages are never imported as documentation evidence.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlsplit, urlunsplit


SCRIPT_RE = re.compile(
    r"<script[^>]+type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>",
    re.IGNORECASE | re.DOTALL,
)
GITHUB_DISCUSSION_RE = re.compile(
    r"^/orgs/[^/]+/discussions/\d+$|^/[^/]+/[^/]+/discussions/\d+$",
    re.IGNORECASE,
)
GITHUB_ISSUE_RE = re.compile(r"^/[^/]+/[^/]+/issues/\d+$", re.IGNORECASE)
MARKDOWN_IMAGE_RE = re.compile(
    r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+[\"'][^\"']*[\"'])?\)", re.MULTILINE
)
HTML_IMAGE_RE = re.compile(
    r"<(?:img|image)\b[^>]*?\bsrc=[\"']([^\"']+)[\"'][^>]*>",
    re.IGNORECASE,
)
IMAGE_URL_RE = re.compile(r"\.(?:png|jpe?g|gif|webp|svg|avif)(?:$|[?#])", re.IGNORECASE)


@dataclass(frozen=True)
class HtmlLink:
    url: str
    text: str


@dataclass(frozen=True)
class HtmlImage:
    url: str
    alt: str
    title: str


class _RichHtmlParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.links: list[HtmlLink] = []
        self.images: list[HtmlImage] = []
        self._anchors: list[dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.casefold(): value or "" for key, value in attrs}
        if tag.casefold() == "a" and values.get("href"):
            self._anchors.append(
                {"url": urljoin(self.base_url, html.unescape(values["href"])), "parts": []}
            )
        if tag.casefold() == "img" and values.get("src"):
            self.images.append(
                HtmlImage(
                    url=urljoin(self.base_url, html.unescape(values["src"])),
                    alt=html.unescape(values.get("alt", "")).strip(),
                    title=html.unescape(values.get("title", "")).strip(),
                )
            )

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "a" and self._anchors:
            value = self._anchors.pop()
            self.links.append(
                HtmlLink(url=str(value["url"]), text=" ".join(value["parts"]).strip())
            )

    def handle_data(self, data: str) -> None:
        if self._anchors and data.strip():
            self._anchors[-1]["parts"].append(data.strip())


def _walk_json(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _visible_text(value: str) -> str:
    from evaluation.dataset.scripts.build import html_to_text

    return html_to_text(value)


def extract_qa_html(page_html: str, source_url: str) -> dict[str, Any]:
    """Extract text, links, and image references from a frozen QAPage."""

    for raw_script in SCRIPT_RE.findall(page_html):
        try:
            document = json.loads(html.unescape(raw_script))
        except json.JSONDecodeError:
            continue
        for candidate in _walk_json(document):
            kind = candidate.get("@type")
            if kind not in {"QAPage", "Question"}:
                continue
            question = candidate.get("mainEntity") if kind == "QAPage" else candidate
            if not isinstance(question, dict):
                continue
            answer = question.get("acceptedAnswer")
            if isinstance(answer, list):
                answer = answer[0] if answer else None
            if not isinstance(answer, dict):
                continue
            question_html = str(question.get("text") or question.get("description") or "")
            answer_html = str(answer.get("text") or answer.get("description") or "")
            if not answer_html:
                continue
            question_parser = _RichHtmlParser(source_url)
            question_parser.feed(question_html)
            answer_parser = _RichHtmlParser(source_url)
            answer_parser.feed(answer_html)
            return {
                "title": _visible_text(str(question.get("name") or "")),
                "question_text": _visible_text(question_html),
                "answer_text": _visible_text(answer_html),
                "question_links": [link.__dict__ for link in question_parser.links],
                "answer_links": [link.__dict__ for link in answer_parser.links],
                "question_images": [image.__dict__ for image in question_parser.images],
                "answer_images": [image.__dict__ for image in answer_parser.images],
            }
    raise ValueError("no QAPage with an accepted answer was found")


def canonical_url(url: str) -> str:
    parsed = urlsplit(html.unescape(url))
    scheme = parsed.scheme.casefold() or "https"
    host = (parsed.hostname or "").casefold()
    port = f":{parsed.port}" if parsed.port else ""
    path = re.sub(r"/+", "/", parsed.path).rstrip("/") or "/"
    return urlunsplit((scheme, host + port, path, "", ""))


def is_linked_qa_url(url: str) -> bool:
    parsed = urlsplit(url)
    if (parsed.hostname or "").casefold() != "github.com":
        return False
    path = re.sub(r"/+", "/", parsed.path).rstrip("/") or "/"
    return bool(GITHUB_DISCUSSION_RE.match(path) or GITHUB_ISSUE_RE.match(path))


def markdown_images(text: str) -> list[dict[str, str]]:
    images = [
        {"url": html.unescape(match.group(2)), "alt": match.group(1).strip(), "title": ""}
        for match in MARKDOWN_IMAGE_RE.finditer(text)
    ]
    images.extend(
        {"url": html.unescape(match.group(1)), "alt": "", "title": ""}
        for match in HTML_IMAGE_RE.finditer(text)
    )
    unique = {(item["url"], item["alt"], item["title"]): item for item in images}
    return list(unique.values())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _dataset_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("dataset must be NAME=DATASET_DIR")
    name, path = value.split("=", 1)
    return name, Path(path)


def _html_dir_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("discussion-dir must be NAME=HTML_DIR")
    name, path = value.split("=", 1)
    return name, Path(path)


def _question_number(dataset: str, question_id: str) -> str:
    prefix = dataset + "-"
    return question_id[len(prefix) :] if question_id.startswith(prefix) else question_id


def _docs_hosts(name: str, manifest: dict[str, Any]) -> set[str]:
    configured = manifest.get("docs_hosts")
    if isinstance(configured, list) and configured:
        return {str(value).casefold() for value in configured}
    fallback = {
        "github_docs": {"docs.github.com"},
        "prisma": {"prisma.io", "www.prisma.io"},
        "supabase": {"supabase.com"},
        "tailwind": {"tailwindcss.com"},
    }
    return fallback.get(name, set())


def _is_internal_doc(url: str, hosts: set[str]) -> bool:
    return (urlsplit(url).hostname or "").casefold() in hosts


def _is_identity_link(link: dict[str, str]) -> bool:
    parsed = urlsplit(link["url"])
    parts = [part for part in parsed.path.split("/") if part]
    text = str(link.get("text") or "").strip()
    return (parsed.hostname or "").casefold() == "github.com" and (
        (len(parts) == 1 and not parts[0].casefold() in {"settings", "features"})
        or text.startswith("@")
    )


def _is_media_link(url: str, known_images: set[str]) -> bool:
    canonical = canonical_url(url)
    return canonical in known_images or bool(IMAGE_URL_RE.search(urlsplit(url).path))


def validate_datasets(
    datasets: dict[str, Path], discussion_dirs: dict[str, Path]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prepared: dict[tuple[str, str], dict[str, Any]] = {}
    source_registry: dict[str, tuple[str, str]] = {}
    corpora: dict[str, dict[str, dict[str, Any]]] = {}
    manifests: dict[str, dict[str, Any]] = {}

    for name, dataset_dir in datasets.items():
        questions = _load_jsonl(dataset_dir / "questions.jsonl")
        corpus = {str(row["doc_id"]): row for row in _load_jsonl(dataset_dir / "corpus.jsonl")}
        manifest_path = dataset_dir / "manifest.json"
        if not manifest_path.exists():
            manifest_path = dataset_dir / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        corpora[name] = corpus
        manifests[name] = manifest
        html_dir = discussion_dirs[name]
        for question in questions:
            qid = str(question["question_id"])
            source_url = str(question.get("source_url") or "")
            html_path = html_dir / f"{_question_number(name, qid)}.html"
            record: dict[str, Any] = {
                "dataset": name,
                "question": question,
                "html_path": html_path,
                "source_url": source_url,
                "parse_error": None,
            }
            try:
                record["content"] = extract_qa_html(
                    html_path.read_text(encoding="utf-8", errors="replace"), source_url
                )
            except Exception as error:
                record["content"] = None
                record["parse_error"] = str(error)
            prepared[(name, qid)] = record
            if source_url:
                source_registry[canonical_url(source_url)] = (name, qid)

    memo: dict[tuple[str, str], tuple[str | None, list[str], list[str]]] = {}

    def expand_answer(
        key: tuple[str, str], active: tuple[tuple[str, str], ...] = ()
    ) -> tuple[str | None, list[str], list[str]]:
        if key in memo:
            return memo[key]
        if key in active:
            return None, [], ["linked_qa_cycle"]
        record = prepared[key]
        content = record.get("content")
        if not isinstance(content, dict):
            result = (None, [], ["missing_or_invalid_frozen_source"])
            memo[key] = result
            return result
        source = canonical_url(str(record["source_url"]))
        targets = sorted(
            {
                canonical_url(str(link["url"]))
                for link in content["answer_links"]
                if is_linked_qa_url(str(link["url"]))
                and canonical_url(str(link["url"])) != source
            }
        )
        answer = str(content["answer_text"]).strip()
        sources: list[str] = []
        errors: list[str] = []
        for target in targets:
            target_key = source_registry.get(target)
            if target_key is None:
                errors.append(f"unresolved_linked_qa:{target}")
                continue
            child_answer, child_sources, child_errors = expand_answer(
                target_key, active + (key,)
            )
            if child_answer is None or child_errors:
                errors.extend(child_errors or [f"missing_linked_answer:{target}"])
                continue
            answer += f"\n\n[Linked accepted answer: {target}]\n{child_answer}"
            sources.append(target)
            sources.extend(child_sources)
        result = (answer if not errors else None, list(dict.fromkeys(sources)), errors)
        memo[key] = result
        return result

    rows: list[dict[str, Any]] = []
    for key, record in prepared.items():
        name, qid = key
        question = record["question"]
        content = record.get("content") or {}
        corpus = corpora[name]
        hosts = _docs_hosts(name, manifests[name])
        resolution = question.get("resolution_kinds") or {}
        docs_urls = [str(url) for url in question.get("docs_urls") or []]
        unresolved_internal = sorted(url for url in docs_urls if not resolution.get(url))
        noninternal_docs = sorted(url for url in docs_urls if not _is_internal_doc(url, hosts))
        missing_qrels = sorted(
            str(doc_id) for doc_id in question.get("qrel_ids") or [] if str(doc_id) not in corpus
        )
        expanded, linked_sources, expansion_errors = expand_answer(key)
        source_url = canonical_url(str(record.get("source_url") or ""))
        answer_links = list(content.get("answer_links") or [])
        question_images = list(content.get("question_images") or [])
        answer_images = list(content.get("answer_images") or [])
        known_images = {
            canonical_url(str(image["url"]))
            for image in question_images + answer_images
            if image.get("url")
        }
        external_links = sorted(
            {
                str(link["url"])
                for link in answer_links
                if not _is_internal_doc(str(link["url"]), hosts)
                and not is_linked_qa_url(str(link["url"]))
                and canonical_url(str(link["url"])) != source_url
                and not _is_identity_link(link)
                and not _is_media_link(str(link["url"]), known_images)
            }
        )
        document_images = {
            str(doc_id): markdown_images(
                str(corpus[str(doc_id)].get("raw_text") or corpus[str(doc_id)].get("rendered_text") or "")
            )
            for doc_id in question.get("qrel_ids") or []
            if str(doc_id) in corpus
        }
        errors = []
        if record.get("parse_error"):
            errors.append("missing_or_invalid_frozen_source")
        if unresolved_internal:
            errors.append("unresolved_internal_documentation")
        if noninternal_docs:
            errors.append("noninternal_documentation_url")
        if missing_qrels:
            errors.append("qrel_missing_from_pinned_corpus")
        if external_links:
            errors.append("external_non_document_link")
        errors.extend(expansion_errors)
        rows.append(
            {
                "dataset": name,
                "question_id": qid,
                "status": "accepted_structurally" if not errors else "rejected",
                "rejection_reasons": list(dict.fromkeys(errors)),
                "query_text": content.get("question_text"),
                "expanded_reference_answer": expanded,
                "resolved_internal_docs_urls": [
                    url for url in docs_urls if resolution.get(url)
                ],
                "qrel_ids": [str(value) for value in question.get("qrel_ids") or []],
                "unresolved_internal_docs": unresolved_internal,
                "noninternal_docs": noninternal_docs,
                "missing_qrels": missing_qrels,
                "linked_answer_sources": linked_sources,
                "question_images": question_images,
                "answer_images": answer_images,
                "document_images": document_images,
                "requires_multimodal_judgment": bool(
                    question_images
                    or answer_images
                    or any(document_images.values())
                ),
                "external_non_document_links": external_links,
            }
        )

    datasets_summary = []
    for name in sorted(datasets):
        subset = [row for row in rows if row["dataset"] == name]
        datasets_summary.append(
            {
                "dataset": name,
                "questions": len(subset),
                "accepted_structurally": sum(row["status"] == "accepted_structurally" for row in subset),
                "rejected": sum(row["status"] == "rejected" for row in subset),
                "with_images": sum(row["requires_multimodal_judgment"] for row in subset),
                "with_external_non_document_links": sum(
                    bool(row["external_non_document_links"]) for row in subset
                ),
            }
        )
    reasons: dict[str, int] = {}
    for row in rows:
        for reason in row["rejection_reasons"]:
            category = reason.split(":", 1)[0]
            reasons[category] = reasons.get(category, 0) + 1
    report = {
        "policy": {
            "evidence_package": "question text/images + expanded accepted-answer text/images + resolved pinned internal documentation",
            "unresolved_internal_docs": "reject",
            "linked_qa": "append frozen accepted answer recursively; reject if unavailable",
            "external_resources": "never imported as documentation evidence",
            "external_non_document_links": "reject, except identity/self/media links and successfully expanded linked QA",
            "images": "preserved as references and require a later pixel-level multimodal judgment",
        },
        "questions": len(rows),
        "accepted_structurally": sum(row["status"] == "accepted_structurally" for row in rows),
        "rejected": sum(row["status"] == "rejected" for row in rows),
        "rejection_reasons": dict(sorted(reasons.items())),
        "datasets": datasets_summary,
    }
    return rows, report


def render_report(report: dict[str, Any]) -> str:
    lines = [
        "# QA source-package structural validation",
        "",
        "This is a deterministic preflight, not the final semantic correctness judgment.",
        "",
        f"- Questions audited: {report['questions']}",
        f"- Structurally eligible: {report['accepted_structurally']}",
        f"- Rejected: {report['rejected']}",
        "",
        "## Dataset results",
        "",
        "| Dataset | Questions | Eligible | Rejected | Image-bearing | Other external links |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["datasets"]:
        lines.append(
            f"| {row['dataset']} | {row['questions']} | {row['accepted_structurally']} | "
            f"{row['rejected']} | {row['with_images']} | "
            f"{row['with_external_non_document_links']} |"
        )
    lines.extend(["", "## Rejection reasons", ""])
    for reason, count in report["rejection_reasons"].items():
        lines.append(f"- `{reason}`: {count}")
    lines.extend(
        [
            "",
            "Image-bearing means at least one image reference occurs in the question, accepted answer, or linked documentation pages. The actual pixels must still be fetched and passed to the multimodal correctness judge before a case is accepted semantically.",
            "",
            "Other external links are retained as audit metadata and reject the case. Profile/self links and image-hosting URLs are ignored; successfully expanded linked-QA targets are represented by their appended answer text.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=_dataset_arg, required=True)
    parser.add_argument("--discussion-dir", action="append", type=_html_dir_arg, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    datasets = dict(args.dataset)
    discussion_dirs = dict(args.discussion_dir)
    missing = sorted(set(datasets) - set(discussion_dirs))
    if missing:
        raise SystemExit(f"missing --discussion-dir for: {', '.join(missing)}")
    rows, report = validate_datasets(datasets, discussion_dirs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "per_question.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_dir / "structurally_eligible.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            if row["status"] == "accepted_structurally":
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_dir / "rejected.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            if row["status"] == "rejected":
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(
        render_report(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
