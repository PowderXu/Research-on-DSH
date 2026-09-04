"""Build DocsQA corpora from pinned Markdown/MDX repos and accepted Discussions.

The accepted answer is retained as evaluation-only reference data. It is never
inserted into the searchable corpus or the agent prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import Request, urlopen

import yaml

from .build import html_to_text


MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)\s]+)(?:\s+[^)]*)?\)")
MDX_LINK_RE = re.compile(r"(?:href|to)=[\"']([^\"']+)[\"']")
HEADING_RE = re.compile(r"^(#{2,6})\s+(.+?)\s*$", re.MULTILINE)
TAILWIND_TITLE_RE = re.compile(r"export\s+const\s+title\s*=\s*[\"']([^\"']+)")
SCRIPT_RE = re.compile(
    r"<script[^>]+type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>",
    re.IGNORECASE | re.DOTALL,
)
HTML_LINK_RE = re.compile(r"<a\b[^>]*?\bhref=[\"']([^\"']+)[\"']", re.IGNORECASE)
HTML_IMAGE_RE = re.compile(r"<img\b([^>]*)>", re.IGNORECASE)
HTML_ATTR_RE = re.compile(r"([\w:-]+)=[\"']([^\"']*)[\"']", re.IGNORECASE)


@dataclass(frozen=True)
class CorpusConfig:
    name: str
    repo_root: Path
    content_root: Path
    docs_hosts: tuple[str, ...]
    route_kind: str
    revision: str


CONFIG_DEFAULTS = {
    "tailwind": {
        "repo": "results/cache/docsqa-source-repos/tailwindcss",
        "content": "src/docs",
        "hosts": ("tailwindcss.com",),
        "route_kind": "tailwind",
    },
    "prisma": {
        "repo": "results/cache/docsqa-source-repos/prisma",
        "content": "apps/docs/content/docs",
        "hosts": ("www.prisma.io", "prisma.io"),
        "route_kind": "prisma",
    },
    "supabase": {
        "repo": "results/cache/docsqa-source-repos/supabase",
        "content": "apps/docs/content",
        "hosts": ("supabase.com",),
        "route_kind": "supabase",
    },
}


def _normalize_path(value: str) -> str:
    path = unquote(urlsplit(html.unescape(value)).path or "/")
    path = re.sub(r"/+", "/", "/" + path.lstrip("/"))
    return path.rstrip("/").casefold() or "/"


def _frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text
    try:
        metadata = yaml.safe_load(text[4:end]) or {}
    except yaml.YAMLError:
        metadata = {}
    return (metadata if isinstance(metadata, dict) else {}), text[end + 5 :]


def _clean_title(value: str) -> str:
    value = re.sub(r"[`*_{}<>]", "", value)
    return re.sub(r"\s+", " ", value).strip()


def _path_route(relative: Path, metadata: dict[str, Any], kind: str) -> str:
    if kind == "tailwind":
        return _normalize_path(f"/docs/{relative.stem}")
    if kind == "prisma":
        declared = metadata.get("url")
        if isinstance(declared, str) and declared.strip():
            return _normalize_path("/docs/" + declared.strip().lstrip("/"))
        value = relative.with_suffix("").as_posix()
        if value.endswith("/index"):
            value = value[: -len("/index")]
        return _normalize_path("/docs/" + value)
    value = relative.with_suffix("").as_posix()
    if value.endswith("/index") or value.endswith("/overview"):
        value = value.rsplit("/", 1)[0]
    return _normalize_path("/docs/" + value)


def _title(relative: Path, metadata: dict[str, Any], body: str, kind: str) -> str:
    declared = metadata.get("title")
    if isinstance(declared, str) and declared.strip():
        return _clean_title(declared)
    if kind == "tailwind":
        matched = TAILWIND_TITLE_RE.search(body)
        if matched:
            return _clean_title(matched.group(1))
    heading = re.search(r"^#\s+(.+?)\s*$", body, re.MULTILINE)
    return _clean_title(heading.group(1) if heading else relative.stem.replace("-", " "))


def _render_mdx(body: str) -> str:
    body = re.sub(r"^import\s+.*?;\s*$", " ", body, flags=re.MULTILINE)
    body = re.sub(r"^export\s+const\s+(?:title|description)\s*=.*?;\s*$", " ", body, flags=re.MULTILINE)
    body = re.sub(r"\{\/\*.*?\*\/\}", " ", body, flags=re.DOTALL)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def _candidate_link_paths(raw: str, source_route: str, allowed_hosts: set[str]) -> list[str]:
    parsed = urlsplit(html.unescape(raw))
    if parsed.scheme in {"http", "https"}:
        if (parsed.hostname or "").casefold() not in allowed_hosts:
            return []
        return [_normalize_path(raw)]
    if raw.startswith(("#", "mailto:", "javascript:")):
        return []
    if raw.startswith("/"):
        return [_normalize_path(raw)]
    base = source_route.rsplit("/", 1)[0] + "/"
    joined = urljoin("https://docs.invalid" + base, raw)
    return [_normalize_path(joined)]


def build_corpus(config: CorpusConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    duplicates: dict[str, list[str]] = {}
    by_route: dict[str, dict[str, Any]] = {}
    for path in sorted(config.content_root.rglob("*")):
        if path.suffix.casefold() not in {".md", ".mdx"} or not path.is_file():
            continue
        relative = path.relative_to(config.content_root)
        if any(part.startswith("_") for part in relative.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        metadata, body = _frontmatter(text)
        route = _path_route(relative, metadata, config.route_kind)
        title = _title(relative, metadata, body, config.route_kind)
        row = {
            "doc_id": route,
            "source_path": relative.as_posix(),
            "title": title,
            "short_title": "",
            "content_type": "documentation",
            "versions": [],
            "redirects": [],
            "raw_text": body,
            "rendered_text": _render_mdx(body),
            "reusable_ids": [],
            "variant_conditioned": False,
            "route": "/" + "/".join([part for part in route.split("/") if part][:2]),
            "sections": [],
            "outgoing_paths": [],
            "outgoing_ids": [],
            "link_edges": [],
        }
        if route in by_route:
            duplicates.setdefault(route, [by_route[route]["source_path"]]).append(relative.as_posix())
            # Prefer the unversioned/current path, then the shorter source path.
            old = by_route[route]
            candidates = sorted(
                (old, row),
                key=lambda value: (
                    bool(re.search(r"/(?:v\d+|legacy)/", "/" + value["source_path"])),
                    len(value["source_path"]),
                ),
            )
            by_route[route] = candidates[0]
        else:
            by_route[route] = row
    rows = [by_route[key] for key in sorted(by_route)]
    by_id = {row["doc_id"]: row for row in rows}
    allowed_hosts = {host.casefold() for host in config.docs_hosts}
    for row in rows:
        edges: list[dict[str, str]] = []
        body = str(row["raw_text"])
        current_section = ""
        headings = [(match.start(), _clean_title(match.group(2))) for match in HEADING_RE.finditer(body)]
        links = [(m.start(), m.group(1), m.group(2)) for m in MARKDOWN_LINK_RE.finditer(body)]
        links += [(m.start(), "", m.group(1)) for m in MDX_LINK_RE.finditer(body)]
        for position, anchor, raw_target in sorted(links):
            for heading_position, heading in headings:
                if heading_position > position:
                    break
                current_section = heading
            for target in _candidate_link_paths(raw_target, row["doc_id"], allowed_hosts):
                if config.route_kind == "prisma" and not target.startswith("/docs/"):
                    target = _normalize_path("/docs" + target)
                if target not in by_id or target == row["doc_id"]:
                    continue
                context = body[max(0, position - 160) : position + 240]
                edges.append(
                    {
                        "target_id": target,
                        "source_section": current_section,
                        "anchor_text": _clean_title(anchor),
                        "context": re.sub(r"\s+", " ", context).strip(),
                        "target_anchor": unquote(urlsplit(raw_target).fragment).casefold(),
                    }
                )
        unique = {(edge["target_id"], edge["anchor_text"], edge["source_section"]): edge for edge in edges}
        row["link_edges"] = list(unique.values())
        row["outgoing_ids"] = sorted({edge["target_id"] for edge in unique.values()})
        row["outgoing_paths"] = list(row["outgoing_ids"])
    stats = {
        "documents": len(rows),
        "markdown_link_edges": sum(len(row["link_edges"]) for row in rows),
        "duplicate_routes": duplicates,
    }
    return rows, stats


def _walk_json(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_json(child)


def _html_images(value: str) -> list[dict[str, str]]:
    images: list[dict[str, str]] = []
    for match in HTML_IMAGE_RE.finditer(value):
        attrs = {key.casefold(): html.unescape(raw) for key, raw in HTML_ATTR_RE.findall(match.group(1))}
        if attrs.get("src"):
            images.append(
                {
                    "url": attrs["src"],
                    "alt": attrs.get("alt", "").strip(),
                    "title": attrs.get("title", "").strip(),
                }
            )
    unique = {(item["url"], item["alt"], item["title"]): item for item in images}
    return list(unique.values())


def parse_discussion_html(page_html: str) -> dict[str, Any]:
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
            if answer_html:
                return {
                    "title": html_to_text(str(question.get("name") or "")),
                    "question": html_to_text(question_html),
                    "accepted_answer": html_to_text(answer_html),
                    "accepted_answer_url": str(answer.get("url") or ""),
                    "question_images": _html_images(question_html),
                    "accepted_answer_images": _html_images(answer_html),
                    "accepted_answer_links": sorted(
                        set(html.unescape(url) for url in HTML_LINK_RE.findall(answer_html))
                    ),
                    "accepted_answer_html": answer_html,
                }
    raise ValueError("no QAPage with an accepted answer was found")


def _fetch_one(record: dict[str, Any], cache_dir: Path, refresh: bool) -> dict[str, Any]:
    number = str(record["discussion_number"])
    cache = cache_dir / f"{number}.html"
    if refresh or not cache.exists():
        request = Request(
            str(record["source_url"]),
            headers={"User-Agent": "docsqa-benchmark/0.3 (+public benchmark construction)"},
        )
        with urlopen(request, timeout=30) as response:
            payload = response.read().decode("utf-8", errors="replace")
        cache.write_text(payload, encoding="utf-8")
        time.sleep(0.05)
    parsed = parse_discussion_html(cache.read_text(encoding="utf-8", errors="replace"))
    return {**record, **parsed}


def fetch_discussions(
    records: list[dict[str, Any]], cache_dir: Path, workers: int, refresh: bool
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_fetch_one, row, cache_dir, refresh): row for row in records}
        for future in as_completed(futures):
            source = futures[future]
            try:
                rows.append(future.result())
            except Exception as error:  # Public pages can disappear during construction.
                failures.append({"source_url": str(source.get("source_url")), "error": str(error)})
    return sorted(rows, key=lambda row: int(row["discussion_number"])), failures


def _is_exact_docs_url(url: str, hosts: set[str]) -> bool:
    parsed = urlsplit(url)
    path = _normalize_path(url)
    return (
        (parsed.hostname or "").casefold() in hosts
        and (path == "/docs" or path.startswith("/docs/"))
    )


def resolve_live_routes(
    urls: Iterable[str],
    *,
    local_routes: set[str],
    cache_path: Path,
    workers: int,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Resolve historical docs redirects without fuzzy qrel matching."""

    cache: dict[str, str] = {}
    if cache_path.exists():
        value = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            cache = {str(key): str(route) for key, route in value.items()}
    pending = sorted(
        {
            str(url)
            for url in urls
            if _normalize_path(str(url)) not in local_routes and str(url) not in cache
        }
    )
    failures: list[dict[str, str]] = []

    def resolve(url: str) -> tuple[str, str]:
        headers = {"User-Agent": "docsqa-benchmark/0.3 (+public benchmark construction)"}
        try:
            request = Request(url, headers=headers, method="HEAD")
            with urlopen(request, timeout=20) as response:
                return url, _normalize_path(response.geturl())
        except Exception as first_error:
            try:
                request = Request(url, headers=headers)
                with urlopen(request, timeout=30) as response:
                    return url, _normalize_path(response.geturl())
            except Exception:
                completed = subprocess.run(
                    [
                        "curl",
                        "--location",
                        "--silent",
                        "--show-error",
                        "--output",
                        "/dev/null",
                        "--max-time",
                        "30",
                        "--write-out",
                        "%{url_effective}",
                        url,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=35,
                )
                if completed.returncode or not completed.stdout.strip():
                    raise first_error
                return url, _normalize_path(completed.stdout.strip())

    with ThreadPoolExecutor(max_workers=min(workers, 4)) as executor:
        futures = {executor.submit(resolve, url): url for url in pending}
        for future in as_completed(futures):
            url = futures[future]
            try:
                original, final_route = future.result()
                cache[original] = final_route
            except Exception as error:
                failures.append({"url": url, "error": str(error)})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    aliases = {
        _normalize_path(url): final_route
        for url, final_route in cache.items()
        if final_route in local_routes
    }
    return aliases, failures


def _intent(query: str) -> str:
    value = query.casefold()
    if any(token in value for token in ("error", "fail", "not work", "cannot", "can't", "issue")):
        return "troubleshooting"
    if any(token in value for token in ("how ", "how do", "setup", "configure", "implement")):
        return "how_to"
    if any(token in value for token in ("difference", "compare", " vs ")):
        return "comparison"
    return "concept"


def build_questions(
    config: CorpusConfig,
    corpus_rows: list[dict[str, Any]],
    discussions: list[dict[str, Any]],
    live_route_aliases: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id = {row["doc_id"]: row for row in corpus_rows}
    hosts = {host.casefold() for host in config.docs_hosts}
    questions: list[dict[str, Any]] = []
    unresolved: list[dict[str, str]] = []
    excluded_unresolved_internal_answers: list[dict[str, Any]] = []
    excluded_high_link_answers: list[dict[str, Any]] = []
    exact_link_count = 0
    for row in discussions:
        qrels: list[str] = []
        urls: list[str] = []
        row_unresolved: list[str] = []
        resolution_kinds: dict[str, str] = {}
        for url in row.get("docs_host_links") or []:
            if not _is_exact_docs_url(str(url), hosts):
                continue
            urls.append(str(url))
            exact_link_count += 1
            path = _normalize_path(str(url))
            candidates = [path]
            if config.route_kind == "prisma" and not path.startswith("/docs/"):
                candidates.append(_normalize_path("/docs" + path))
            resolved = next((candidate for candidate in candidates if candidate in by_id), None)
            resolution_kind = "canonical_path"
            if resolved is None:
                resolved = (live_route_aliases or {}).get(path)
                resolution_kind = "live_http_redirect"
            if resolved:
                qrels.append(resolved)
                resolution_kinds[str(url)] = resolution_kind
            else:
                unresolved.append({"discussion": str(row["discussion_number"]), "url": str(url), "path": path})
                row_unresolved.append(str(url))
        qrels = list(dict.fromkeys(qrels))
        question_text = str(row.get("question") or "").strip()
        title = str(row.get("title") or "").strip()
        query = f"{title}\n\n{question_text}".strip()
        if row_unresolved:
            excluded_unresolved_internal_answers.append(
                {
                    "discussion": str(row["discussion_number"]),
                    "urls": sorted(set(row_unresolved)),
                }
            )
        if len(qrels) > 10:
            excluded_high_link_answers.append(
                {"discussion": str(row["discussion_number"]), "qrel_count": len(qrels)}
            )
            continue
        if not query or not str(row.get("accepted_answer") or "").strip():
            continue
        linked = False
        if len(qrels) > 1:
            relevant = set(qrels)
            linked = any(
                target in relevant
                for source in qrels
                for target in by_id[source].get("outgoing_ids", [])
                if target != source
            )
        evidence_structure = "single" if len(qrels) == 1 else "linked" if linked else "dispersed"
        digest = int(hashlib.sha256(f"{config.name}:{row['discussion_number']}".encode()).hexdigest()[:8], 16)
        split = "dev" if digest % 5 == 0 else "test"
        questions.append(
            {
                "question_id": f"{config.name}-{row['discussion_number']}",
                "dataset": config.name,
                "source_url": row["source_url"],
                "accepted_answer_url": row["accepted_answer_url"],
                "title": title,
                "query": query,
                "reference_answer": row["accepted_answer"],
                "question_images": row.get("question_images") or [],
                "reference_answer_images": row.get("accepted_answer_images") or [],
                "reference_answer_links": row.get("accepted_answer_links") or [],
                "docs_urls": urls,
                "qrel_ids": qrels,
                "qrel_count": len(qrels),
                "resolution_kinds": resolution_kinds,
                "intent_category": _intent(query),
                "evidence_category": "single_page" if len(qrels) == 1 else "multi_page_linked" if linked else "multi_page_dispersed",
                "evidence_structure": evidence_structure,
                "split": split,
            }
        )
    return questions, {
        "exact_host_links": exact_link_count,
        "resolved_questions": len(questions),
        "unresolved_links": unresolved,
        "excluded_unresolved_internal_answers": excluded_unresolved_internal_answers,
        "excluded_high_link_answers": excluded_high_link_answers,
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            encoded = json.dumps(row, ensure_ascii=False)
            encoded = (
                encoded.replace("\u0085", "\\u0085")
                .replace("\u2028", "\\u2028")
                .replace("\u2029", "\\u2029")
            )
            handle.write(encoded + "\n")


def write_dataset(
    output: Path,
    config: CorpusConfig,
    corpus: list[dict[str, Any]],
    questions: list[dict[str, Any]],
    corpus_stats: dict[str, Any],
    question_stats: dict[str, Any],
    fetch_failures: list[dict[str, str]],
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "corpus.jsonl", corpus)
    _write_jsonl(output / "questions.jsonl", questions)
    split_dir = output / "splits"
    split_dir.mkdir(exist_ok=True)
    for name, predicate in {
        "validation": lambda row: row["split"] == "dev",
        "test": lambda row: row["split"] == "test",
        "train": lambda row: False,
    }.items():
        (split_dir / f"{name}.json").write_text(
            json.dumps([row for row in questions if predicate(row)], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    manifest = {
        "dataset": config.name,
        "source_revision": config.revision,
        "corpus_revision": config.revision,
        "repo_root": str(config.repo_root),
        "content_root": str(config.content_root),
        "docs_hosts": list(config.docs_hosts),
        "documents": len(corpus),
        "questions": len(questions),
        "splits": {
            "dev": sum(row["split"] == "dev" for row in questions),
            "test": sum(row["split"] == "test" for row in questions),
        },
        "qrel_count_distribution": {
            str(count): sum(row["qrel_count"] == count for row in questions)
            for count in sorted({row["qrel_count"] for row in questions})
        },
        "corpus_stats": corpus_stats,
        "question_stats": question_stats,
        "fetch_failures": fetch_failures,
        "validity_notes": [
            "Reference answers and qrels come from accepted public GitHub Discussion answers.",
            "Only exact documentation host and canonical route matches become qrels; unresolved links are excluded, not guessed.",
            "Any accepted answer containing an unresolved internal documentation link is rejected as a complete QA case.",
            "The pinned current docs can drift from historical accepted answers.",
            "Accepted-answer text is evaluation-only and is not part of corpus.jsonl or agent prompts.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(CONFIG_DEFAULTS), required=True)
    parser.add_argument(
        "--source-file",
        "--yield-file",
        dest="source_file",
        type=Path,
        required=True,
        help=(
            "Frozen JSONL Discussion manifest. Rows may contain only project, "
            "discussion_number, and source_url; documentation links are derived "
            "from the accepted answer after download. --yield-file is retained "
            "as a compatibility alias."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    default = CONFIG_DEFAULTS[args.dataset]
    repo_root = project_root / str(default["repo"])
    content_root = repo_root / str(default["content"])
    if not content_root.is_dir():
        raise SystemExit(f"missing pinned content root: {content_root}")
    revision = __import__("subprocess").check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()
    config = CorpusConfig(
        args.dataset,
        repo_root,
        content_root,
        tuple(default["hosts"]),
        str(default["route_kind"]),
        revision,
    )
    candidates = [
        json.loads(line)
        for line in args.source_file.read_text(encoding="utf-8").splitlines()
        if line
    ]
    candidates = [
        row
        for row in candidates
        if str(row.get("project") or row.get("dataset") or args.dataset)
        in {args.dataset, config.name}
    ]
    hosts = {host.casefold() for host in config.docs_hosts}
    corpus, corpus_stats = build_corpus(config)
    discussions, failures = fetch_discussions(
        candidates,
        project_root / "results/cache/discussions" / args.dataset,
        args.workers,
        args.refresh,
    )
    for row in discussions:
        if not row.get("docs_host_links"):
            row["docs_host_links"] = [
                str(url)
                for url in row.get("accepted_answer_links") or []
                if _is_exact_docs_url(str(url), hosts)
            ]
    doc_urls = [
        str(url)
        for row in discussions
        for url in row.get("docs_host_links") or []
        if _is_exact_docs_url(str(url), hosts)
    ]
    live_aliases, redirect_failures = resolve_live_routes(
        doc_urls,
        local_routes={str(row["doc_id"]) for row in corpus},
        cache_path=project_root / "results/cache/discussions" / args.dataset / "redirects.json",
        workers=args.workers,
    )
    questions, question_stats = build_questions(config, corpus, discussions, live_aliases)
    question_stats["live_redirect_aliases"] = len(live_aliases)
    question_stats["redirect_failures"] = redirect_failures
    manifest = write_dataset(
        args.output_dir, config, corpus, questions, corpus_stats, question_stats, failures
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
