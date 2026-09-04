from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlsplit

import yaml


LOCALE_RE = re.compile(r"^[a-z]{2}(?:-[a-z]{2})?$", re.IGNORECASE)
VERSION_PREFIX_RE = re.compile(
    r"^(?:enterprise-cloud|enterprise-server|free-pro-team|github-ae)(?:@[^/]+)?$",
    re.IGNORECASE,
)
DOCS_LINK_RE = re.compile(
    r"href=[\"'](https?://docs\.github\.com[^\"']+)", re.IGNORECASE
)
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]*\]\((/[^)\s]+)\)")
MARKDOWN_LINK_DETAIL_RE = re.compile(r"\[([^\]]*)\]\((/[^)\s]+)\)")
REUSABLE_RE = re.compile(r"{%\s*data\s+reusables\.([\w.-]+)\s*%}")
VARIABLE_RE = re.compile(r"{%\s*data\s+variables\.([\w.-]+)\s*%}")
HEADING_RE = re.compile(r"^(#{2,6})\s+(.+?)\s*$", re.MULTILINE)
URL_TEXT_RE = re.compile(r"https?://\S+", re.IGNORECASE)


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.block_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"p", "li", "br", "h1", "h2", "h3", "h4", "pre", "blockquote"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "pre", "blockquote"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def html_to_text(value: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(value)
    text = html.unescape("".join(parser.parts))
    text = URL_TEXT_RE.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def slugify_heading(value: str) -> str:
    value = re.sub(r"{%.*?%}", "", value)
    value = re.sub(r"<[^>]+>", "", value)
    value = re.sub(r"[`*_~]", "", value).strip().lower()
    value = re.sub(r"[^\w\s-]", "", value)
    return re.sub(r"[-\s]+", "-", value).strip("-")


def normalize_docs_path(url_or_path: str) -> tuple[str, str]:
    parsed = urlsplit(html.unescape(url_or_path))
    path = unquote(parsed.path or "/")
    parts = [part for part in path.split("/") if part]
    if parts and LOCALE_RE.match(parts[0]):
        parts.pop(0)
    if parts and VERSION_PREFIX_RE.match(parts[0]):
        parts.pop(0)
    path = "/" + "/".join(parts)
    path = re.sub(r"/+", "/", path).rstrip("/") or "/"
    return path.lower(), unquote(parsed.fragment).lower()


def _split_frontmatter(text: str) -> tuple[list[str], str]:
    if not text.startswith("---\n"):
        return [], text
    end = text.find("\n---\n", 4)
    if end < 0:
        return [], text
    return text[4:end].splitlines(), text[end + 5 :]


def _frontmatter(lines: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    active_list: str | None = None
    active_map: str | None = None
    for line in lines:
        scalar = re.match(r"^([A-Za-z][\w-]*):\s*(.*?)\s*$", line)
        if scalar:
            key, value = scalar.groups()
            active_list = key if value == "" else None
            active_map = key if value == "" else None
            result[key] = [] if value == "" else value.strip("'\"")
            continue
        list_item = re.match(r"^\s+-\s+(.*?)\s*$", line)
        if list_item and active_list:
            current = result.setdefault(active_list, [])
            if isinstance(current, list):
                current.append(list_item.group(1).strip("'\""))
            continue
        mapping = re.match(r"^\s+([\w-]+):\s*(.*?)\s*$", line)
        if mapping and active_map:
            if not isinstance(result.get(active_map), dict):
                result[active_map] = {}
            result[active_map][mapping.group(1)] = mapping.group(2).strip("'\"")
    return result


def _canonical_path(relative_path: Path) -> str:
    value = "/" + relative_path.with_suffix("").as_posix()
    if value.endswith("/index"):
        value = value[: -len("/index")]
    return value.lower() or "/"


def _flatten_variables(value: Any, prefix: str = "") -> dict[str, str]:
    output: dict[str, str] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten_variables(child, child_prefix))
    elif isinstance(value, (str, int, float, bool)):
        output[prefix] = str(value)
    return output


@dataclass
class Section:
    section_id: str
    heading: str
    anchor: str
    raw_text: str
    rendered_text: str
    reusable_ids: list[str] = field(default_factory=list)
    variant_conditioned: bool = False


@dataclass
class CorpusPage:
    doc_id: str
    source_path: str
    title: str
    short_title: str
    content_type: str
    versions: list[str]
    redirects: list[str]
    raw_text: str
    rendered_text: str
    reusable_ids: list[str]
    variant_conditioned: bool
    route: str
    sections: list[Section]
    outgoing_paths: list[str] = field(default_factory=list)
    outgoing_ids: list[str] = field(default_factory=list)
    link_edges: list[dict[str, str]] = field(default_factory=list)


@dataclass
class DiscussionQuestion:
    question_id: str
    source_url: str
    title: str
    query: str
    community_category: str
    accepted_answer_url: str
    docs_urls: list[str]
    qrel_ids: list[str]
    qrel_anchors: dict[str, list[str]]
    resolution_kinds: dict[str, str]
    url_resolutions: dict[str, dict[str, str]]
    intent_category: str
    evidence_category: str
    evidence_flags: list[str]
    split: str


class GitHubDocsCorpus:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.content_root = repo_root / "content"
        self.reusables_root = repo_root / "data" / "reusables"
        self.variables_root = repo_root / "data" / "variables"
        self.reusables = self._load_reusables()
        self.variables = self._load_variables()
        self.pages = self._load_pages()
        self.by_id = {page.doc_id: page for page in self.pages}
        self.path_lookup: dict[str, tuple[str, str]] = {}
        self.slug_lookup: dict[str, list[str]] = defaultdict(list)
        for page in self.pages:
            self.path_lookup[page.doc_id] = (page.doc_id, "canonical")
            for redirect in page.redirects:
                normalized, _ = normalize_docs_path(redirect)
                self.path_lookup[normalized] = (page.doc_id, "frontmatter_redirect")
            self.slug_lookup[page.doc_id.rsplit("/", 1)[-1]].append(page.doc_id)
        self._resolve_graph()

    def _load_reusables(self) -> dict[str, str]:
        output: dict[str, str] = {}
        if not self.reusables_root.exists():
            return output
        for path in self.reusables_root.rglob("*.md"):
            key = path.relative_to(self.reusables_root).with_suffix("").as_posix().replace("/", ".")
            output[key] = path.read_text(encoding="utf-8", errors="replace")
        return output

    def _load_variables(self) -> dict[str, str]:
        output: dict[str, str] = {}
        if not self.variables_root.exists():
            return output
        for path in self.variables_root.rglob("*.yml"):
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                continue
            prefix = path.relative_to(self.variables_root).with_suffix("").as_posix().replace("/", ".")
            output.update(_flatten_variables(data, prefix))
        return output

    def render(self, text: str, seen: frozenset[str] = frozenset()) -> str:
        def reusable_replace(match: re.Match[str]) -> str:
            reusable_id = match.group(1)
            if reusable_id in seen:
                return reusable_id.replace(".", " ")
            value = self.reusables.get(reusable_id)
            if value is None:
                return reusable_id.replace(".", " ")
            return self.render(value, seen | {reusable_id})

        def variable_replace(match: re.Match[str]) -> str:
            key = match.group(1)
            return self.variables.get(key, key.rsplit(".", 1)[-1].replace("_", " "))

        rendered = REUSABLE_RE.sub(reusable_replace, text)
        rendered = VARIABLE_RE.sub(variable_replace, rendered)
        rendered = re.sub(
            r"\bsecret_scanning_[0-9a-f]{20,}_[0-9a-z]{5,}\b",
            "secret_scanning_REDACTED_DUMMY_TOKEN",
            rendered,
        )
        rendered = re.sub(r"{%\s*(?:ifversion|elsif|else|endif).*?%}", " ", rendered)
        rendered = re.sub(r"{%.*?%}", " ", rendered)
        return rendered

    def _sections(self, doc_id: str, body: str) -> list[Section]:
        matches = list(HEADING_RE.finditer(body))
        if not matches:
            return []
        output: list[Section] = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
            raw = body[match.start() : end].strip()
            anchor = slugify_heading(match.group(2))
            output.append(
                Section(
                    section_id=f"{doc_id}#{anchor}",
                    heading=match.group(2).strip(),
                    anchor=anchor,
                    raw_text=raw,
                    rendered_text=self.render(raw),
                    reusable_ids=sorted(set(REUSABLE_RE.findall(raw))),
                    variant_conditioned="{% ifversion" in raw or "{% elsif" in raw,
                )
            )
        return output

    def _load_pages(self) -> list[CorpusPage]:
        pages: list[CorpusPage] = []
        for path in sorted(self.content_root.rglob("*.md")):
            relative = path.relative_to(self.content_root)
            text = path.read_text(encoding="utf-8", errors="replace")
            frontmatter_lines, body = _split_frontmatter(text)
            metadata = _frontmatter(frontmatter_lines)
            doc_id = _canonical_path(relative)
            versions = metadata.get("versions", {})
            if isinstance(versions, dict):
                version_names = sorted(versions)
            else:
                version_names = []
            redirects = metadata.get("redirect_from", [])
            if not isinstance(redirects, list):
                redirects = []
            parts = [part for part in relative.parts[:-1] if part]
            route = "/" + "/".join(parts[:2]) if parts else "/"
            pages.append(
                CorpusPage(
                    doc_id=doc_id,
                    source_path=str(relative),
                    title=str(metadata.get("title", relative.stem)),
                    short_title=str(metadata.get("shortTitle", "")),
                    content_type=str(metadata.get("contentType", "")),
                    versions=version_names,
                    redirects=[str(value) for value in redirects],
                    raw_text=body,
                    rendered_text=self.render(body),
                    reusable_ids=sorted(set(REUSABLE_RE.findall(body))),
                    variant_conditioned="{% ifversion" in body or "{% elsif" in body,
                    route=route,
                    sections=self._sections(doc_id, body),
                    outgoing_paths=sorted(set(MARKDOWN_LINK_RE.findall(body))),
                    link_edges=self._raw_link_edges(body),
                )
            )
        return pages

    def _raw_link_edges(self, body: str) -> list[dict[str, str]]:
        edges: list[dict[str, str]] = []
        active_section = ""
        for line in body.splitlines():
            heading = re.match(r"^#{2,6}\s+(.+?)\s*$", line)
            if heading:
                active_section = slugify_heading(heading.group(1))
            for match in MARKDOWN_LINK_DETAIL_RE.finditer(line):
                context = re.sub(r"\s+", " ", self.render(line)).strip()
                edges.append(
                    {
                        "target_path": match.group(2),
                        "anchor_text": self.render(match.group(1)).strip(),
                        "source_section": active_section,
                        "context": context[:600],
                    }
                )
        return edges

    def resolve(self, url_or_path: str, allow_slug_fallback: bool = True) -> tuple[str | None, str, str]:
        path, anchor = normalize_docs_path(url_or_path)
        direct = self.path_lookup.get(path)
        if direct:
            return direct[0], anchor, direct[1]
        if allow_slug_fallback:
            candidates = self.slug_lookup.get(path.rsplit("/", 1)[-1], [])
            if len(candidates) == 1:
                return candidates[0], anchor, "unique_slug"
        return None, anchor, "unresolved"

    def _resolve_graph(self) -> None:
        for page in self.pages:
            destinations: set[str] = set()
            resolved_edges: list[dict[str, str]] = []
            seen_edges: set[tuple[str, str, str]] = set()
            for edge in page.link_edges:
                target, target_anchor, _ = self.resolve(edge["target_path"])
                if not target or target == page.doc_id:
                    continue
                key = (target, edge["source_section"], edge["anchor_text"])
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                destinations.add(target)
                resolved_edges.append(
                    {
                        "target_id": target,
                        "target_anchor": target_anchor,
                        "anchor_text": edge["anchor_text"],
                        "source_section": edge["source_section"],
                        "context": edge["context"],
                    }
                )
            for raw_path in page.outgoing_paths:
                target, _, _ = self.resolve(raw_path)
                if target and target != page.doc_id:
                    destinations.add(target)
            page.outgoing_ids = sorted(destinations)
            page.link_edges = resolved_edges


def _extract_json_ld(text: str) -> dict[str, Any] | None:
    matches = re.findall(r'<script type="application/ld\+json">(.*?)</script>', text, re.DOTALL)
    for value in reversed(matches):
        try:
            parsed = json.loads(html.unescape(value))
        except json.JSONDecodeError:
            continue
        if parsed.get("@type") == "QAPage":
            return parsed
    return None


def _community_category(text: str) -> str:
    match = re.search(
        r'href="/orgs/community/discussions/categories/[^"]+">\s*([^<]+?)\s*</a>',
        text,
        re.DOTALL,
    )
    return html.unescape(match.group(1)).strip() if match else "Unknown"


def _intent(title: str, query: str, qrel_ids: list[str]) -> str:
    value = f"{title} {query}".lower()
    if any(token in value for token in ("billing", "charged", "subscription", "invoice", "privacy", "terms", "account")):
        return "policy_billing_account"
    if re.search(r"\b(error|failed|failure|unable|cannot|can't|doesn't|not working|404|403|502|bug)\b", value):
        return "troubleshooting"
    if re.search(r"\b(difference|why|what is|what are|understand|purpose)\b", value):
        return "explanation_comparison"
    if re.search(r"\b(possible|support(?:ed)?|limit|maximum|can i|could i|is there)\b", value):
        return "capability_limit"
    if re.search(r"\b(how|configure|setup|set up|create|add|enable|use|syntax|migrate|change)\b", value):
        return "how_to_configuration"
    if any(doc_id.startswith("/site-policy/") or doc_id.startswith("/billing/") for doc_id in qrel_ids):
        return "policy_billing_account"
    return "other_product_question"


def _section_for_anchor(page: CorpusPage, anchor: str) -> Section | None:
    if not anchor:
        return None
    exact = next((section for section in page.sections if section.anchor == anchor), None)
    if exact:
        return exact
    return next(
        (
            section
            for section in page.sections
            if anchor.startswith(section.anchor) or section.anchor.startswith(anchor)
        ),
        None,
    )


def _evidence_shape(
    corpus: GitHubDocsCorpus,
    qrel_ids: list[str],
    qrel_anchors: dict[str, list[str]],
    query: str,
) -> tuple[str, list[str]]:
    flags: set[str] = set()
    pages = [corpus.by_id[doc_id] for doc_id in qrel_ids]
    linked = any(
        right.doc_id in left.outgoing_ids or left.doc_id in right.outgoing_ids
        for left in pages
        for right in pages
        if left.doc_id != right.doc_id
    )
    for page in pages:
        anchors = qrel_anchors.get(page.doc_id, [])
        selected = [_section_for_anchor(page, anchor) for anchor in anchors if anchor]
        selected = [section for section in selected if section is not None]
        if any(section.reusable_ids for section in selected):
            flags.add("anchor_uses_reusable")
        elif not anchors and page.reusable_ids:
            flags.add("page_uses_reusable")
        if any(section.variant_conditioned for section in selected):
            flags.add("anchor_is_variant_conditioned")
        elif not anchors and page.variant_conditioned:
            flags.add("page_is_variant_conditioned")
    variant_cue = bool(
        re.search(
            r"\b(enterprise|server|cloud|version|plan|free|team|organization|personal account|macos|windows|linux)\b",
            query,
            re.IGNORECASE,
        )
    )
    if variant_cue:
        flags.add("query_has_variant_cue")
    if len(qrel_ids) > 1:
        category = "multi_page_linked" if linked else "multi_page_dispersed"
    elif "anchor_uses_reusable" in flags:
        category = "single_page_composed_anchor"
    elif "anchor_is_variant_conditioned" in flags and variant_cue:
        category = "single_page_variant_anchor"
    elif qrel_anchors.get(qrel_ids[0]):
        category = "single_page_section"
    else:
        category = "single_page_direct"
    return category, sorted(flags)


def _split(question_id: str, evidence_category: str) -> str:
    digest = hashlib.sha256(f"github-docs-v1:{evidence_category}:{question_id}".encode()).digest()
    return "dev" if int.from_bytes(digest[:4], "big") % 4 == 0 else "test"


def parse_discussion(path: Path, corpus: GitHubDocsCorpus) -> DiscussionQuestion | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    data = _extract_json_ld(text)
    if not data:
        return None
    question = data.get("mainEntity") or {}
    answer = question.get("acceptedAnswer") or {}
    raw_answer = str(answer.get("text", ""))
    docs_urls = sorted(set(DOCS_LINK_RE.findall(raw_answer)))
    if not docs_urls:
        return None
    qrel_ids: set[str] = set()
    anchors: dict[str, set[str]] = defaultdict(set)
    resolution_kinds: dict[str, str] = {}
    url_resolutions: dict[str, dict[str, str]] = {}
    for url in docs_urls:
        doc_id, anchor, kind = corpus.resolve(url)
        resolution_kinds[url] = kind
        url_resolutions[url] = {
            "doc_id": doc_id or "",
            "anchor": anchor,
            "resolution_kind": kind,
        }
        if doc_id:
            qrel_ids.add(doc_id)
            if anchor:
                anchors[doc_id].add(anchor)
    if not qrel_ids:
        return None
    title = html.unescape(str(question.get("name", ""))).strip()
    body = html_to_text(str(question.get("text", "")))
    body = re.sub(r"(?im)^(?:discussion type|feature/topic area|body)\s*$", "", body)
    body = re.sub(r"\n\s*\n+", "\n", body).strip()
    query = f"{title}\n{body}".strip()[:6000]
    ordered_qrels = sorted(qrel_ids)
    ordered_anchors = {key: sorted(value) for key, value in sorted(anchors.items())}
    evidence_category, flags = _evidence_shape(
        corpus, ordered_qrels, ordered_anchors, query
    )
    question_id = path.stem
    return DiscussionQuestion(
        question_id=question_id,
        source_url=f"https://github.com/orgs/community/discussions/{question_id}",
        title=title,
        query=query,
        community_category=_community_category(text),
        accepted_answer_url=str(answer.get("url", "")),
        docs_urls=docs_urls,
        qrel_ids=ordered_qrels,
        qrel_anchors=ordered_anchors,
        resolution_kinds=resolution_kinds,
        url_resolutions=url_resolutions,
        intent_category=_intent(title, query, ordered_qrels),
        evidence_category=evidence_category,
        evidence_flags=flags,
        split=_split(question_id, evidence_category),
    )


def build_questions(discussions_dir: Path, corpus: GitHubDocsCorpus) -> list[DiscussionQuestion]:
    questions: list[DiscussionQuestion] = []
    for path in sorted(discussions_dir.glob("*.html"), key=lambda value: int(value.stem)):
        parsed = parse_discussion(path, corpus)
        if parsed:
            questions.append(parsed)
    return questions


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            encoded = json.dumps(row, ensure_ascii=False)
            # JSON permits these Unicode separators inside strings, but Python's
            # splitlines() treats them as record boundaries. Escape them so one
            # JSON object is always exactly one physical JSONL line.
            encoded = (
                encoded.replace("\u0085", "\\u0085")
                .replace("\u2028", "\\u2028")
                .replace("\u2029", "\\u2029")
            )
            handle.write(encoded + "\n")


def prepare(
    repo_root: Path,
    discussions_dir: Path,
    output_dir: Path,
    revision: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = GitHubDocsCorpus(repo_root)
    questions = build_questions(discussions_dir, corpus)

    def question_record(question: DiscussionQuestion) -> dict[str, Any]:
        row = asdict(question)
        # URL-level diagnostics are used during construction, but the released
        # source package keeps only the canonical resolution map and factual
        # evidence-structure slice.
        row.pop("url_resolutions", None)
        row["qrel_count"] = len(question.qrel_ids)
        row["evidence_structure"] = (
            "single"
            if len(question.qrel_ids) <= 1
            else "linked"
            if question.evidence_category == "multi_page_linked"
            else "dispersed"
        )
        return row

    _write_jsonl(
        output_dir / "questions.jsonl",
        (question_record(question) for question in questions),
    )
    _write_jsonl(
        output_dir / "corpus.jsonl",
        (
            {
                "doc_id": page.doc_id,
                "source_path": page.source_path,
                "title": page.title,
                "short_title": page.short_title,
                "content_type": page.content_type,
                "versions": page.versions,
                "route": page.route,
                "rendered_text": page.rendered_text,
                "outgoing_ids": page.outgoing_ids,
                "link_edges": page.link_edges,
                "reusable_ids": page.reusable_ids,
                "variant_conditioned": page.variant_conditioned,
            }
            for page in corpus.pages
        ),
    )
    unresolved = Counter(
        kind for question in questions for kind in question.resolution_kinds.values()
    )
    manifest = {
        "benchmark": "GitHub Docs real support questions v1",
        "corpus_revision": revision,
        "documents": len(corpus.pages),
        "questions": len(questions),
        "questions_by_split": Counter(question.split for question in questions),
        "questions_by_intent": Counter(question.intent_category for question in questions),
        "questions_by_evidence_category": Counter(
            question.evidence_category for question in questions
        ),
        "questions_by_community_category": Counter(
            question.community_category for question in questions
        ),
        "docs_url_resolution_kinds": unresolved,
        "query_leakage_control": "All visible HTTP(S) URLs are removed from question text.",
        "qrel_provenance": "Distinct docs.github.com URLs in the accepted answer only.",
        "evaluation_unit": "canonical documentation page; URL anchors retained for diagnostics",
    }
    serializable = json.loads(json.dumps(manifest, default=dict))
    (output_dir / "manifest.json").write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return serializable


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the GitHub Docs real-question benchmark")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--discussions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    manifest = prepare(
        args.repo_root,
        args.discussions_dir,
        args.output_dir,
        args.revision,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
