"""Resolve, validate, cache, and label image evidence for DocsQA audits."""

from __future__ import annotations

import base64
import hashlib
import html
import io
import ipaddress
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError


FORMAT_TO_MIME = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
    "GIF": "image/gif",
}
MIME_TO_EXTENSION = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}
PROJECT_ALIASES = {
    "github_docs": "github-docs",
    "tailwind": "tailwind-css",
}


@dataclass(frozen=True)
class ImageRoute:
    url_prefix: str
    local_root: PurePosixPath


@dataclass(frozen=True)
class ProjectImageConfig:
    project: str
    source_root: Path
    image_base_url: str
    routes: tuple[ImageRoute, ...]


def _project(value: str) -> str:
    return PROJECT_ALIASES.get(value, value)


def _public_url(url: str) -> str:
    """Remove query credentials/fragments before persisting image provenance."""

    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _decoded_camo_target(url: str) -> str:
    if urlsplit(url).hostname != "camo.githubusercontent.com":
        return ""
    encoded = urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1]
    try:
        return bytes.fromhex(encoded).decode("utf-8", errors="replace")
    except ValueError:
        return ""


def _is_non_evidence_image(url: str, alt: str, title: str) -> bool:
    """Drop link-preview chrome and known documentation placeholders."""

    lowered = url.casefold().strip()
    decoded = _decoded_camo_target(url).casefold()
    label = f"{alt} {title}".casefold().strip()
    if lowered in {"...", "http://www.example.com/logo.jpg", "https://www.example.com/logo.jpg"}:
        return True
    if "favicon" in lowered or "favicon" in decoded or "fluidicon" in lowered:
        return True
    if lowered.endswith("/img/logo.svg") and not label:
        return True
    if lowered.endswith("/img/building.jpg") and not label:
        return True
    return False


def _safe_remote_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("image URL must be HTTP(S)")
    hostname = parsed.hostname.casefold()
    if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".local"):
        raise ValueError("local image hosts are forbidden")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return
    if not address.is_global:
        raise ValueError("non-public image address is forbidden")


def load_project_image_configs(
    config_path: Path,
    source_root: Path,
) -> dict[str, ProjectImageConfig]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    configs: dict[str, ProjectImageConfig] = {}
    for raw in payload.get("sources") or []:
        project = str(raw["id"])
        routes = tuple(
            ImageRoute(
                url_prefix=str(route["url_prefix"]),
                local_root=PurePosixPath(str(route["local_root"])),
            )
            for route in raw.get("image_routes") or []
        )
        configs[project] = ProjectImageConfig(
            project=project,
            source_root=(source_root / str(raw["checkout_dir"])).resolve(),
            image_base_url=str(raw["image_base_url"]),
            routes=routes,
        )
    return configs


def collect_image_references(
    question: dict[str, Any],
    corpus: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return stable, provenance-labelled image references for one QA package."""

    project = _project(str(question.get("project") or question.get("dataset") or ""))
    question_id = str(question["question_id"])
    references: list[dict[str, Any]] = []

    def add(role: str, images: Iterable[dict[str, Any]], doc_id: str | None = None) -> None:
        for position, image in enumerate(images, start=1):
            url = str(image.get("url") or "").strip()
            if not url:
                continue
            alt = str(image.get("alt") or "").strip()
            title = str(image.get("title") or "").strip()
            if _is_non_evidence_image(url, alt, title):
                continue
            source_path = None
            if doc_id and doc_id in corpus:
                document = corpus[doc_id]
                source_path = str(
                    document.get("repository_source_path")
                    or document.get("source_path")
                    or ""
                )
            identity = f"{question_id}\0{role}\0{doc_id or ''}\0{position}\0{url}"
            references.append(
                {
                    "image_id": "img-" + hashlib.sha256(identity.encode()).hexdigest()[:16],
                    "question_id": question_id,
                    "project": project,
                    "role": role,
                    "doc_id": doc_id,
                    "position": position,
                    "source_url": url,
                    "page_url": str(question.get("source_url") or ""),
                    "alt": alt,
                    "title": title,
                    "document_source_path": source_path,
                }
            )

    add("question", question.get("question_images") or [])
    add("accepted_answer", question.get("reference_answer_images") or [])
    for doc_id in question.get("qrel_ids") or []:
        add(
            "linked_document",
            (question.get("document_images") or {}).get(str(doc_id), []),
            str(doc_id),
        )
    return references


class ImageResolver:
    def __init__(
        self,
        configs: dict[str, ProjectImageConfig],
        cache_dir: Path,
        *,
        allow_remote: bool,
        max_bytes: int = 20 * 1024 * 1024,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.configs = configs
        self.cache_dir = cache_dir.resolve()
        self.allow_remote = allow_remote
        self.max_bytes = max_bytes
        self.timeout_seconds = timeout_seconds
        self._page_cache: dict[str, str] = {}
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _local_candidate(
        self, reference: dict[str, Any], config: ProjectImageConfig
    ) -> Path | None:
        value = str(reference["source_url"])
        if urlsplit(value).scheme:
            return None
        if value.startswith("/"):
            for route in config.routes:
                if not value.startswith(route.url_prefix):
                    continue
                suffix = value[len(route.url_prefix) :]
                candidate = (
                    config.source_root / route.local_root / PurePosixPath(suffix)
                ).resolve()
                try:
                    candidate.relative_to(config.source_root)
                except ValueError:
                    raise ValueError("image path escapes the pinned source root")
                if candidate.is_file():
                    return candidate
            return None
        document_source = str(reference.get("document_source_path") or "")
        if not document_source:
            return None
        candidate = (
            config.source_root
            / PurePosixPath(document_source).parent
            / PurePosixPath(value)
        ).resolve()
        try:
            candidate.relative_to(config.source_root)
        except ValueError:
            raise ValueError("relative image path escapes the pinned source root")
        return candidate if candidate.is_file() else None

    @staticmethod
    def _remote_url(reference: dict[str, Any], config: ProjectImageConfig) -> str:
        value = str(reference["source_url"])
        return value if urlsplit(value).scheme else urljoin(config.image_base_url, value)

    def _read_local(self, path: Path) -> bytes:
        size = path.stat().st_size
        if size > self.max_bytes:
            raise ValueError(f"image exceeds {self.max_bytes} bytes")
        return path.read_bytes()

    def _read_remote(self, url: str) -> tuple[bytes, str]:
        if not self.allow_remote:
            raise ValueError("remote download disabled and image is not cached/local")
        _safe_remote_url(url)
        request = Request(
            url,
            headers={"User-Agent": "docsqa-image-audit/1.0", "Accept": "image/*"},
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            final_url = response.geturl()
            _safe_remote_url(final_url)
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > self.max_bytes:
                raise ValueError(f"image exceeds {self.max_bytes} bytes")
            data = response.read(self.max_bytes + 1)
        if len(data) > self.max_bytes:
            raise ValueError(f"image exceeds {self.max_bytes} bytes")
        return data, final_url

    def _refresh_private_user_url(self, reference: dict[str, Any]) -> str | None:
        original = str(reference.get("source_url") or "")
        if urlsplit(original).hostname != "private-user-images.githubusercontent.com":
            return None
        page_url = str(reference.get("page_url") or "")
        if not page_url:
            return None
        _safe_remote_url(page_url)
        if page_url not in self._page_cache:
            request = Request(
                page_url,
                headers={"User-Agent": "docsqa-image-audit/1.0", "Accept": "text/html"},
            )
            with urlopen(request, timeout=self.timeout_seconds) as response:
                data = response.read(10 * 1024 * 1024 + 1)
            if len(data) > 10 * 1024 * 1024:
                raise ValueError("source page exceeds 10485760 bytes")
            self._page_cache[page_url] = html.unescape(data.decode("utf-8", errors="replace"))
        target = _public_url(original)
        matches = [
            value.rstrip("\\")
            for value in re.findall(
            r"https://private-user-images\.githubusercontent\.com/[^\s\"'<>]+",
            self._page_cache[page_url],
            )
        ]
        return next((value for value in matches if _public_url(value) == target), None)

    @staticmethod
    def _inspect(data: bytes) -> tuple[str, int, int]:
        try:
            with Image.open(io.BytesIO(data)) as image:
                image_format = str(image.format or "").upper()
                mime_type = FORMAT_TO_MIME.get(image_format)
                if not mime_type:
                    raise ValueError(f"unsupported image format: {image_format or 'unknown'}")
                if image_format == "GIF" and bool(getattr(image, "is_animated", False)):
                    raise ValueError("animated GIF is unsupported")
                width, height = image.size
                image.verify()
        except UnidentifiedImageError as error:
            raise ValueError("downloaded content is not a supported image") from error
        if width < 1 or height < 1:
            raise ValueError("image has invalid dimensions")
        return mime_type, int(width), int(height)

    def _cached(self, locator_key: str) -> Path | None:
        matches = list((self.cache_dir / locator_key[:2]).glob(locator_key + ".*"))
        return matches[0] if len(matches) == 1 else None

    def resolve(self, reference: dict[str, Any]) -> dict[str, Any]:
        project = _project(str(reference["project"]))
        config = self.configs.get(project)
        if config is None:
            return {**reference, "status": "failed", "error": "unknown_project"}
        local_path = self._local_candidate(reference, config)
        remote_url = self._remote_url(reference, config)
        stable_locator = (
            f"local:{local_path}"
            if local_path is not None
            else f"remote:{_public_url(remote_url)}"
        )
        locator_key = hashlib.sha256(stable_locator.encode()).hexdigest()
        try:
            cached = self._cached(locator_key)
            if cached is not None:
                data = self._read_local(cached)
                source_kind = "cache"
                resolved_url = _public_url(remote_url)
            elif local_path is not None:
                data = self._read_local(local_path)
                source_kind = "pinned_repository"
                resolved_url = None
            else:
                try:
                    data, final_url = self._read_remote(remote_url)
                except Exception as first_error:
                    refreshed = self._refresh_private_user_url(reference)
                    if not refreshed or refreshed == remote_url:
                        raise first_error
                    data, final_url = self._read_remote(refreshed)
                source_kind = "remote"
                resolved_url = _public_url(final_url)
            mime_type, width, height = self._inspect(data)
            extension = MIME_TO_EXTENSION[mime_type]
            destination = self.cache_dir / locator_key[:2] / f"{locator_key}{extension}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                destination.write_bytes(data)
            return {
                **reference,
                "source_url": _public_url(str(reference["source_url"])),
                "status": "resolved",
                "error": None,
                "source_kind": source_kind,
                "resolved_url": resolved_url,
                "cache_path": str(destination),
                "mime_type": mime_type,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
                "width": width,
                "height": height,
            }
        except Exception as error:
            message = str(error)
            if "remote download disabled" in message:
                error_code = "remote_unavailable_offline"
            elif "animated GIF" in message:
                error_code = "animated_gif_unsupported"
            elif "unsupported image format" in message or "not a supported image" in message:
                error_code = "unsupported_or_non_image"
            elif "exceeds" in message:
                error_code = "image_too_large"
            elif "forbidden" in message:
                error_code = "unsafe_image_url"
            elif type(error).__name__ == "HTTPError":
                error_code = "http_error"
            elif type(error).__name__ == "URLError":
                error_code = "network_error"
            else:
                error_code = "image_resolution_error"
            alt = str(reference.get("alt") or "").strip()
            generic_alt = alt.casefold() in {"", "image", "screenshot", "logo", "photo"}
            if (
                reference.get("role") == "linked_document"
                and error_code == "http_error"
                and len(alt) >= 20
                and not generic_alt
            ):
                return {
                    **reference,
                    "source_url": _public_url(str(reference["source_url"])),
                    "status": "text_fallback",
                    "error_code": error_code,
                    "error": f"{type(error).__name__}: {error}",
                    "fallback_kind": "pinned_document_alt_text",
                    "fallback_text": alt,
                    "pixel_verified": False,
                }
            return {
                **reference,
                "source_url": _public_url(str(reference["source_url"])),
                "status": "failed",
                "error_code": error_code,
                "error": f"{type(error).__name__}: {error}",
            }


def image_data_url(record: dict[str, Any]) -> str:
    if record.get("status") != "resolved":
        raise ValueError(f"image is unresolved: {record.get('image_id')}")
    data = Path(str(record["cache_path"])).read_bytes()
    if hashlib.sha256(data).hexdigest() != record.get("sha256"):
        raise ValueError(f"cached image hash mismatch: {record.get('image_id')}")
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{record['mime_type']};base64,{encoded}"
