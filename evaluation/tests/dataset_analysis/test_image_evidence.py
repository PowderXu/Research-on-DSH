from __future__ import annotations

import base64
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError

from PIL import Image

from dataset_analysis.image_evidence import (
    ImageResolver,
    ImageRoute,
    ProjectImageConfig,
    collect_image_references,
    image_data_url,
)


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 8), color=(20, 40, 60)).save(path, format="PNG")


def test_collect_image_references_preserves_role_and_document_provenance() -> None:
    question = {
        "question_id": "github-docs::1",
        "project": "github-docs",
        "question_images": [{"url": "https://example.test/q.png", "alt": "error"}],
        "reference_answer_images": [{"url": "https://example.test/a.png"}],
        "qrel_ids": ["github-docs::/page"],
        "document_images": {
            "github-docs::/page": [{"url": "/assets/result.png", "alt": "result"}]
        },
    }
    corpus = {
        "github-docs::/page": {
            "repository_source_path": "content/page.md",
        }
    }

    rows = collect_image_references(question, corpus)

    assert [row["role"] for row in rows] == [
        "question",
        "accepted_answer",
        "linked_document",
    ]
    assert rows[-1]["doc_id"] == "github-docs::/page"
    assert len({row["image_id"] for row in rows}) == 3


def test_collect_image_references_drops_link_preview_chrome() -> None:
    favicon = (
        "https://camo.githubusercontent.com/hash/"
        "68747470733a2f2f646f63732e6769746875622e636f6d2f6173736574732f696d616765732f736974652f66617669636f6e2e737667"
    )
    question = {
        "question_id": "github-docs::1",
        "project": "github-docs",
        "source_url": "https://github.com/orgs/community/discussions/1",
        "question_images": [{"url": favicon, "alt": ""}],
        "reference_answer_images": [{"url": "https://example.test/answer.png"}],
        "qrel_ids": [],
        "document_images": {},
    }
    rows = collect_image_references(question, {})
    assert len(rows) == 1
    assert rows[0]["role"] == "accepted_answer"
    assert rows[0]["page_url"] == question["source_url"]


def test_resolver_uses_pinned_asset_and_builds_hash_checked_data_url(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    image_path = source_root / "assets/result.png"
    _png(image_path)
    config = ProjectImageConfig(
        project="github-docs",
        source_root=source_root,
        image_base_url="https://docs.github.com",
        routes=(ImageRoute("/assets/", PurePosixPath("assets")),),
    )
    resolver = ImageResolver(
        {"github-docs": config}, tmp_path / "cache", allow_remote=False
    )
    record = resolver.resolve(
        {
            "image_id": "img-1",
            "question_id": "github-docs::1",
            "project": "github-docs",
            "role": "linked_document",
            "doc_id": "github-docs::/page",
            "position": 1,
            "source_url": "/assets/result.png",
            "alt": "result",
            "title": "",
            "document_source_path": "content/page.md",
        }
    )

    assert record["status"] == "resolved"
    assert record["source_kind"] == "pinned_repository"
    assert record["mime_type"] == "image/png"
    assert (record["width"], record["height"]) == (12, 8)
    prefix, encoded = image_data_url(record).split(",", 1)
    assert prefix == "data:image/png;base64"
    assert base64.b64decode(encoded).startswith(b"\x89PNG")


def test_resolver_fails_closed_when_remote_image_is_unavailable_offline(
    tmp_path: Path,
) -> None:
    config = ProjectImageConfig(
        project="prisma",
        source_root=tmp_path / "source",
        image_base_url="https://www.prisma.io",
        routes=(),
    )
    resolver = ImageResolver({"prisma": config}, tmp_path / "cache", allow_remote=False)
    record = resolver.resolve(
        {
            "image_id": "img-2",
            "question_id": "prisma::1",
            "project": "prisma",
            "role": "linked_document",
            "doc_id": "prisma::/docs",
            "position": 1,
            "source_url": "/img/missing.png",
            "alt": "",
            "title": "",
            "document_source_path": "apps/docs/content/docs/index.mdx",
        }
    )

    assert record["status"] == "failed"
    assert record["error_code"] == "remote_unavailable_offline"
    assert "remote download disabled" in record["error"]


def test_private_image_refresh_strips_json_escape_suffix(tmp_path: Path) -> None:
    resolver = ImageResolver({}, tmp_path / "cache", allow_remote=True)
    page_url = "https://github.com/orgs/community/discussions/1"
    public = "https://private-user-images.githubusercontent.com/1/2/image.png"
    resolver._page_cache[page_url] = f'{{"src":\"{public}?jwt=fresh-token\"}}'
    refreshed = resolver._refresh_private_user_url(
        {"source_url": f"{public}?jwt=expired", "page_url": page_url}
    )
    assert refreshed == f"{public}?jwt=fresh-token"


def test_missing_linked_document_image_can_use_descriptive_pinned_alt_text(
    tmp_path: Path,
) -> None:
    class MissingRemoteResolver(ImageResolver):
        def _read_remote(self, url: str) -> tuple[bytes, str]:
            raise HTTPError(url, 404, "missing", {}, None)

    config = ProjectImageConfig(
        project="prisma",
        source_root=tmp_path / "source",
        image_base_url="https://www.prisma.io",
        routes=(),
    )
    resolver = MissingRemoteResolver(
        {"prisma": config}, tmp_path / "cache", allow_remote=True
    )
    record = resolver.resolve(
        {
            "image_id": "img-alt",
            "question_id": "prisma::1",
            "project": "prisma",
            "role": "linked_document",
            "doc_id": "prisma::/docs",
            "position": 1,
            "source_url": "/img/missing.png",
            "alt": "A diagram showing the complete migration sequence.",
            "title": "",
            "document_source_path": "apps/docs/content/docs/index.mdx",
        }
    )
    assert record["status"] == "text_fallback"
    assert record["fallback_kind"] == "pinned_document_alt_text"
    assert record["pixel_verified"] is False
